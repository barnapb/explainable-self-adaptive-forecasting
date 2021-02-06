import streamlit as st
import shap
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import xgboost
import json
import datetime
import time
import base64
from pathlib import Path
from smdebug.trials import create_trial
from smdebug.rules.rule_invoker import invoke_rule
from smdebug.exceptions import NoMoreData
from smdebug.rules.rule import Rule
from matplotlib.animation import FuncAnimation
import boto3
import io
import collections
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from streamlit import caching
import graphviz as graphviz
from PIL import Image
from streamlit import caching



#@st.cache(suppress_st_warning=True)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


class WeightRatio(Rule):
    def __init__(self, base_trial, ):
        super().__init__(base_trial)  
        self.tensors = collections.OrderedDict()  
    
    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(regex='.*weight'):
            if "gradient" not in tname:
                if "lstm" in tname:
                    try:
                        tensor = self.base_trial.tensor(tname).value(step)
                        if tname not in self.tensors:
                            self.tensors[tname] = {}

                        #st.write(f" Tensor  {tname}  has weights with variance: {np.var(tensor.flatten())} ")
                        self.tensors[tname][step] = tensor
                    except:
                        self.logger.warning(f"Can not fetch tensor {tname}")
                        #st.write(f"Can not fetch tensor {tname}")
        return False


class GradientsLayer(Rule):
    def __init__(self, base_trial):
        super().__init__(base_trial)  
        self.tensors = collections.OrderedDict()  
        
    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(regex='.*gradient'):
            if "lstm" in tname:
                try:
                    tensor = self.base_trial.tensor(tname).value(step)
                    if tname not in self.tensors:
                        self.tensors[tname] = {}

                    #self.logger.info(f" Tensor  {tname}  has gradients range: {np.min(tensor)} {np.max(tensor)} ")
                    self.tensors[tname][step] = tensor
                except:
                    self.logger.warning(f"Can not fetch tensor {tname}")
        return False
    
#@st.cache(allow_output_mutation=True)
def load_predictions(bucket, s3_key):
    
    
    s3 = boto3.client('s3')
    
    #obj_pred = s3.get_object(Bucket='prototype-ml-veolia-ro-ra', Key='data-channels/deepar/predictions/deepAR_predictions_20201006113127.csv')
    obj_pred = s3.get_object(Bucket=bucket, Key=s3_key)
    
    df_pred = pd.read_csv(io.BytesIO(obj_pred['Body'].read()), sep=",", index_col=0,parse_dates=True,dayfirst=True)
    
    list_col_mean = df_pred.filter(regex="_mean").columns
    df_pred_mean = df_pred.loc[:, df_pred.columns.isin(list_col_mean)]

    list_col_low = df_pred.filter(regex="_low_quantile").columns
    df_pred_low = df_pred.loc[:, df_pred.columns.isin(list_col_low)]

    list_col_up = df_pred.filter(regex="_up_quantile").columns
    df_pred_up = df_pred.loc[:, df_pred.columns.isin(list_col_up)]
        
    
    return df_pred,df_pred_mean,df_pred_low,df_pred_up


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_input_data(bucket, s3_key):
    
    s3 = boto3.client('s3')
    
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    df_ = pd.read_csv(io.BytesIO(obj['Body'].read()), sep=",", index_col=0,parse_dates=True)
    
   
    col_list = df_.columns
    col_list_target = [x.split('_')[0] for x in col_list]
    col_list_pivot = [x.split('_')[1] for x in col_list]
    
    return df_, col_list, list(set(col_list_target)), list(set(col_list_pivot))
    
@st.cache(allow_output_mutation=True)
def panda_profiler(df_features):
    pr = ProfileReport(df_features, explorative=True)
    return pr

#@st.cache
def shap_0(df_features, shap_values_xgb, X):
    
            
    date = st.date_input("Select date", value=df_features.index[0], min_value=df_features.index[0], max_value=df_features.index[-1])
    date = date.strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            
    data_point = df_features.index.get_loc(date)
      
    st_shap(shap.force_plot(shap_values_xgb.base_values[data_point], shap_values_xgb.values[data_point,:], X.iloc[data_point,:]),150)
    
    st.write("___")
    st.markdown("Visualize feature attributions such as Shapley values as forces. Each feature value is a force that either increases or decreases the prediction. The prediction starts from the baseline. The baseline for Shapley values is the average of all predictions. In the plot, each Shapley value is an arrow that pushes to increase (positive value) or decrease (negative value) the prediction. These forces balance each other out at the actual prediction of the data instance.")
    st.write("___")
#@st.cache(suppress_st_warning=True)
def shap_1(shap_values_xgb):
    
    #st.pyplot(shap.plots.bar(shap_values_xgb), clear_figure=False, bbox_inches='tight',dpi=500,pad_inches=1)
    fig = shap.plots.bar(shap_values_xgb)
    fname = "shap_summary.png"
    plt.savefig(fname, bbox_inches="tight",pad_inches=0.5)
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(fname))
    
    st.markdown(
                header_html, unsafe_allow_html=True,
                )
    
    st.write("___")
    st.markdown("For obtaining the global importance, the absolute Shapley values per feature across the data are averaged, and the features sorted by decreasing importance. SHAP feature importance is an alternative to permutation feature importance. There is a big difference between both importance measures: Permutation feature importance is based on the decrease in model performance. SHAP is based on magnitude of feature attributions. The feature importance plot is useful, but contains no information beyond the importances.")
    st.write("___")
    
#@st.cache(suppress_st_warning=True)
def shap_2(shap_values_xgb):
    

    #st.pyplot(shap.plots.beeswarm(shap_values_xgb), clear_figure=False, bbox_inches='tight',dpi=500,pad_inches=1)
    fig = shap.plots.beeswarm(shap_values_xgb)
    fname = "shap_summary.png"
    plt.savefig(fname, bbox_inches="tight",pad_inches=0.5)
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(fname))
    
    st.markdown(
                header_html, unsafe_allow_html=True,
                )
    #plt.clf()
    st.write("___")
    st.markdown("The summary plot combines feature importance with feature effects. Each point on the summary plot is a Shapley value for a feature and an instance. The position on the y-axis is determined by the feature and on the x-axis by the Shapley value. The color represents the value of the feature from low to high. Overlapping points are jittered in y-axis direction, so we get a sense of the distribution of the Shapley values per feature. The features are ordered according to their importance.")
    st.write("___")

#@st.cache(suppress_st_warning=True)
def shap_3(shap_values_xgb):    
    
       
    #st.pyplot(shap.plots.heatmap(shap_values_xgb), bbox_inches='tight',dpi=500,pad_inches=1)
    fig = shap.plots.heatmap(shap_values_xgb)
    fname = "shap_summary.png"
    plt.savefig(fname, bbox_inches="tight",pad_inches=0.5)
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(fname))
    st.markdown(
                header_html, unsafe_allow_html=True,
                )
    
    st.write("___")
    st.markdown("The output of the model is shown above the heatmap matrix (centered around the explaination’s base value), and the global importance of each model input shown as a bar plot on the right hand side of the plot (by default this is the absolute mean measure of overall importance).")
    st.write("___")

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

#@st.cache(suppress_st_warning=True)
def shap_4(shap_values_xgb, X): 
    
    feature = st.selectbox('Select feature',(shap_values_xgb.feature_names))
    
    fig = shap.dependence_plot(feature, shap_values_xgb.values, X)
    fname = "shap_summary.png"
    plt.savefig(fname, bbox_inches="tight",pad_inches=0.5)
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(fname))
    st.markdown(
                header_html, unsafe_allow_html=True,
                )
    
    st.write("___")
    st.markdown("SHAP feature dependence might be the simplest global interpretation plot: For the selected feature {},  for each data instance, a point is plotted with the feature value on the x-axis and the corresponding Shapley value on the y-axis.nVertical dispersion at a single value of {}, represents the interaction effects with other features. To help reveal these interactions, another feature is selected for coloring".format(feature,feature))
    st.write("___")

#@st.cache(suppress_st_warning=True)
def shap_5(shap_values_xgb, X): 
            
    N_ROWS = shap_values_xgb.shape[0]
    N_SAMPLES = min(100, N_ROWS)
    sampled_indices = np.random.randint(N_ROWS, size=N_SAMPLES)

    st_shap(shap.force_plot(shap_values_xgb.base_values[0],
                            shap_values_xgb.values[sampled_indices, :],
                            X.iloc[sampled_indices, :],
                            link='logit'),500)
    
    st.write("___")
    st.markdown("SHAP clustering works by clustering on Shapley values of each instance. This means that clustering instances by explanation similarity. All SHAP values have the same unit -- the unit of the prediction space. The figure above uses hierarchical agglomerative clustering to order the instances. The plot consists of many force plots, each of which explains the prediction of an instance, by rotating the force plots vertically and placing them side by side according to their clustering similarity. Each position on the x-axis is an instance of the data. Red SHAP values increase the prediction, blue values decrease it.")
    st.write("___")
    
#@st.cache(suppress_st_warning=True)
def shap_6(shap_values_xgb, shap_interaction_values, X,y, df):
    
    #st.pyplot(shap.plots.bar(shap_values_xgb,clustering=clustering), clear_figure=False, bbox_inches='tight',dpi=500,pad_inches=1)
    
    value = st.slider('Select number of data points before and after',
                        min_value = 1, max_value = 100, value=10, step=1)
    
        
    start_ = datetime.datetime(int(df.index[value].year), int(df.index[value].month), int(df.index[value].day),int(df.index[value].hour),int(df.index[value].minute),int(df.index[value].second))

    end_ = datetime.datetime(int(df.index[-value].year), int(df.index[-value].month), int(df.index[-value].day),int(df.index[-value].hour),int(df.index[-value].minute),int(df.index[-value].second))
    date_ = st.slider("Select date", value=start_, min_value=start_, max_value=end_)
    
    date_ = date_.strftime("%Y-%m-%d %H:%M:%S")
    date_ = datetime.datetime.strptime(date_, '%Y-%m-%d %H:%M:%S')
    
    date_index = df.index.get_loc(date_)
    
    features = X.iloc[date_index-value:date_index+value]

    
    fig = shap.decision_plot(shap_values_xgb.base_values[0],shap_interaction_values[date_index-value:date_index+value],features)
    fname = "shap_summary.png"
    plt.savefig(fname, bbox_inches="tight",pad_inches=0.5)
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(fname))
    st.markdown(
                header_html, unsafe_allow_html=True,
                )
    
    st.write("___")
    st.markdown("SHAP decision plots highlight how the model generates its predictions (i.e., how the splits are made). ")
    st.write("___")

#@st.cache(suppress_st_warning=True)
def shap_7(shap_interaction_values, X):
     
        
    #st.pyplot(shap.summary_plot(shap_interaction_values, X), clear_figure=False, bbox_inches='tight',dpi=500,pad_inches=1)
    
    fig = shap.summary_plot(shap_interaction_values, X)
    fname = "shap_summary.png"
    plt.savefig(fname, bbox_inches="tight",pad_inches=0.5)
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(fname))
    st.markdown(
                header_html, unsafe_allow_html=True,
                )
    
    st.write("___")
    st.markdown("SHAP interaction values are a generalization of SHAP values to higher order interactions. The proxy model returns a matrix for every prediction, where the main effects are on the diagonal and the interaction effects are off-diagonal. The main effects are similar to the SHAP values you would get for a linear model, and the interaction effects captures all the higher-order interactions and divides them up among the pairwise interaction terms. Note that the sum of the entire interaction matrix is the difference between the model's current output and expected output, and so the interaction effects on the off-diagonal are split in half (since there are two of each). When plotting interaction effects the SHAP package automatically multiplies the off-diagonal values by two to get the full interaction effect.")
    st.write("___")

#@st.cache(suppress_st_warning=True)
def shap_8(shap_values_xgb, shap_interaction_values, X):
    
    feature = st.selectbox('Select feature',(shap_values_xgb.feature_names))
            
    for feat in shap_values_xgb.feature_names:
        if not feat==feature:
            
            #st.pyplot(shap.dependence_plot((feature, feat),shap_interaction_values, X), clear_figure=False, bbox_inches='tight',dpi=500,pad_inches=1)
            fig = shap.dependence_plot((feature, feat),shap_interaction_values, X)
            fname = "shap_summary.png"
            plt.savefig(fname, bbox_inches="tight",pad_inches=0.5)
            header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
            img_to_bytes(fname))
            st.markdown(
                header_html, unsafe_allow_html=True,
                )

    st.write("___")
    st.markdown("Generating a dependence plot on the SHAP interaction values, allows to separately observe the main effects and the interaction effects. Plotting the main effects for {} and some of the interaction effects for {}. The main effects plot has no vertical dispersion because the interaction effects are all captured in the off-diagonal terms.".format(feature,feature))
    st.write("___")

@st.cache(suppress_st_warning=True)
def shap_init(bucket, option_pivot,option_target):
    
    # load dummy shap explainer
    X,y = shap.datasets.boston()
    model_xgb = xgboost.XGBRegressor(n_estimators=5, max_depth=2).fit(X, y)
    background = shap.maskers.Independent(X, max_samples=10)
    explainer_xgb = shap.TreeExplainer(model_xgb, background)
    shap_values_xgb = explainer_xgb(X)
    
    # load corresponding shap file
    shap_key = "easf-artefacts/shap/" + "xgboost_shap_" + option_pivot + "_" + option_target + ".json"
        
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=shap_key)
    content = obj['Body']
    jsonObject = json.loads(content.read())
    
    shap_values_xgb.values=np.array(jsonObject['shap_values'])
    shap_values_xgb.base_values=np.array(jsonObject['shap_base_values'])
    shap_values_xgb.data=np.array(jsonObject['shap_data'])
    shap_values_xgb.feature_names=jsonObject['shap_feature_names']
    clustering=np.array(jsonObject['shap_clustering'])
    y=np.array(jsonObject['y'])
    
        
    shap_interaction_values=np.array(jsonObject['shap_interaction_values'])
        
    X=pd.DataFrame(np.array(jsonObject['X']),columns=jsonObject['shap_feature_names'])
    
    return X,y, shap_values_xgb, shap_interaction_values, clustering

@st.cache(allow_output_mutation=True)
def load_trial_tensors(s3_debugger_key):
    
    trial_deepAR = create_trial(s3_debugger_key)
    
    rule_weights = rule_weights_invoke(trial_deepAR)
    rule_gradients = rule_gradients_invoke(trial_deepAR)
    
    return rule_weights, rule_gradients

@st.cache(allow_output_mutation=True)
def tensor_df(tensors, tname):
    
    steps = list(tensors[tname].keys())
                
    hist_data =[]
    group_labels=[]

    for step in steps:
        if step > 0: 
            x = tensors[tname][step].flatten()
            hist_data.append(x)
            group_labels.append(str("step_{}".format(step)))

    return hist_data, group_labels

@st.cache(allow_output_mutation=True)
def rule_weights_invoke(trial_deepAR):

    rule_weights = WeightRatio(trial_deepAR)
    try:
        with st.spinner('Wait to load the tensors...'):
            invoke_rule(rule_weights)
    except NoMoreData:
        pass
    
    return rule_weights
        
@st.cache(allow_output_mutation=True)
def rule_gradients_invoke(trial_deepAR):

    rule_gradients = GradientsLayer(trial_deepAR)
    try:
        with st.spinner('Wait to load the tensors...'):
            invoke_rule(rule_gradients)
    except NoMoreData:
        pass
    
    return rule_gradients

def invoke_predictions_lambda(payload, lambda_client, function_name):

    binary = json.dumps(payload)

    invoke_response = lambda_client.invoke(FunctionName=function_name,
                                            Payload=binary,
                                            InvocationType='RequestResponse')

    result_deepar = invoke_response['Payload'].read()

    if "errorMessage" in json.loads(result_deepar):
        return json.loads(result_deepar)['errorMessage']
    else:
        return json.loads(result_deepar)['status']

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data_init(real_config):

    df_raw, _, _, _ = load_input_data(bucket=real_config['BucketName'], s3_key=real_config['EnhancedDatasetKeyTarget'])
    df_features, col_list, col_list_target, col_list_pivot = load_input_data(bucket=real_config['BucketName'], s3_key=real_config['EnhancedDatasetKeyExpSmooth'])
    df_od_score, _, _, _ = load_input_data(bucket=real_config['BucketName'], s3_key=real_config['EnhancedDatasetKeyODLabel'])
    df_od_labels, _, _, _ = load_input_data(bucket=real_config['BucketName'], s3_key=real_config['EnhancedDatasetKeyODLabelAll'])
    
    return df_raw, df_features, col_list, col_list_target, col_list_pivot, df_od_score, df_od_labels

#@st.cache(allow_output_mutation=True)
def overwrite_config(BucketName,config_key,temp_config): 
    
    s3 = boto3.resource('s3')
    
    temp_config['ProcessingContainerArguments'][1] = str(temp_config['TargetVariablesList'])
    temp_config['ProcessingContainerArguments'][3] = str(temp_config['TimeseriesFreq'])
    temp_config['ProcessingContainerArguments'][5] = str(temp_config['PivotColumn'])
    temp_config['ProcessingContainerArguments'][7] = str(temp_config['TimestampColumn'])
    temp_config['ProcessingContainerArguments'][9] = str(temp_config['InterpolationOrder'])
    temp_config['ProcessingContainerArguments'][11] = str(temp_config['InterpolationMethod'])
    temp_config['ProcessingContainerArguments'][13] = str(temp_config['PredictionLength'])
    temp_config['ProcessingContainerArguments'][15] = str(temp_config['StepSize'])
    temp_config['ProcessingContainerArguments'][17] = str(temp_config['TestDataSize'])
    temp_config['ProcessingContainerArguments'][19] = str(temp_config['MinSeasonalGuess'])
    temp_config['ProcessingContainerArguments'][21] = str(temp_config['MaxSeasonalGuess'])
    temp_config['ProcessingContainerArguments'][23] = str(temp_config['SeasonalSmoothingThresold'])
    temp_config['ProcessingContainerArguments'][25] = str(temp_config['WindowsFolds'])
    temp_config['ProcessingContainerArguments'][27] = str(temp_config['RollingWindow'])
    temp_config['ProcessingContainerArguments'][29] = str(temp_config['XgboostEstimators'])
    temp_config['ProcessingContainerArguments'][31] = str(temp_config['XgboostDepth'])
    temp_config['ProcessingContainerArguments'][33] = str(temp_config['XgboostShapSamples'])
    temp_config['ProcessingContainerArguments'][35] = str(temp_config['XgboostEta'])
    temp_config['ProcessingContainerArguments'][37] = str(temp_config['XgboostNjobs'])
    temp_config['ProcessingContainerArguments'][39] = str(temp_config['XgboostGamma'])
    temp_config['ProcessingContainerArguments'][41] = str(temp_config['FeatureExpOrder'])
    temp_config['ProcessingContainerArguments'][43] = str(temp_config['Contamination'])
    temp_config['ProcessingContainerArguments'][45] = str(temp_config['Nneighbors'])
    temp_config['ProcessingContainerArguments'][47] = str(temp_config['ShapInteractionFlag'])
    temp_config['ProcessingContainerArguments'][49] = str(temp_config['KmeansClusters'])
    temp_config['ProcessingContainerArguments'][51] = str(temp_config['KmeansIters'])
    temp_config['ProcessingContainerArguments'][53] = str(temp_config['AlgoSelection'])
    temp_config['ProcessingContainerArguments'][55] = str(temp_config['NeuralNchangepoints'])
    temp_config['ProcessingContainerArguments'][57] = str(temp_config['NeuralChangepointsRange'])
    temp_config['ProcessingContainerArguments'][59] = str(temp_config['NeuralYearlySeasonality'])
    temp_config['ProcessingContainerArguments'][61] = str(temp_config['NeuralWeaklySeasonality'])
    temp_config['ProcessingContainerArguments'][63] = str(temp_config['NeuralDailySeasonality'])
    temp_config['ProcessingContainerArguments'][65] = str(temp_config['NeuralNforecasts'])
    temp_config['ProcessingContainerArguments'][67] = str(temp_config['NeuralNlags'])

    
    s3object = s3.Object(BucketName, config_key)
    s3object.put(Body=(bytes(json.dumps(temp_config).encode('UTF-8'))))
    time.sleep(1)
    

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def read_config(BucketName,config_key): 
    
    s3 = boto3.resource('s3')
    content_object = s3.Object(BucketName, config_key)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    real_config = json.loads(file_content)
 
    return real_config

def pipeline_graph(placeholder, node, mode):
    
    graph = graphviz.Digraph(engine='sfdp')
    
    if mode=="False":
        graph.attr('node', shape='doublecircle', style='filled', color="lightpink")
        graph.node(node)
        graph.attr('node', shape='circle', style='filled', color="lightblue1")
        graph.edge('start', 'preprocessing')
        graph.edge('preprocessing','finished')
        graph.edge('preprocessing','failed')
        placeholder.graphviz_chart(graph, use_container_width=True)
    elif mode=="True":
        graph.attr('node', shape='doublecircle', style='filled', color="lightpink")
        graph.node(node)
        graph.attr('node', shape='circle', style='filled', color="lightblue1")
        graph.edge('start', 'preprocessing')
        graph.edge('preprocessing', 'training')
        graph.edge('preprocessing', 'tuning')
        graph.edge('training', 'evaluation')
        graph.edge('tuning', 'evaluation')
        graph.edge('evaluation', 'deployment')
        graph.edge('deployment', 'profiling')
        graph.edge('profiling', 'finished')
        graph.edge('profiling', 'failed')
        placeholder.graphviz_chart(graph, use_container_width=True)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def app_init():
    

    #region_name = boto3.session.Session().region_name

    BucketName = os.environ['BucketName']
    region_name = os.environ['Region']
    FunctionName = os.environ['UtilityLambdaARN']
    
    config_key= "config/parameters.json"
    
    lambda_client = boto3.client('lambda', region_name)

    real_config = read_config(BucketName,config_key)


    if BucketName not in real_config['BucketName']:

        real_config['BucketName']= BucketName
        real_config['TrainDataInputPathDeepAR'] = "s3://{}/{}".format(BucketName,real_config['TrainDataInputPathDeepAR'])
        real_config['TestDataInputPathDeepAR'] = "s3://{}/{}".format(BucketName,real_config['TestDataInputPathDeepAR'] )
        real_config['OutputPathDeepAR'] = "s3://{}/{}".format(BucketName,real_config['OutputPathDeepAR'] )
        real_config['TrainDataInputPathGluonTS'] = "s3://{}/{}".format(BucketName,real_config['TrainDataInputPathGluonTS'])
        real_config['TestDataInputPathGluonTS'] = "s3://{}/{}".format(BucketName,real_config['TestDataInputPathGluonTS'])
        real_config['S3OutputPathGluonTSDebugger'] = "s3://{}/{}".format(BucketName,real_config['S3OutputPathGluonTSDebugger'])
        real_config['S3OutputPathGluonTS'] = "s3://{}/{}".format(BucketName,real_config['S3OutputPathGluonTS'])
        real_config['RawSourceFile'] = "s3://{}/{}".format(BucketName,real_config['RawSourceFile'])
        real_config['ProcessingOutputTrain'] = "s3://{}/{}".format(BucketName,real_config['ProcessingOutputTrain'])
        real_config['ProcessingOutputTest'] = "s3://{}/{}".format(BucketName,real_config['ProcessingOutputTest'])
        real_config['ProcessingOutputScaler'] = "s3://{}/{}".format(BucketName,real_config['ProcessingOutputScaler'])
        real_config['ProcessingOutputFeatures'] = "s3://{}/{}".format(BucketName,real_config['ProcessingOutputFeatures'])
        real_config['ProcessingOutputShap'] = "s3://{}/{}".format(BucketName,real_config['ProcessingOutputShap'])
        real_config['ProcessingCode'] = "s3://{}/{}".format(BucketName,real_config['ProcessingCode'])
        real_config['GluonTScodePath'] = "s3://{}/{}".format(BucketName,real_config['GluonTScodePath'])


        overwrite_config(BucketName,config_key,real_config)

    return BucketName, config_key, real_config, lambda_client, FunctionName

@st.cache(allow_output_mutation=True)
def get_gif():
    
    file_ = open("src/images/logo.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    
    return data_url

def main():

    st.set_option('deprecation.showPyplotGlobalUse', False)
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_page_config(page_title="Explainable and Self-Adaptive Forecasting App",initial_sidebar_state="expanded")
    
    #st.markdown('<style>body{background-color: white;}</style>',unsafe_allow_html=True)
    st.markdown('<style>.css-1aumxhk {background-color: #F0F2F6;background-image: none; color: #262730}}</style>',unsafe_allow_html=True)
    st.markdown('<style>.css-2trqyj {background-color: rgb(255, 182, 193); font-weight: 100; line-height: 1.6; color: white;}}</style>',unsafe_allow_html=True)
    #set_png_as_page_bg('src/images/logo_img.png')
   
    shap.initjs()
    BucketName, config_key, real_config, lambda_client, FunctionName = app_init()

    logo1, logo2 = st.sidebar.beta_columns((1, 3))
    logo1.image("src/images/aws.png",use_column_width=True) 
    #st.sidebar.image("src/images/logo.gif")
    
    data_url = get_gif()

    st.sidebar.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="256">',unsafe_allow_html=True)
    #st.sidebar.markdown('<img src="src/images/logo.gif"/>', unsafe_allow_html=True)
    

    # load features
    if real_config['PipelineRun']=="True":
        df_raw, df_features, col_list, col_list_target, col_list_pivot, df_od_score, df_od_labels = load_data_init(real_config)


    # TimeStamp to generate unique job names
    TS = datetime.datetime.strftime(datetime.datetime.utcnow(), '%Y-%m-%d-%H-%M-%S')


    # main menu
    app_mode = st.sidebar.selectbox("Choose the view layout",
        ["Home","Pipeline","Configuration","Profiling","Features", "SHAP", "Tensors", "Forecasts"])

    #st.sidebar.write("___")

    if app_mode == "Home":

        logo1, gif1, logo2, gif2, logo3, gif3, logo4 = st.beta_columns((2, 1, 2, 1, 2, 1, 2))
        gif1.text("")
        gif2.text("")
        gif3.text("")
        gif1.image("src/images/action.gif",use_column_width=True)
        gif2.image("src/images/action.gif",use_column_width=True)
        gif3.image("src/images/action.gif",use_column_width=True)
        logo2.subheader("Explainable")
        logo2.text("models")
        logo1.subheader("Self-Adaptive")
        logo1.text("covariates")
        logo3.subheader("Neural")
        logo3.text("forecasting")
        logo4.subheader("Probabilistic")
        logo4.text("estimates")
        st.write("")
        #st.code(">> all  the  'cool'  stuff  and  x >> 1  'extra'.")
        #st.write("___")
        st.write("")
        st.write("")
        st.write("")
        st.write("___")
        
        home_1 = st.beta_expander('Key Features')
        home_2 = st.beta_expander("What | How It's Solved")
        home_3 = st.beta_expander("Layouts")
        home_4 = st.beta_expander("Author | Team")
        home_5 = st.beta_expander("Upcoming Features")
        home_6 = st.beta_expander("Cloud Architecture")
        
        home_4.subheader("Paul Barna")
        home_4.text("Senior ML Scientist | Prototyping Architect")
        home_4.markdown("With a background in Modelling and Simulation, mostly intersted into the fields of Optimization and Optimal Control.")
        home_4.write("[LinkedIn](https://www.linkedin.com/in/paul-otniel-barna-05a02296/)")
        home_4.write("[EMEA Prototyping Team](https://w.amazon.com/bin/view/AWS_EMEA_Prototyping_Labs)")
        home_4.write("___")
        
        home_1.subheader("Model Interpretability")
        home_1.text("Tailored SHAP")
        home_1.write("___")
        home_1.markdown("Explaining feature importance and accumulated local effects, using SHAP, a game theoretic approach to explain the output of any machine learning model.")
        home_1.markdown("By using the XGBoost surrogate model to compute the full SHAP values, the user would be able to run a comprehensive set of analysis and understand both global and local contributions for each sample-prediction pair.")
        home_1.markdown("By capturing tensors while training (gradients and weights distributions), a complete understanding is achieved on how the network performs and how to tune it accordingly.")
        home_1.write("___")
        
        home_1.subheader("Self-Adaptive Covariates Subspaces")
        home_1.text("Novelty")
        home_1.write("___")
        home_1.markdown("A novel approach for use cases where:")
        home_1.markdown("+ accumulated local effects [which grow over time] shall not be disregarded or attributed to noise")
        home_1.markdown("+ non-stationarity is a known factor (explicitly modelled as part of the null hypothesis)")
        home_1.markdown("+ group-level effects shall not be lost within the overall predictive distribution")
        home_1.markdown("+ missing data [large gaps] shall not compromise the learning of a progressive underlying process")
        home_1.markdown("+ the best target realization [univariately built multi-step projection] carries a high degree of predictability")
        home_1.write("___")
        
        home_1.subheader("Neural Forecasting")
        home_1.text("DeepAR repurposed")
        home_1.write("___")
        home_1.markdown("A state-of-the-art Autoregressive Recurrent Network for multi-step forecasts of multiple-related time series.")
        home_1.markdown("A neural approach for use cases where:")
        home_1.markdown("+ interested in the full predictive distribution, not just the single best realisation")
        home_1.markdown("+ there is significant uncertainty growth over time from the data")
        home_1.markdown("+ widely-varying scales which requires rescaling and velocity-based sampling")
        home_1.write("___")
        
        home_1.subheader("Probabilistic Estimates")
        home_1.text("DeepAR repurposed")
        home_1.write("___")
        home_1.markdown("Estimating the probability distribution of a time series’ future given its past. This is achieved in the form of Monte Carlo samples, by computing quantile estimates for all sub-ranges in the prediction horizon.")
        home_1.write("___")
        
        home_2.subheader("What")
        home_2.markdown("Overcoming the challenges of non-permanent missing data, while preserving the integrity of the signal, as well as the ideal univariately built multi-step projection (best realisation).")
        home_2.subheader("How")
        home_2.markdown("Using either spline interpolation, followed by a triple exponential smoothing (where the smoothing parameters are optimised for each series independently) or Neural Prophet (A Neural Network based Time-Series model) to generate a synthetic pair for each time-series. Additionally, the selected after-mentioned univariately built model is used to predict n steps ahead (n - prediction length) and thus prepare the series best realisation replica, which is further used as a covariate for each target, within the main forecasting setting.")
        home_2.write("___")
        
        home_2.subheader("What")
        home_2.markdown("Accumulated local effects which are deviant from the general data distribution could uncover a temporary or permanent change within the underlying processes.")
        home_2.subheader("How")
        home_2.markdown("Helping the model focus where the deviation has occurred as well as what has caused it, by building a supportive self-adapted covariates space. Outliers are detected using a constructed empirical copula to predict tail probabilities for each data point. Quantifying the abnormality contribution of each dimension within a newly designed signal for each predictor, would allow the model to focus on certain subspaces interactions. A heuristic contamination threshold is set on a global “extremeness” score computed with an Auto-Encoder setting, followed by a user-defined rolling window sum, and fed to the model as an additional covariate.")
        home_2.write("___")

        home_2.subheader("What")
        home_2.markdown("Allowing the model to learn item-specific behaviour or group-level effects would significantly improve the overall estimates distribution.")
        home_2.subheader("How")
        home_2.markdown("Grouping the time-series both by a user-specified category, as well as by the cluster it belongs to, after running a K-means time series clustering with dynamic time warping.")


        home_3.image("src/images/menu.png")
        home_3.subheader("Pipeline")
        home_3.text("Pipeline Flow")
        home_3.subheader("Configuration")
        home_3.text("Problem Formulation")
        home_3.subheader("Profiling")
        home_3.text("Exploratory Data Analysis")
        home_3.subheader("Features")
        home_3.text("Feature Engineering")
        home_3.subheader("SHAP")
        home_3.text("SHAP Analysis")
        home_3.subheader("Tensors")
        home_3.text("Tensors Analysis")
        home_3.subheader("Forecasts")
        home_3.text("Forecast Analysis")
        
        home_5.write("")
        home_5.text("Causality Simulation")
        home_5.text("Experiments Tracking")
        home_5.text("Additional Metrics")
        home_5.text("Stratified Sampling [PySpark Preprocessing Step]")
        st.write("")
        st.write("___")
        
        with open("src/images/EASF_diagram.pdf", 'rb') as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf">' 
        home_6.markdown(pdf_display, unsafe_allow_html=True)
        
        #home_6.image("src/images/diagram.png",width=1000)
        st.write("")
        st.write("___")
        

    elif app_mode == "Pipeline":

        with st.beta_container():

            mode="False"
            node="start"
            run_flag=False

            sf_payload = real_config.copy()
            sf_payload['ModelName'] = sf_payload['ModelName'] + "-" + TS
            sf_payload['TuningJobName'] = sf_payload['TuningJobName'] + "-" + TS
            sf_payload['TrainingJobNameDeepAR'] = sf_payload['TrainingJobNameDeepAR'] + "-" + TS
            sf_payload['TrainingJobNameGluonTS'] = sf_payload['TrainingJobNameGluonTS'] + "-" + TS
            sf_payload['ProcessingJobName'] = sf_payload['ProcessingJobName'] + "-" + TS

            mode = st.sidebar.selectbox('Enable Training',('False', 'True'))
            st.sidebar.write("___")
            
            if mode=="True":
                sf_payload['TrainingFlag']="True"
            else:
                sf_payload['TrainingFlag']="False"

            # check if pipeline already runnning
            sf_payload['StepFunctionAPI'] = "status"
            binary = json.dumps(sf_payload)

            invoke_response = lambda_client.invoke(FunctionName=FunctionName,
                                                                Payload=binary,
                                                                InvocationType='RequestResponse')

            response = invoke_response['Payload'].read()

            state_id = int(json.loads(response)['state_id'])
            response = json.loads(response)['response']
            
            if response=="execution in progress":
                run_flag=True
            
            st.sidebar.text("Run    |   Stop")
            run_pl, stop_pl = st.sidebar.beta_columns((1,4))
            
            placeholder_run = run_pl.empty()
            placeholder_stop = stop_pl.empty()
            placeholder = st.empty()
            placeholder_status = st.sidebar.empty()
            placeholder_status.text(response)
            palceholder_bar = st.sidebar.empty()
            
            pipeline_graph(placeholder, node=node, mode=mode)
            
            run_button = placeholder_run.button('>>')
            stop_button = placeholder_stop.button("<<")
            
            if run_button or run_flag:

                sf_payload['StepFunctionAPI'] = "invoke"

                binary = json.dumps(sf_payload)

                invoke_response = lambda_client.invoke(FunctionName=FunctionName,
                                                        Payload=binary,
                                                        InvocationType='RequestResponse')

                response = invoke_response['Payload'].read()
                failed_details = json.loads(response)['fail_details']
                response = json.loads(response)['response']

                placeholder_status.text(response)
                
                sf_payload['StepFunctionAPI'] = "delete"

                binary = json.dumps(sf_payload)

                invoke_response = lambda_client.invoke(FunctionName=FunctionName,
                                                        Payload=binary,
                                                        InvocationType='RequestResponse')

                sf_payload['StepFunctionAPI'] = "status"
                binary = json.dumps(sf_payload)

                node="start"

                pipeline_graph(placeholder, node=node, mode=mode)

                my_bar = palceholder_bar.progress(0)
                #st.markdown('<style>.st-fg {background-color: rgb(255, 182, 193)}</style>',unsafe_allow_html=True)
                st.markdown('<style>.stProgress > div > div > div > div {background-color: rgb(255, 182, 193);</style>',unsafe_allow_html=True)

                while True:

                    time.sleep(2)

                    invoke_response = lambda_client.invoke(FunctionName=FunctionName,
                                                                Payload=binary,
                                                                InvocationType='RequestResponse')

                    response = invoke_response['Payload'].read()

                    state_id = int(json.loads(response)['state_id'])
                    failed_details = json.loads(response)['fail_details']
                    response = json.loads(response)['response']

                    placeholder_status.text(response)

                    if mode=="True":

                        if state_id < 8 and response=="execution in progress":

                            node="preprocessing"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(0)

                        if state_id > 8 and state_id < 29 and response=="execution in progress":

                            node="training"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(25)

                        if state_id > 28 and state_id < 36 and response=="execution in progress":

                            node="evaluation"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(50)

                        if state_id > 35 and state_id < 46 and response=="execution in progress":

                            node="deployment"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(75)

                        if state_id > 45 and state_id < 50 and response=="execution in progress":

                            node="profiling"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(75)

                        if response=="NO execution in progress" or response=="execution succeeded":

                            node="finished"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(100)

                            placeholder_status.text("Execution finished")

                            real_config["PipelineRun"]="True"
                            real_config["TrainingJobNameGluonTSDebugger"]=sf_payload["TrainingJobNameGluonTS"]

                            overwrite_config(BucketName,config_key,real_config)
                            caching.clear_cache()

                            break
                        
                        if response=="execution failed" or response=="execution aborted":
                            
                            node="failed"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(100)

                            placeholder_status.text(failed_details)
                            
                            break

                    elif mode=="False":

                        if state_id < 8 and response=="execution in progress":

                            node="preprocessing"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(0)

                        if response=="NO execution in progress" or response=="execution succeeded":

                            node="finished"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(100)

                            placeholder_status.text("Execution finished")

                            real_config["PipelineRun"]="True"

                            overwrite_config(BucketName,config_key,real_config)
                            caching.clear_cache()
                            
                            break
                            
                        if response=="execution failed" or response=="execution aborted":
                            
                            node="failed"

                            pipeline_graph(placeholder, node=node, mode=mode)

                            my_bar.progress(100)

                            placeholder_status.text(failed_details)
                            
                            #option_="Idle"
                            
                            break
                            
                    if stop_button:

                        sf_payload['StepFunctionAPI']="stop"
                        binary = json.dumps(sf_payload)

                        placeholder_status.text("stopping")

                        invoke_response = lambda_client.invoke(FunctionName=FunctionName,
                                                                            Payload=binary,
                                                                            InvocationType='RequestResponse')

                        response = invoke_response['Payload'].read()

                        response = json.loads(response)['response']

                        placeholder_status.text(response)

                        node="failed"

                        pipeline_graph(placeholder, node=node, mode=mode)

                        time.sleep(5)
                        
                        break

                    time.sleep(10)
                        

    elif app_mode == "Configuration":
        
        st.sidebar.write("___")
        
        current_config=real_config.copy()
        

        with st.beta_expander('AWS Settings'):

            current_config['RawSourceFile'] = st.text_input("RawSourceFile", current_config['RawSourceFile'])
            current_config['ModelName'] = st.text_input("ModelName", current_config['ModelName'])
            current_config['EndpointName'] = st.text_input("EndpointName", current_config['EndpointName'])
            current_config['TuningJobName'] = st.text_input("TuningJobName", current_config['TuningJobName'])
            current_config['TrainingJobNameDeepAR'] = st.text_input("TrainingJobNameDeepAR", current_config['TrainingJobNameDeepAR'])
            current_config['TrainingJobNameGluonTS'] = st.text_input("TrainingJobNameGluonTS", current_config['TrainingJobNameGluonTS'])
            current_config['ProcessingJobName'] = st.text_input("ProcessingJobName", current_config['ProcessingJobName'])


        with st.beta_expander('hyperparameters Feature Enigneering'):

            current_config['TargetVariablesList'] = st.text_input("TargetVariablesList", current_config['TargetVariablesList'])
            current_config['PivotColumn'] = st.text_input("PivotColumn", current_config['PivotColumn'])
            current_config['TimestampColumn'] = st.text_input("TimestampColumn", current_config['TimestampColumn'])
            current_config['AlgoSelection'] = st.selectbox('AlgoSelection',('NeuralProphet', 'ExponentialSmoothing'))
            current_config['InterpolationOrder'] = st.number_input('InterpolationOrder',min_value=1,max_value=5,value=current_config['InterpolationOrder'])
            current_config['InterpolationMethod'] = st.selectbox('InterpolationMethod',('spline','linear', 'time','index','pad','nearest','zero','slinear','quadratic','cubic','barycentric','polynomial','krogh','piecewise_polynomial','pchip','akima','cubicspline','from_derivatives'))
            current_config['StepSize'] = st.number_input('StepSize',min_value=1,max_value=30,value=current_config['StepSize'])
            current_config['TestDataSize'] = st.number_input('TestDataSize',value=current_config['TestDataSize'])
            current_config['MinSeasonalGuess'] = st.number_input('MinSeasonalGuess',value=current_config['MinSeasonalGuess'])
            current_config['MaxSeasonalGuess'] = st.number_input('MaxSeasonalGuess',value=current_config['MaxSeasonalGuess'])
            current_config['SeasonalSmoothingThresold'] = st.number_input('SeasonalSmoothingThresold',min_value=0.01, max_value=1.0,value=current_config['SeasonalSmoothingThresold'],  format="%f", step=0.01)
            current_config['WindowsFolds'] = st.number_input('WindowsFolds',min_value=1,max_value=7,value=current_config['WindowsFolds'])
            current_config['RollingWindow'] = st.number_input('RollingWindow',value=current_config['RollingWindow'])
            current_config['FeatureExpOrder'] = st.number_input('FeatureExpOrder',min_value=1,max_value=5,value=current_config['FeatureExpOrder'])
            current_config['Contamination'] = st.number_input('Contamination',min_value=0.01, max_value=0.5,value=current_config['Contamination'],  format="%f", step=0.01)
            current_config['Nneighbors'] = st.number_input('Nneighbors',value=current_config['Nneighbors'])
            current_config['KmeansClusters'] = st.number_input('KmeansClusters',min_value=3,max_value=100,value=current_config['KmeansClusters'])
            current_config['KmeansIters'] = st.number_input('KmeansIters',min_value=1,max_value=100,value=current_config['KmeansIters'])
            current_config['NeuralNchangepoints'] = st.number_input('NeuralNchangepoints',value=current_config['NeuralNchangepoints'])
            current_config['NeuralChangepointsRange'] = st.number_input('NeuralChangepointsRange',min_value=0.00,max_value=1.00,value=current_config['NeuralChangepointsRange'])
            current_config['NeuralYearlySeasonality'] = st.selectbox('NeuralYearlySeasonality',('auto','True','False'))
            current_config['NeuralWeaklySeasonality'] = st.selectbox('NeuralWeaklySeasonality',('auto','True','False'))
            current_config['NeuralDailySeasonality'] = st.selectbox('NeuralDailySeasonality',('auto','True','False'))
            current_config['NeuralNforecasts'] = st.number_input('NeuralNforecasts',value=current_config['NeuralNforecasts'])
            current_config['NeuralNlags'] = st.number_input('NeuralNlags',value=current_config['NeuralNlags'])

        with st.beta_expander('hyperparameters DeepAR'):

            current_config['EpochsDeepAR'] = str(st.number_input('EpochsDeepAR',min_value=1,max_value=1000,value=int(current_config['EpochsDeepAR'])))
            current_config['PredictionLength'] = str(st.number_input('PredictionLength',value=int(current_config['PredictionLength'])))
            current_config['ContextLenght'] = str(st.number_input('ContextLenght',min_value=1,max_value=200,value=int(current_config['ContextLenght'])))
            current_config['NumLayersDeepAR'] = str(st.number_input('NumLayersDeepAR',min_value=1,max_value=8,value=int(current_config['NumLayersDeepAR'])))
            current_config['DropoutRateDeepAR'] = str(st.number_input('DropoutRateDeepAR',min_value=0.00, max_value=0.2,value=float(current_config['DropoutRateDeepAR'])))
            current_config['LearningRateDeepAR'] = str(st.number_input('LearningRateDeepAR',min_value=1e-4,max_value=1e-1,value=float(current_config['LearningRateDeepAR']), format="%f", step=0.001))
            current_config['TimeseriesFreq'] = st.selectbox('TimeseriesFreq',('D','H','W','M'))
            current_config['TrainInstanceTypeDeepAR'] = st.selectbox('TrainInstanceTypeDeepAR',('ml.c5.4xlarge', 'ml.c5.2xlarge','ml.c5.xlarge'))
            current_config['TrainInstanceCount'] = st.number_input('TrainInstanceCount',min_value=1,max_value=5,value=current_config['TrainInstanceCount'])
            current_config['NumCellsDeepAR'] = str(st.number_input('NumCellsDeepAR',min_value=30,max_value=100,value=int(current_config['NumCellsDeepAR'])))
            current_config['RMSEthresholdDeepAR'] = str(st.number_input('RMSEthresholdDeepAR',min_value=0.0, max_value=100.0,value=float(current_config['RMSEthresholdDeepAR']),  format="%f", step=0.1))
            current_config['Save_intervalGluonTS'] = str(st.number_input('Save_intervalGluonTS',min_value=1,max_value=1000,value=int(current_config['Save_intervalGluonTS'])))


        with st.beta_expander('hyperparameters Optimization'):

            current_config['HPOEnabled'] = st.selectbox('HPOEnabled',('False', 'True'))
            current_config['MaxJobsHPODeepAR'] = st.number_input('MaxJobsHPODeepAR',min_value=1,max_value=50,value=current_config['MaxJobsHPODeepAR'])
            current_config['MaxParallelJobsHPODeepAR'] = st.number_input('MaxJobsHPODeepAR',min_value=1,max_value=5,value=current_config['MaxParallelJobsHPODeepAR'])
            current_config['ObjectiveMetricDeepAR'] = st.selectbox('ObjectiveMetricDeepAR',('test:RMSE', 'test:mean_wQuantileLoss', 'train:final_loss'))
            current_config['LearningRateDeepAR_min'] = str(st.number_input('LearningRateDeepAR_min',min_value=1e-5,max_value=1e-1,value=float(current_config['LearningRateDeepAR_min']), format="%f", step=0.001))
            current_config['LearningRateDeepAR_max'] = str(st.number_input('LearningRateDeepAR_max',min_value=1e-5,max_value=1e-1,value=float(current_config['LearningRateDeepAR_max']), format="%f", step=0.001))
            current_config['EpochsDeepAR_min'] = str(st.number_input('EpochsDeepAR_min',min_value=1,max_value=1000,value=int(current_config['EpochsDeepAR_min'])))
            current_config['EpochsDeepAR_max'] = str(st.number_input('EpochsDeepAR_max',min_value=1,max_value=1000,value=int(current_config['EpochsDeepAR_max'])))
            current_config['NumCellsDeepAR_min'] = str(st.number_input('NumCellsDeepAR_min',min_value=30,max_value=200,value=int(current_config['NumCellsDeepAR_min'])))
            current_config['NumCellsDeepAR_max'] = str(st.number_input('NumCellsDeepAR_max',min_value=30,max_value=200,value=int(current_config['NumCellsDeepAR_max'])))
            current_config['NumLayersDeepAR_min'] = str(st.number_input('NumLayersDeepAR_min',min_value=1,max_value=8,value=int(current_config['NumLayersDeepAR_min'])))
            current_config['NumLayersDeepAR_max'] = str(st.number_input('NumLayersDeepAR_max',min_value=1,max_value=8,value=int(current_config['NumLayersDeepAR_max'])))
            current_config['ContextLenght_min'] = str(st.number_input('ContextLenght_min',min_value=1,max_value=500, value=int(current_config['ContextLenght_min'])))
            current_config['ContextLenght_max'] = str(st.number_input('ContextLenght_max',min_value=1,max_value=500, value=int(current_config['ContextLenght_max'])))


        with st.beta_expander('hyperparameters Surrogate Model'):
    
            current_config['ShapInteractionFlag'] = st.selectbox('ShapInteractionFlag',('True', 'False'))
            current_config['XgboostEstimators'] = st.number_input('XgboostEstimators',min_value=1,max_value=1000,value=current_config['XgboostEstimators'])
            current_config['XgboostDepth'] = st.number_input('XgboostDepth',min_value=1,max_value=6,value=current_config['XgboostDepth'])
            current_config['XgboostShapSamples'] = st.number_input('XgboostShapSamples',min_value=50,max_value=1000,value=current_config['XgboostShapSamples'],step=10)
            current_config['XgboostEta'] = st.number_input('XgboostEta',min_value=0.0, max_value=1.0,value=current_config['XgboostEta'])
            current_config['XgboostNjobs'] = st.number_input('XgboostNjobs',min_value=1,max_value=10,value=current_config['XgboostNjobs'])
            current_config['XgboostGamma'] = st.number_input('XgboostGamma',min_value=0.0,value=current_config['XgboostGamma'])

        
    
        if st.sidebar.button('submit'):
            
            overwrite_config(BucketName,config_key,current_config)
            #node="preprocessing"
            st.sidebar.text("Updates submitted successfully")
            caching.clear_cache()
            BucketName, config_key, real_config, lambda_client, FunctionName = app_init()
            df_raw, df_features, col_list, col_list_target, col_list_pivot, df_od_score, df_od_labels = load_data_init(real_config)

#         st.sidebar.write("___")
#         agree_5 = st.sidebar.checkbox('Current config values')
#         if agree_5:
#             st.write(real_config)


    elif app_mode == "Profiling" and real_config['PipelineRun']=="True":

        #st.sidebar.write("...data profiling storyline")
        with st.beta_container():

            option_pivot = st.sidebar.selectbox('Select category',(col_list_pivot))
            st.sidebar.write("___")
            frac_number = st.sidebar.number_input('Select sampling fraction', min_value=0.1, max_value=1.0,value=0.5,step=0.1)
            
            agree_raw_pr = st.checkbox("Raw Data Profiling")
            if agree_raw_pr:

                df_profiler = df_raw.sample(frac = frac_number).sort_index()

                col_list_ = df_profiler.filter(regex=option_pivot).columns
                df_profiler = df_profiler.loc[:, df_profiler.columns.isin(col_list_)]

                pr = panda_profiler(df_profiler)
                st_profile_report(pr)
            
            agree_syn_pr = st.checkbox("Syntetic Data Profiling")
            if agree_syn_pr:
    
                df_profiler_ = df_features.sample(frac = frac_number).sort_index()

                col_list__ = df_profiler_.filter(regex=option_pivot).columns
                df_profiler_ = df_profiler_.loc[:, df_profiler_.columns.isin(col_list__)]

                pr = panda_profiler(df_profiler_)
                st_profile_report(pr)


    elif app_mode == "Features" and real_config['PipelineRun']=="True":

        #st.sidebar.write("...storyline")
        with st.beta_container():

            option_pivot = st.sidebar.selectbox('Select category',(col_list_pivot))
            st.sidebar.write("___")

            col_list_raw = df_raw.filter(regex=option_pivot).columns
            df_raw_display = df_raw.loc[:, df_raw.columns.isin(col_list_raw)]

            col_list_fetaures = df_features.filter(regex=option_pivot).columns
            df_features_display = df_features.loc[:, df_features.columns.isin(col_list_fetaures)]

            col_list_od_labels = df_od_labels.filter(regex=option_pivot).columns
            df_od_labels_display = df_od_labels.loc[:, df_od_labels.columns.isin(col_list_od_labels)]
            df_od_labels_display = df_od_labels_display.add_suffix('_od_label')

            col_list_od_score = df_od_score.filter(regex=option_pivot).columns
            df_od_score_display = df_od_score.loc[:, df_od_score.columns.isin(col_list_od_score)]
            df_od_score_display = df_od_score_display.add_suffix('_abn_contrib')

            scaler_dynamic = StandardScaler()
            df_od_score_display[df_od_score_display.columns] = scaler_dynamic.fit_transform(df_od_score_display[df_od_score_display.columns])


            with st.beta_expander("Raw Data Frame"):
                st.subheader("Raw Data Frame")
                st.dataframe(df_raw_display.style.background_gradient(cmap='viridis'))
                
            with st.beta_expander("Syntetic Data Frame"):                 
                st.subheader("Syntetic Data Frame")
                st.dataframe(df_features_display.style.background_gradient(cmap='viridis'))

            frames = [df_raw_display, df_features_display, df_od_score_display, df_od_labels_display]
            df_result = pd.concat(frames, axis=1, sort=False)

            values = st.sidebar.slider(
                                'Select number of data points to display',
                                    min_value = 0, max_value = df_result.shape[0], value=200, step=100)

            df_result = df_result[-values:]

            with st.beta_expander("Plot Analysis"):

                option = st.multiselect(
                'Which feature to select?',default=[df_raw_display.columns[0], df_features_display.columns[0], df_od_score_display.columns[0], df_od_labels_display.columns[0]],
                    options = list(df_result.columns))

                if option:

                    fig_plotly = px.line(df_result, x=df_result.index, y=option, facet_row_spacing=0.01, render_mode="svg")
                    fig_plotly.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                    fig_plotly.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig_plotly,use_container_width=True)

                    #plot_exp.area_chart(df_result[option])
                


    elif app_mode == "SHAP" and real_config['PipelineRun']=="True":

        #with st.beta_container():
        
        shap_mode = st.sidebar.selectbox("Choose the shap layout",
            ["Local Explanation","SHAP Bar Plot", "Beeswarm Plot", "Heatmap", "Dependence Plots", "Force Plot", "Decision Plot","Interactions Summary Plot","Interactions Dependence Plots"])
        st.sidebar.write("___")

        option_pivot = st.sidebar.selectbox('Select category',(col_list_pivot))

        # select target
        option_target = st.sidebar.selectbox('Select target',(col_list_target))

        X,y, shap_values_xgb, shap_interaction_values, clustering = shap_init(BucketName, option_pivot,option_target)


        if shap_mode=="Local Explanation":
            shap_0(df_features, shap_values_xgb, X)


        elif shap_mode=="SHAP Bar Plot":
            shap_1(shap_values_xgb)

        elif shap_mode=="Beeswarm Plot":
            shap_2(shap_values_xgb)

        elif shap_mode=="Heatmap":
            shap_3(shap_values_xgb)

        elif shap_mode=="Dependence Plots":
            shap_4(shap_values_xgb, X)

        elif shap_mode=="Force Plot":
            shap_5(shap_values_xgb, X)

        elif shap_mode=="Decision Plot":
            shap_6(shap_values_xgb, shap_interaction_values, X,y, df_features)

        elif shap_mode=="Interactions Summary Plot":
            shap_7(shap_interaction_values, X)

        elif shap_mode=="Interactions Dependence Plots":
            shap_8(shap_values_xgb, shap_interaction_values, X)


    elif app_mode == "Tensors" and real_config['PipelineRun']=="True":
        # deepAr Tensors
        with st.beta_container():

            # load debuger tensors                               
            s3_debugger_key = "{}/{}/debug-output".format(real_config['S3OutputPathGluonTSDebugger'],real_config['TrainingJobNameGluonTSDebugger'])

            rule_weights, rule_gradients = load_trial_tensors(s3_debugger_key)

            tensor_mode = st.sidebar.selectbox("Choose the tensor layout", ["Weights","Gradients"])

            if tensor_mode=="Weights":

                tensors = rule_weights.tensors
                tnames = list(tensors.keys())
                tname = st.selectbox('Select tensor',(tnames))

                hist_data, group_labels = tensor_df(tensors, tname)

                # Create distplot
                fig = ff.create_distplot(hist_data, group_labels, show_hist=False,show_rug=False)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                fig.layout.update(title='Weight Density Curve')

                st.plotly_chart(fig)

                st.write("___")
                st.markdown("This rule retrieves the weight tensors and checks the variance. If the distribution does not change much across steps it may indicate that the learning rate is too low, that gradients are too small or that the training has converged to a minimum.")
                st.write("___")

            elif tensor_mode=="Gradients":
                #st.subheader("Gradients")

                tensors = rule_gradients.tensors
                tnames = list(tensors.keys())
                tname = st.selectbox('Select tensor',(tnames))


                hist_data, group_labels = tensor_df(tensors, tname)

                # Create distplot
                fig = ff.create_distplot(hist_data, group_labels, show_hist=False,show_rug=False, curve_type="normal")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                fig.layout.update(title='Gradients Density Curve')
                st.plotly_chart(fig)
                
                st.write("___")
                st.markdown("The rule retrieves the gradients and plots their distribution. If variance is tiny, that means that the model parameters do not get updated effectively with each training step or that the training has converged to a minimum.")
                st.write("___")

    elif app_mode == "Forecasts" and real_config['PipelineRun']=="True":

        with st.beta_container():

            train_start=real_config['TrainDatasetStart']
            forecast_day=real_config['ForecastDay']

            option_pivot = st.sidebar.selectbox('Select category',(col_list_pivot))
            Confidence = st.number_input('Confidence',min_value=1,max_value=100, value=80)


            start_dataset = pd.Timestamp(df_features.index[0], freq=real_config['TimeseriesFreq'])
            end_forecast = pd.Timestamp(df_features.index[-1], freq=real_config['TimeseriesFreq'])
            end_training = end_forecast - ((real_config['WindowsFolds'] + 1)*int(real_config['PredictionLength']))*end_forecast.freq
            end_forecast = end_forecast - int(real_config['PredictionLength']) * end_forecast.freq
            end_context = end_training - int(real_config['ContextLenght']) * end_forecast.freq
            start_forecast = end_training + end_forecast.freq

            # select target
            option_target = st.sidebar.selectbox('Select target',(col_list_target))
            
            ph_metrics = st.sidebar.empty()
            
            start_dataset_ = datetime.datetime(int(start_dataset.year), int(start_dataset.month), int(start_dataset.day),int(start_dataset.hour),int(start_dataset.minute),int(start_dataset.second))

            end_context_ = datetime.datetime(int(end_context.year), int(end_context.month), int(end_context.day),int(end_context.hour),int(end_context.minute),int(end_context.second))

            start_forecast_ = datetime.datetime(int(start_forecast.year), int(start_forecast.month), int(start_forecast.day),int(start_forecast.hour),int(start_forecast.minute),int(start_forecast.second))

            end_forecast_ = datetime.datetime(int(end_forecast.year), int(end_forecast.month), int(end_forecast.day),int(end_forecast.hour),int(end_forecast.minute),int(end_forecast.second))

            train_start = st.slider("Select context start date", value=start_dataset_, min_value=start_dataset_, max_value=end_context_)
            forecast_day = st.slider("Select forecast start date", value=start_forecast_, min_value=start_forecast_, max_value=end_forecast_)

            train_start = train_start.strftime("%Y-%m-%d %H:%M:%S")
            train_start = datetime.datetime.strptime(train_start, '%Y-%m-%d %H:%M:%S')

            forecast_day = forecast_day.strftime("%Y-%m-%d %H:%M:%S")
            forecast_day = datetime.datetime.strptime(forecast_day, '%Y-%m-%d %H:%M:%S')

            real_config["Confidence"]=str(Confidence)
            real_config["TrainDatasetStart"]=str(train_start)
            real_config["ForecastDay"]=str(forecast_day)

            option_regex = option_target + "_" + option_pivot

            df_features_ = df_raw.filter(regex=option_regex)

            train_start_point = df_features_.index.get_loc(train_start)
            forecast_day_point = df_features_.index.get_loc(forecast_day)

            df_temp_ = df_features_.loc[df_features_.index[train_start_point]:df_features_.index[forecast_day_point]]
            df_temp_ = df_temp_.add_suffix('_selected')

            frames_ = [df_features_, df_temp_]
            df_result_ = pd.concat(frames_, axis=1, sort=False)

            plot_holder = st.empty()

            fig_plotly = px.line(df_result_,x=df_result_.index, y=df_result_.columns, facet_row_spacing=1, render_mode="svg", color_discrete_sequence=["lightsalmon", "firebrick"])
            fig_plotly.update_xaxes(rangeslider_visible=True)
            fig_plotly.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            plot_holder.plotly_chart(fig_plotly,use_container_width=True)
             
            st.sidebar.write("___")
            #st.sidebar.text("Endpoint API")
            st.sidebar.text("Invoke | Delete")
            col1, col2 = st.sidebar.beta_columns((1,4))
            
            if col1.button('>>'):
                with st.spinner('Serving Predictions -- if it takes longer than 1 min, it means the serve function is waiting for the endpoint to be created/updated'):

                    response = invoke_predictions_lambda(real_config, lambda_client, function_name=os.environ['ServeLambdaARN'])

                    if response=='Serving Request Succesful':

                        overwrite_config(BucketName,config_key,real_config)

                        df_pred,df_pred_mean,df_pred_low,df_pred_up = load_predictions(bucket=BucketName, s3_key=real_config['PredictionsDeepARKey'])

                        st.sidebar.text(response)
                        
                        real_config['PredictionFlag']="True"
                        
                        overwrite_config(BucketName,config_key,real_config)
                    else:
                        
                        st.sidebar.text("endpoint not running")
            
            if col2.button('<<'):          
                
                mt_payload['StepFunctionAPI'] = "delete"

                binary = json.dumps(mt_payload)

                invoke_response = lambda_client.invoke(FunctionName=FunctionName,
                                                        Payload=binary,
                                                        InvocationType='RequestResponse')
                
                confirm_msg = invoke_response['Payload'].read()

                confirm_msg = json.loads(confirm_msg)['response']
                
                st.sidebar.text(confirm_msg)
                
            if real_config['PredictionFlag']=="True":

                df_pred,df_pred_mean,df_pred_low,df_pred_up = load_predictions(bucket=BucketName, s3_key=real_config['PredictionsDeepARKey'])
                
                df_temp = df_features.loc[df_pred.index[0]:df_pred.index[-1]]

                df_temp.columns = df_temp.columns.str.replace(r'_dynamicFeat', '')

                option_temp = list(df_temp.filter(regex=option_regex).columns)
                option_mean = list(df_pred_mean.filter(regex=option_regex).columns)
                option_low = list(df_pred_low.filter(regex=option_regex).columns)
                option_high = list(df_pred_up.filter(regex=option_regex).columns)

                st.subheader("Quantiles Close-up")

                fig_plotly_ = go.Figure()
                fig_plotly.add_trace(go.Scatter(x=df_pred_mean.index, y=df_pred_mean[option_mean[0]].tolist(), name=str(option_mean[0]), mode='lines', 
                                                            line = dict(color='firebrick')))
                fig_plotly.add_trace(go.Scatter(x=df_pred_mean.index, y=list(df_pred_low[option_low[0]].tolist()), name=str(option_low[0]), mode='lines',
                                                                        line = dict(color='limegreen')))
                fig_plotly.add_trace(go.Scatter(x=df_pred_mean.index, y=list(df_pred_up[option_high[0]].tolist()), name=str(option_high[0]), mode='lines',
                                                                        line=dict(color='green')))
    #             fig_plotly.add_trace(go.Scatter(x=df_pred_mean.index, y=list(df_temp[option_temp[0]].tolist()), name=str(option_temp[0])+"_actual", mode='lines',
    #                                                                     line=dict(color='lightsalmon')))

                fig_plotly_.add_trace(go.Scatter(x=df_pred_mean.index, y=df_pred_mean[option_mean[0]].tolist(), name=str(option_mean[0]),
                                                            line = dict(color='firebrick', width=4, dash='solid')))
                fig_plotly_.add_trace(go.Scatter(x=df_pred_mean.index, y=list(df_pred_low[option_low[0]].tolist()), name=str(option_low[0]),
                                                                        line = dict(color='limegreen', width=4, dash='dot')))
                fig_plotly_.add_trace(go.Scatter(x=df_pred_mean.index, y=list(df_pred_up[option_high[0]].tolist()), name=str(option_high[0]),
                                                                        line=dict(color='green', width=4, dash='dot')))
                fig_plotly_.add_trace(go.Scatter(x=df_pred_mean.index, y=list(df_temp[option_temp[0]].tolist()), name=str(option_temp[0])+"_actual",
                                                                        line=dict(color='lightsalmon', width=4, dash='solid')))

                fig_plotly_.update_layout(plot_bgcolor='rgba(0,0,0,0)')

                plot_holder.plotly_chart(fig_plotly,use_container_width=True)
                st.plotly_chart(fig_plotly_,use_container_width=True)
                        
                if ph_metrics.selectbox('Query metrics',('False', 'True'))=="True":
                    st.subheader("Metrics")

                    
                    mt_payload=real_config.copy()
                    mt_payload['StepFunctionAPI']="metrics"
                    binary = json.dumps(mt_payload)

                    invoke_response = lambda_client.invoke(FunctionName=FunctionName,
                                                            Payload=binary,
                                                            InvocationType='RequestResponse')

                    metrics = invoke_response['Payload'].read()

                    metrics = json.loads(metrics)['metrics']
                    
                    st.write("___")
                    st.write("Mean Quantile Loss",metrics[1])
                    st.write("Train Loss",metrics[3])
                    st.write("Test RMSE",metrics[5])
                    st.write("___")
               
    

    
    
if __name__ == "__main__":
    debug = os.getenv('DASHBOARD_DEBUG', 'false') == 'true'
    if debug:
        main()
    else:
        try:
            main()
        except Exception as e:
            st.error('Internal error occurred.')