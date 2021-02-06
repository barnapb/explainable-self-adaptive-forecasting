import argparse
import os
import warnings
import pandas as pd
import numpy as np
from pandas import read_csv
from numpy import array
import json
import io
import re
import pickle
from io import StringIO
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
import shap
import xgboost
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from pyod.models.copod import COPOD
import matplotlib
from neuralprophet import NeuralProphet
from pyod.models.auto_encoder import AutoEncoder

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# Holt Winter’s Exponential Smoothing
def exp_smoothing_forecast(train, test, config):
    t,d,s,p,b,r = config
    
    # define model
    model = ExponentialSmoothing(train, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    
    #retrieve the seasonal smoothing coefficient
    smoothing_seasonal = model_fit.params['smoothing_seasonal']

    # predict
    yhat = model_fit.predict(len(train),len(train)+len(test)-1)
    
    return yhat, smoothing_seasonal

def strip_end(text, suffix):
    if not text.endswith(suffix):
        return text
    return text[:len(text)-len(suffix)]
# rmse
def calculate_error(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# split the target vraiable dataset into train/test
def validation_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# walk-forward validation
def walk_forward_val(data, n_test, cfg):
    
    # split dataset
    train, test = validation_split(data, n_test)
    
    # seed history with training dataset
    train = [x for x in train]
    
    # fit model and predict
    yhat, smoothing_seasonal = exp_smoothing_forecast(train, test, cfg)
    
    error = calculate_error(test, yhat)
    
    return error, smoothing_seasonal

# evaluate the model, return None if validation failed
def evaluate_model(data, n_test, cfg):

    result = None
    key = cfg
    
    result, smoothing_seasonal = walk_forward_val(data, n_test, cfg)
    
    return (key, result, smoothing_seasonal)

# grid search optimizer
def optimizer(data, cfg_list, n_test, smoothing_seasonal_threshold, parallel=True):
    
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(evaluate_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [evaluate_model(data, n_test, cfg) for cfg in cfg_list]
    
    # remove empty results and entries where the seasonal smoothing coefficient is less than threshold
    scores = [r for r in scores if r[1] != None]
    scores = [r for r in scores if r[2] >= smoothing_seasonal_threshold]
    
    # sort the configurations list by error
    scores.sort(key=lambda tup: tup[1])

    return scores

# create a set of exponential smoothing configurations
def config_def(seasonal=[None]):
    
    models = list()
    
    # define config list (now limited to a set of options, while still searching through the seasonal cycles)
    # for the next release the rest of them will be enabled

    t_params = ['add']
    d_params = [False]
    s_params = ['add']
    p_params = seasonal
    b_params = [False]
    r_params = [False]
  
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t,d,s,p,b,r]
                            models.append(cfg)
    return models
    
def exp_smoothing_forecast_main(series, config, prediction_lenght):
    
    t,d,s,p,b,r = config
    
    series = array(series)
    
    # define model
    model = ExponentialSmoothing(series, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)

    yhat = model_fit.predict(0,len(series)+ prediction_lenght)
    return yhat

def encode_target(ts):
    return [x if np.isfinite(x) else "NaN" for x in ts]
    
def check_dataset_consistency(train_dataset, test_dataset=None):
    d = train_dataset[0]
    has_dynamic_feat = 'dynamic_feat' in d
    if has_dynamic_feat:
        num_dynamic_feat = len(d['dynamic_feat'])
    has_cat = 'cat' in d
    if has_cat:
        num_cat = len(d['cat'])
    
    def check_ds(ds):
        for i, d in enumerate(ds):
            if has_dynamic_feat:
                assert 'dynamic_feat' in d
                assert num_dynamic_feat == len(d['dynamic_feat'])
                for f in d['dynamic_feat']:
                    assert len(d['target']) == len(f)
            if has_cat:
                assert 'cat' in d
                assert len(d['cat']) == num_cat
    check_ds(train_dataset)
    if test_dataset is not None:
        check_ds(test_dataset)
        
def write_dicts_to_file(path, data):
    with open(path, 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode('utf-8'))

def abn_contrb_task(clf_target_all,contamination,idx):
    result = clf_target_all.explain_outlier(idx, cutoffs=[1-contamination,0.99])
    return list(result[0]), result[1][0]

def nerualProphet_task(NEURAL_N_CHANGEPOINTS, NEURAL_CHANGEPOINTS_RANGE, NEURAL_YEARLY_SEASONALITY, NEURAL_WEEKLY_SEASONALITY, NEURAL_DAILY_SEASONALITY, NEURAL_N_FORECASTS, NEURAL_N_LAGS, RESAMPLE_FREQ, PRED_LENGTH, data_syntetic, col):
    
    prophet_df=pd.DataFrame(columns=["ds","y"])
    prophet_df["ds"]=data_syntetic[col].index
    prophet_df["y"]=data_syntetic[col].values
    
    if NEURAL_YEARLY_SEASONALITY == "True": NEURAL_YEARLY_SEASONALITY=True
    elif NEURAL_YEARLY_SEASONALITY == "False": NEURAL_YEARLY_SEASONALITY=False

    if NEURAL_WEEKLY_SEASONALITY == "True": NEURAL_WEEKLY_SEASONALITY=True
    elif NEURAL_WEEKLY_SEASONALITY == "False": NEURAL_WEEKLY_SEASONALITY=False

    if NEURAL_DAILY_SEASONALITY == "True": NEURAL_DAILY_SEASONALITY=True
    elif NEURAL_DAILY_SEASONALITY == "False": NEURAL_DAILY_SEASONALITY=False

    m = NeuralProphet(n_changepoints=NEURAL_N_CHANGEPOINTS,changepoints_range=NEURAL_CHANGEPOINTS_RANGE, yearly_seasonality=NEURAL_YEARLY_SEASONALITY, weekly_seasonality=NEURAL_WEEKLY_SEASONALITY, daily_seasonality=NEURAL_DAILY_SEASONALITY, n_forecasts=NEURAL_N_FORECASTS, n_lags=NEURAL_N_LAGS)
    metrics = m.fit(prophet_df, freq=RESAMPLE_FREQ)
    future = m.make_future_dataframe(prophet_df, periods=PRED_LENGTH, n_historic_predictions=True)
    forecast = m.predict(future)
    
    return forecast['yhat1'], col

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-variables-list', type=str, default="")
    parser.add_argument('--resample-freq', type=str, default="D")
    parser.add_argument('--pivot-column', type=str, default="")
    parser.add_argument('--timestamp-column', type=str, default="")
    parser.add_argument('--interpolation-order', type=int, default=3)
    parser.add_argument('--interpolation-method', type=str, default="spline")
    parser.add_argument('--prediction-lenght', type=int, default=7)
    parser.add_argument('--step-size', type=int, default=7)
    parser.add_argument('--test-data-size', type=int, default=180)
    parser.add_argument('--min-seasonal-guess', type=int, default=7)
    parser.add_argument('--max-seasonal-guess', type=int, default=30)
    parser.add_argument('--seasonal-smoothing-thresold', type=float, default=0.05)
    parser.add_argument('--windows-folds', type=int, default=3)
    parser.add_argument('--rolling-window', type=int, default=7)
    parser.add_argument('--xgboost-estimators', type=int, default=100)
    parser.add_argument('--xgboost-depth', type=int, default=6)
    parser.add_argument('--xgboost-shap-samples', type=int, default=1000)
    parser.add_argument('--xgboost-eta', type=float, default=0.3)
    parser.add_argument('--xgboost-njobs', type=int, default=1)
    parser.add_argument('--xgboost-gamma', type=float, default=0)
    parser.add_argument('--feature-exp-order', type=int, default=2)
    parser.add_argument('--contamination', type=float, default=0.05)
    parser.add_argument('--n-neighbors', type=int, default=2)
    parser.add_argument('--shap-interaction-flag', type=bool, default=True)
    parser.add_argument('--kmeans-clusters', type=int, default=3)
    parser.add_argument('--kmeans-iters', type=int, default=10)
    parser.add_argument('--algo-selection', type=str, default="NeuralProphet")
    parser.add_argument('--neural-n-changepoints', type=int, default=5)
    parser.add_argument('--neural-changepoints-range', type=float, default=0.8)
    parser.add_argument('--neural-yearly-seasonality', type=str, default="auto")
    parser.add_argument('--neural-weekly-seasonality', type=str, default="auto")
    parser.add_argument('--neural-daily-seasonality', type=str, default="auto")
    parser.add_argument('--neural-n-forecasts', type=int, default=1)
    parser.add_argument('--neural-n-lags', type=int, default=0)

    
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))

    input_data_path_raw = os.path.join('/opt/ml/processing/input', 'Air_Pollution_in_Seoul.csv')
    
    TARGET_VARIABLES = args.target_variables_list
    RESAMPLE_FREQ =  args.resample_freq
    PIVOT_COLUMN = args.pivot_column
    TS_COLUMN = args.timestamp_column
    INTERP_ORDER = args.interpolation_order
    INTERP_METHOD = args.interpolation_method
    PRED_LENGTH = args.prediction_lenght
    STEP_SIZE = args.step_size
    TEST_SIZE = args.test_data_size
    MIN_SEASONAL_GUESS = args.min_seasonal_guess
    MAX_SEASONAL_GUESS = args.max_seasonal_guess
    SEASONAL_SMOOTHING_THRE = args.seasonal_smoothing_thresold
    WINDOWS_FOLDS = args.windows_folds
    ROLLING_WINDOW = args.rolling_window
    XGBOOST_ESTIMATORS = args.xgboost_estimators
    XGBOOST_DEPTH = args.xgboost_depth
    XGBOOST_SHAP_SAMPLES = args.xgboost_shap_samples
    XGBOOST_ETA = args.xgboost_eta
    XGBOOST_NJOBS = args.xgboost_njobs
    XGBOOST_GAMMA = args.xgboost_gamma
    FETAURE_EXP_ORDER =args.feature_exp_order
    CONTAMINATION=args.contamination
    N_NEIGHBORS=args.n_neighbors
    SHAP_INTERECTION_FLAG=args.shap_interaction_flag
    KMEANS_CLUSTERS=args.kmeans_clusters
    KMEANS_ITERS=args.kmeans_iters
    ALGO_SELECTION=args.algo_selection
    NEURAL_N_CHANGEPOINTS=args.neural_n_changepoints
    NEURAL_CHANGEPOINTS_RANGE=args.neural_changepoints_range
    NEURAL_YEARLY_SEASONALITY=args.neural_yearly_seasonality
    NEURAL_WEEKLY_SEASONALITY=args.neural_weekly_seasonality
    NEURAL_DAILY_SEASONALITY=args.neural_daily_seasonality
    NEURAL_N_FORECASTS=args.neural_n_forecasts
    NEURAL_N_LAGS=args.neural_n_lags

    target_column_list=list(TARGET_VARIABLES.split(','))
    TARGET_VARIABLES_RAW = TARGET_VARIABLES + "," + PIVOT_COLUMN
    TARGET_VARIABLES = TARGET_VARIABLES.replace(",","_|")+"_"
    TARGET_VARIABLES_RAW = TARGET_VARIABLES_RAW.replace(",","|")

    
    raw_data = pd.read_csv(input_data_path_raw, sep=",", index_col=TS_COLUMN, parse_dates=True, dayfirst=True)

    # drop columns which are not within the target variables list/ pivot Column

    list_variables = raw_data.filter(regex=TARGET_VARIABLES_RAW).columns

    raw_data = raw_data.loc[:, raw_data.columns.isin(list_variables)]
   
    max_len = float("-inf")
    
    concat_data = raw_data.copy()
    concat_data = concat_data.resample(RESAMPLE_FREQ).mean()
    concat_data[:] = np.NaN
    
    appended_data=[]
    
    # Generate the enhanced dataset as columnar data
    for i in raw_data[PIVOT_COLUMN].dropna().unique():
        
        temp_data = raw_data.loc[raw_data[PIVOT_COLUMN] == i]
        
        temp_data = temp_data.resample(RESAMPLE_FREQ).mean()
        
        temp_data = temp_data.loc[~temp_data.index.duplicated()]
        
        temp_data = temp_data.reindex(concat_data.index)
        
        temp_data = temp_data.add_suffix('_{}'.format(str(i)))
        
        temp_data = temp_data.drop(temp_data.filter(regex=PIVOT_COLUMN).columns, axis=1)
        
        appended_data.append(temp_data)
        

    appended_data = pd.concat(appended_data,axis=1,sort=False)

    TRAIN_START = str(appended_data.index[0])
    TRAIN_END = str(appended_data.index[-1])

    TARGET_VARIABLES_LIST = appended_data.filter(regex=TARGET_VARIABLES).columns

    TARGET_VARIABLES_LIST = list(TARGET_VARIABLES_LIST)

    PD_SHAPE = appended_data.shape[0]

    if ALGO_SELECTION=="ExponentialSmoothing":

        for TARGET_COLUMN in TARGET_VARIABLES_LIST:

            print(TARGET_COLUMN)
        
            data_syntetic = appended_data.loc[:, appended_data.columns.isin([TARGET_COLUMN])]
   
            # run interploation
            data_syntetic = data_syntetic.interpolate(method=INTERP_METHOD,order=INTERP_ORDER, limit_area='inside')

            # load dataset
            series = data_syntetic[TARGET_COLUMN]

            first_valid_idx = series.first_valid_index()
            last_valid_idx = series.last_valid_index()
            
            series = series.dropna()
            
            data = series.values

            # generate a list of seasonal cycles
            seasonal_list = np.arange(MIN_SEASONAL_GUESS, MAX_SEASONAL_GUESS, STEP_SIZE).tolist()
            
            # model configs
            cfg_list = config_def(seasonal=seasonal_list)

            # run the grid search optimizer with exponential smoothing
            scores = optimizer(data, cfg_list, TEST_SIZE, SEASONAL_SMOOTHING_THRE)

            if not scores:
                final_config = ['add', False, 'add', MIN_SEASONAL_GUESS, False, False]
            else:
                final_config = scores[0][0]

            print(scores)
            
            data_dynamic = data_syntetic.copy()
            
            # reindex the frame to accomodate the prediction length   
            
            new_index = pd.date_range(data_syntetic.index[0], periods=PD_SHAPE + PRED_LENGTH, freq=RESAMPLE_FREQ)
            data_dynamic= data_dynamic.reindex(new_index)

            loc_first_valid_idx = data_dynamic.index.get_loc(first_valid_idx)
            loc_last_valid_idx = data_dynamic.index.get_loc(last_valid_idx)

            # Fit the exponential smoothing on the interpolated data and generate the dynamic features 
        
            column_name = TARGET_COLUMN + '_dynamicFeat'
            result_exp_smooth = exp_smoothing_forecast_main(data, final_config, data_dynamic.shape[0]-loc_last_valid_idx-1)
            data_dynamic[TARGET_COLUMN].iloc[loc_first_valid_idx:] = result_exp_smooth[1:]
            data_dynamic = data_dynamic.rename(columns={TARGET_COLUMN: column_name})

            appended_data = pd.concat([appended_data,data_dynamic],axis=1,sort=False)

    elif ALGO_SELECTION=="NeuralProphet":

        data_syntetic = appended_data.loc[:, appended_data.columns.isin(TARGET_VARIABLES_LIST)]

        # run interploation
        data_syntetic = data_syntetic.interpolate(method=INTERP_METHOD,order=INTERP_ORDER, limit_area='inside')

        # fit Neural Prophet on the interpolated data and generate the dynamic features
        executor = Parallel(n_jobs=cpu_count(), backend='loky')
        tasks = (delayed(nerualProphet_task)(NEURAL_N_CHANGEPOINTS, NEURAL_CHANGEPOINTS_RANGE, NEURAL_YEARLY_SEASONALITY, NEURAL_WEEKLY_SEASONALITY, NEURAL_DAILY_SEASONALITY, NEURAL_N_FORECASTS, NEURAL_N_LAGS, RESAMPLE_FREQ, PRED_LENGTH, data_syntetic, col) for _, col in  enumerate(data_syntetic.columns))
        result_np = executor(tasks)

        data_dynamic = data_syntetic.copy()

        # reindex the frame to accomodate the prediction length
        new_index = pd.date_range(data_syntetic.index[0], periods=PD_SHAPE + PRED_LENGTH, freq=RESAMPLE_FREQ)
        data_dynamic= data_dynamic.reindex(new_index)

        # load the results
        for i, col in enumerate(data_dynamic.columns):
            if result_np[i][1]==col:
                data_dynamic[col]=list(result_np[i][0])
                
        data_dynamic = data_dynamic.add_suffix('_dynamicFeat')

        appended_data = pd.concat([appended_data,data_dynamic],axis=1,sort=False)

        
    # filter the enhanced dataset
 
    list_targets = ','.join(TARGET_VARIABLES_LIST).replace(",","|")

    list_columns = appended_data.filter(regex=list_targets).columns

    appended_data = appended_data.loc[:, appended_data.columns.isin(list_columns)]
    
    list_columns_dynamic = appended_data.filter(regex='_dynamicFeat').columns
    
    
    data_target = appended_data.loc[:, ~appended_data.columns.isin(list_columns_dynamic)]
    
    data_target_ui = data_target.copy()

    # normalize the target data, using a Robust Scaler
    scaler_target = RobustScaler()
    data_target[data_target.columns] = scaler_target.fit_transform(data_target[data_target.columns])
    

    scaler_scale = scaler_target.scale_
    scaler_center = scaler_target.center_
    
    data_dynamic = appended_data.loc[:, appended_data.columns.isin(list_columns_dynamic)]
    
    
    # running PyOD for the dynamic features in order to generate the attention signals

    data_dynamic_od_score = data_dynamic.copy()
    data_dynamic_od_label = data_dynamic.copy()
    data_dynamic_od_label[:]=0
    data_dynamic_od_score[:]=0
    data_dynamic_od_score_all = data_dynamic.copy()
    data_dynamic_od_label_all = data_dynamic.copy()

    clf_target_copod = COPOD(contamination=CONTAMINATION)
    clf_target_copod.fit(data_dynamic)

    clf_target_auto = AutoEncoder(epochs=30, contamination=CONTAMINATION)
    clf_target_auto.fit(data_dynamic)

    executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
    tasks = (delayed(abn_contrb_task)(clf_target_copod,CONTAMINATION,id_) for id_, _ in  enumerate(data_dynamic.index))
    result_all = executor(tasks)

    for idx, _ in enumerate(result_all):
        for i, el in enumerate(result_all[idx][0]):
            data_dynamic_od_score[data_dynamic_od_score.columns[i]][idx] = el
            if el > result_all[idx][1]:
                data_dynamic_od_label[data_dynamic_od_label.columns[i]][idx] = el

    for dynamic_col in data_dynamic.columns:

            data_dynamic_od_score[dynamic_col] = (data_dynamic_od_score[dynamic_col].rolling(window=ROLLING_WINDOW).mean())**FETAURE_EXP_ORDER
            data_dynamic_od_label[dynamic_col] = (data_dynamic_od_label[dynamic_col].rolling(window=ROLLING_WINDOW).mean())**FETAURE_EXP_ORDER

            data_dynamic_od_score_all[dynamic_col] = clf_target_auto.decision_scores_
            data_dynamic_od_label_all[dynamic_col] = clf_target_auto.labels_ 

            data_dynamic_od_score_all[dynamic_col] = (data_dynamic_od_score_all[dynamic_col].rolling(window=ROLLING_WINDOW).mean())**FETAURE_EXP_ORDER
            data_dynamic_od_label_all[dynamic_col] = data_dynamic_od_label_all[dynamic_col].rolling(window=ROLLING_WINDOW).sum()


    # data imputation for the dynamic features (null values not accepted within DeepAR for the dynamic features)
    imp_knn_exp = KNNImputer(n_neighbors=N_NEIGHBORS)
    imp_knn_od_score = KNNImputer(n_neighbors=N_NEIGHBORS)
    imp_knn_od_label = SimpleImputer(fill_value=0, strategy='constant')
    imp_knn_od_score_all = KNNImputer(n_neighbors=N_NEIGHBORS)
    imp_knn_od_label_all = SimpleImputer(fill_value=0, strategy='constant')


    imp_knn_exp.fit(data_dynamic.iloc[:,:])
    data_dynamic.iloc[:,:] = imp_knn_exp.transform(data_dynamic.iloc[:,:])

    imp_knn_od_score.fit(data_dynamic_od_score.iloc[:,:])
    data_dynamic_od_score.iloc[:,:] = imp_knn_od_score.transform(data_dynamic_od_score.iloc[:,:])

    imp_knn_od_label.fit(data_dynamic_od_label.iloc[:,:])
    data_dynamic_od_label.iloc[:,:] = imp_knn_od_label.transform(data_dynamic_od_label.iloc[:,:])

    imp_knn_od_score_all.fit(data_dynamic_od_score_all.iloc[:,:])
    data_dynamic_od_score_all.iloc[:,:] = imp_knn_od_score_all.transform(data_dynamic_od_score_all.iloc[:,:])

    imp_knn_od_label_all.fit(data_dynamic_od_label_all.iloc[:,:])
    data_dynamic_od_label_all.iloc[:,:] = imp_knn_od_label_all.transform(data_dynamic_od_label_all.iloc[:,:])

    # save a copy for the dashboard
    data_dynamic_ui = data_dynamic.copy()


    # normalize the dynamic data, using a Robust Scaler
    scaler_dynamic = RobustScaler()
    data_dynamic[data_dynamic.columns] = scaler_dynamic.fit_transform(data_dynamic[data_dynamic.columns])

    # save to S3

    data_target_ui.to_csv('/opt/ml/processing/features/Enhanced_Dataset.csv')
    data_target.to_csv('/opt/ml/processing/features/Enhanced_Dataset_Scaled.csv')
    data_dynamic_ui.to_csv('/opt/ml/processing/features/Enhanced_Dataset_DynamicFeatures.csv')
    data_dynamic.to_csv('/opt/ml/processing/features/Enhanced_Dataset_DynamicFeatures_Scaled.csv')
    data_dynamic_od_score.to_csv('/opt/ml/processing/features/Enhanced_Dataset_OD_Scores.csv')
    data_dynamic_od_label.to_csv('/opt/ml/processing/features/Enhanced_Dataset_OD_Labels.csv')
    data_dynamic_od_score_all.to_csv('/opt/ml/processing/features/Enhanced_Dataset_OD_Scores_All.csv')
    data_dynamic_od_label_all.to_csv('/opt/ml/processing/features/Enhanced_Dataset_OD_Labels_All.csv')
    
    num_timeseries = data_target.shape[1]
    num_timeseries_dynamic = data_dynamic.shape[1]
    
    # run KMeans clustering
    formatted_dataset = to_time_series_dataset([data_dynamic[col] for col in data_dynamic.columns])
    km_model = TimeSeriesKMeans(n_clusters=KMEANS_CLUSTERS, metric="dtw", max_iter=KMEANS_ITERS)
    km_model.fit_predict(formatted_dataset)

    # Use the labels to generate the secondary array of categorical features
    sec_label = km_model.labels_

    # Generate a 1st categorical domain, where the time series are grouped by targets

    primary_label = sec_label.copy()

    for i, col in enumerate(data_dynamic.columns):
        for j, target in enumerate(target_column_list):
            if target in col:
                primary_label[i]=j

    # write the cat fields to a json field (to be used during inference)
    cat_dictionary ={ 
                    "target_list": target_column_list,
                    "primary_cat": primary_label,
                    "secondary_cat": sec_label
                    } 
  
    json_object_ = json.dumps(cat_dictionary, cls=NumpyEncoder) 
        
    with open("/opt/ml/processing/features/Categorical_Features.json", "w") as outfile_: 
        outfile_.write(json_object_)


    # covert the time series into lists
    timeseries = []
    timeseries_dynamic = []
    #timeseries_dynamic_od_score = []
    timeseries_dynamic_od_label = []
    #timeseries_dynamic_od_score_all = []
    timeseries_dynamic_od_label_all = []

    for i in range(num_timeseries):
        timeseries.append(data_target.iloc[:,i])
        timeseries_dynamic.append(data_dynamic.iloc[:,i])
        #timeseries_dynamic_od_score.append(data_dynamic_od_score.iloc[:,i])
        timeseries_dynamic_od_label.append(data_dynamic_od_label.iloc[:,i])
        #timeseries_dynamic_od_score_all.append(data_dynamic_od_score_all.iloc[:,i])
        timeseries_dynamic_od_label_all.append(data_dynamic_od_label_all.iloc[:,i])
    
    # configure the train dataset boundaries based on the user-defined dates 
    # leave enough datapoints for the test set

    start_dataset = pd.Timestamp(TRAIN_START, freq=RESAMPLE_FREQ)
    end_training = pd.Timestamp(TRAIN_END, freq=RESAMPLE_FREQ)
    
    end_training = end_training - ((WINDOWS_FOLDS + 1)*PRED_LENGTH)*end_training.freq
    
    # prepare the json format required by DeepAR
    # start— A string with the format YYYY-MM-DD HH:MM:SS. The start timestamp can't contain time zone information.
    # target— An array of floating-point values or integers that represent the time series. 
    # You can encode missing values as null literals, or as "NaN" strings in JSON, or as nan floating-point values in Parquet.
    # dynamic_feat (optional)— An array of arrays of floating-point values or integers that represents the vector of custom feature time series (dynamic features). 
    # If you set this field, all records must have the same number of inner arrays (the same number of feature time series). 
    # In addition, each inner array must have the same length as the associated target value. 
    # Missing values are not supported in the features. For example, if target time series represents the demand of different products, 
    # an associated dynamic_feat might be a boolean time-series which indicates whether a promotion was applied (1) to the particular product or not (0):
    # {"start": ..., "target": [1, 5, 10, 2], "dynamic_feat": [[0, 1, 1, 0]]}

    training_data = [
    {
        "start": str(start_dataset),
        "target": encode_target(ts[start_dataset:end_training]),
        "dynamic_feat": [timeseries_dynamic[i][start_dataset:end_training].tolist(),
                         timeseries_dynamic_od_label[i][start_dataset:end_training].tolist(),
                         timeseries_dynamic_od_label_all[i][start_dataset:end_training].tolist()],
        "cat":[int(primary_label[i]), int(sec_label[i])]
    }
    for i, ts in enumerate(timeseries)
    ]
    
    test_data = [
        {
            "start": str(start_dataset),
            "target": encode_target(ts[start_dataset:end_training + (k*PRED_LENGTH)*end_training.freq]),
            "dynamic_feat": [timeseries_dynamic[i][start_dataset:end_training + (k*PRED_LENGTH)*end_training.freq].tolist(),
                             timeseries_dynamic_od_label[i][start_dataset:end_training + (k*PRED_LENGTH)*end_training.freq].tolist(),
                             timeseries_dynamic_od_label_all[i][start_dataset:end_training + (k*PRED_LENGTH)*end_training.freq].tolist()],
            "cat":[int(primary_label[i]), int(sec_label[i])]
        }
        for k in range(1,WINDOWS_FOLDS+1)
        for i, ts in enumerate(timeseries)
    ]
    
    training_data_gluonts = [
    {
        "start": str(start_dataset),
        "target": encode_target(ts[start_dataset:end_training]),
        "feat_dynamic_real": [timeseries_dynamic[i][start_dataset:end_training].tolist(),
                              timeseries_dynamic_od_label[i][start_dataset:end_training].tolist(),
                              timeseries_dynamic_od_label_all[i][start_dataset:end_training].tolist()],
        "feat_static_cat":[int(primary_label[i]), int(sec_label[i])]
    }
    for i, ts in enumerate(timeseries)
    ]
    
    test_data_gluonts = [
        {
            "start": str(start_dataset),
            "target": encode_target(ts[start_dataset:end_training + (k*PRED_LENGTH)*end_training.freq]),
            "feat_dynamic_real": [timeseries_dynamic[i][start_dataset:end_training + (k*PRED_LENGTH)*end_training.freq].tolist(),
                                  timeseries_dynamic_od_label[i][start_dataset:end_training + (k*PRED_LENGTH)*end_training.freq].tolist(),
                                  timeseries_dynamic_od_label_all[i][start_dataset:end_training + (k*PRED_LENGTH)*end_training.freq].tolist()],
            "feat_static_cat":[int(primary_label[i]), int(sec_label[i])]
        }
        for k in range(1,WINDOWS_FOLDS+1)
        for i, ts in enumerate(timeseries)
    ]
    
    # check consistency
    check_dataset_consistency(training_data, test_data)
    
    # write the DeepAR json files
    write_dicts_to_file("/opt/ml/processing/train/train_deepar.json", training_data)
    write_dicts_to_file("/opt/ml/processing/test/test_deepar.json", test_data)
    
    write_dicts_to_file("/opt/ml/processing/train/train_gluonts.json", training_data_gluonts)
    write_dicts_to_file("/opt/ml/processing/test/test_gluonts.json", test_data_gluonts)
 
    # compute shap values
    
    data_dynamic_ui = data_dynamic_ui.rename(columns=lambda x: x.strip("_dynamicFeat"))
    col_list = data_dynamic_ui.columns
    col_list = [x.split('_')[-1] for x in col_list]
    col_list = list(set(col_list))
    
    for target_column in target_column_list:
        for idx in col_list:
            col_list_rack = data_dynamic_ui.filter(regex="_{}$".format(idx)).columns
            data_raw_rack = data_dynamic_ui.loc[:, data_dynamic_ui.columns.isin(col_list_rack)]
            data_raw_rack = data_raw_rack.rename(columns=lambda x: strip_end(x,"_{}".format(idx)))
            
            X = data_raw_rack.drop(labels=[target_column], axis=1)
            y = data_raw_rack[target_column]
            
            model_xgb = xgboost.XGBRegressor(n_estimators=XGBOOST_ESTIMATORS, learning_rate=XGBOOST_ETA, n_jobs = XGBOOST_NJOBS, max_depth=XGBOOST_DEPTH, gamma=XGBOOST_GAMMA).fit(X, y)
            background = shap.maskers.Independent(X, max_samples=XGBOOST_SHAP_SAMPLES)
    

            # explain the XGBoost model with SHAP
            explainer_xgb = shap.TreeExplainer(model_xgb, background)
            shap_values_xgb = explainer_xgb(X)

            clustering = shap.utils.hclust(X, y)

            if SHAP_INTERECTION_FLAG:
                shap_interaction_values = shap.TreeExplainer(model_xgb).shap_interaction_values(X)

                shap_dictionary ={ 
                        "expected_values": explainer_xgb.expected_value,
                        "shap_values" : shap_values_xgb.values,
                        "shap_base_values" : shap_values_xgb.base_values,
                        "shap_data" : shap_values_xgb.data,
                        "shap_feature_names": shap_values_xgb.feature_names,
                        "shap_clustering": clustering,
                        "shap_interaction_values":shap_interaction_values,
                        "X": X.values,
                        "y": y.values,
                        "target_column": target_column
                    } 
            else:
                shap_dictionary ={ 
                        "shap_values" : shap_values_xgb.values,
                        "shap_base_values" : shap_values_xgb.base_values,
                        "shap_data" : shap_values_xgb.data,
                        "shap_feature_names": shap_values_xgb.feature_names,
                        "shap_clustering": clustering,
                        "X": X.values,
                        "y": y.values,
                        "target_column": target_column
                    } 

                
            # Serializing json  
            json_object = json.dumps(shap_dictionary, cls=NumpyEncoder) 
                
            with open("/opt/ml/processing/shap/xgboost_shap_{}_{}.json".format(idx,target_column), "w") as outfile: 
                outfile.write(json_object) 
            


    scaler_scale_train_output_path = os.path.join('/opt/ml/processing/scaler', 'deeparscaler_scale.pkl')
    scaler_center_train_output_path = os.path.join('/opt/ml/processing/scaler', 'deeparscaler_center.pkl')
   
    pickle.dump(scaler_scale, open(scaler_scale_train_output_path, "wb"))
    pickle.dump(scaler_center, open(scaler_center_train_output_path, "wb"))
