import json
import boto3
import logging
import numpy as np
import pandas as pd
import io
import pickle
from io import StringIO
import datetime
import time
from decimal import Decimal
from sklearn.preprocessing import RobustScaler


REGION = 'eu-west-1'
S3_CLIENT = boto3.client('s3', region_name=REGION)
S3_RESOURCE = boto3.resource('s3', region_name=REGION)
SM_CLIENT = boto3.client('sagemaker',region_name=REGION)
runtime= boto3.client('runtime.sagemaker')


def predict(endpoint, contentType, ts, cat=None, dynamic_feat=None, 
            num_samples=30, return_samples=False, quantiles=["0.1", "0.5", "0.9"]):
    """Requests the prediction of for the time series listed in `ts`, each with the (optional)
    corresponding category listed in `cat`.

    ts -- `pandas.Series` object, the time series to predict
    cat -- integer, the group associated to the time series (default: None)
    num_samples -- integer, number of samples to compute at prediction time (default: 100)
    return_samples -- boolean indicating whether to include samples in the response (default: False)
    quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])

    Return value: list of `pandas.DataFrame` objects, each containing the predictions
    """
    prediction_time = ts.index[-1] + datetime.timedelta(days=1)
    quantiles = [str(q) for q in quantiles]
    req = __encode_request(ts, cat, dynamic_feat, num_samples, return_samples, quantiles)
    res = runtime.invoke_endpoint(EndpointName=endpoint,
                                   ContentType=contentType,
                                   Body=req)
    #res = super(DeepARPredictor, self).predict(req)
    return __decode_response(res, ts.index.freq, prediction_time, return_samples)

def __encode_request(ts, cat, dynamic_feat, num_samples, return_samples, quantiles):
    instance = series_to_dict(ts, cat if cat is not None else None, dynamic_feat if dynamic_feat else None)

    configuration = {
        "num_samples": num_samples,
        "output_types": ["quantiles", "samples"] if return_samples else ["quantiles"],
        "quantiles": quantiles
    }

    http_request_data = {
        "instances": [instance],
        "configuration": configuration
    }

    return json.dumps(http_request_data).encode('utf-8')

def __decode_response(response, freq, prediction_time, return_samples):
    # we only sent one time series so we only receive one in return
    # however, if possible one will pass multiple time series as predictions will then be faster
    predictions = json.loads(response['Body'].read().decode())['predictions'][0]
    prediction_length = len(next(iter(predictions['quantiles'].values())))
    prediction_index = pd.date_range(start=prediction_time, freq=freq, periods=prediction_length)        
    if return_samples:
        dict_of_samples = {'sample_' + str(i): s for i, s in enumerate(predictions['samples'])}
    else:
        dict_of_samples = {}
    return pd.DataFrame(data={**predictions['quantiles'], **dict_of_samples}, index=prediction_index)

def encode_target(ts):
    return [x if np.isfinite(x) else "NaN" for x in ts]        

def series_to_dict(ts, cat=None, dynamic_feat=None):
    """Given a pandas.Series object, returns a dictionary encoding the time series.

    ts -- a pands.Series object with the target time series
    cat -- an integer indicating the time series category

    Return value: a dictionary
    """
    obj = {"start": str(ts.index[0]), "target": encode_target(ts)}
    if cat is not None:
        obj["cat"] = cat
    if dynamic_feat is not None:
        obj["dynamic_feat"] = dynamic_feat        
    return obj
    
def lambda_handler(event, context):
    
    BUCKET_NAME = event['BucketName']
    ENHANCED_DATASET_KEY_TARGET = event['EnhancedDatasetKeyTargetScaled']
    ENHANCED_DATASET_KEY_EXP_SMOOTH = event['EnhancedDatasetKeyExpSmoothScaled']
    ENHANCED_DATASET_KEY_OD_LABEL = event['EnhancedDatasetKeyODLabel']
    ENHANCED_DATASET_KEY_OD_LABEL_ALL = event['EnhancedDatasetKeyODLabelAll']
    ENHANCED_DATASET_CATEGORICAL = event['EnhancedDatasetKeyCategoricalFeatures']
    FREQ = event['TimeseriesFreq']
    PRED_LENGTH = int(event['PredictionLength'])
    ENDPOINT_NAME = event['EndpointName']
    CONFIDENCE = int(event['Confidence'])
    TRAIN_START = event['TrainDatasetStart']
    ADD_SAMPLES = event['AddSamples']
    NUM_SAMPLES = int(event['NumberOfSamples'])
    FORECAST_DAY = event['ForecastDay']
    SCALER_KEY = event['ScaleKey']
    CENTER_KEY = event['CenterKey']



    TS = datetime.datetime.strftime(datetime.datetime.utcnow(), '%Y-%m-%d-%H-%M-%S')


    PREDICTIONS_DEEPAR_KEY = event['PredictionsDeepARKey']
    
    # list existing endpoints
    response_endpoint = SM_CLIENT.list_endpoints(
                                        SortBy='CreationTime',
                                        SortOrder='Ascending',
                                        MaxResults=1,
                                        NameContains=ENDPOINT_NAME
                                    )
    
    # pick up the latest one
    ENDPOINT_NAME = response_endpoint['Endpoints'][0]['EndpointName']
   
    # wait until active
    while True:
        status = SM_CLIENT.describe_endpoint(
                                            EndpointName=ENDPOINT_NAME
                                            )
        if status['EndpointStatus'] == 'InService': break
        time.sleep(10)
    
    # read target data
    obj_data = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=ENHANCED_DATASET_KEY_TARGET)
    body_data = obj_data['Body']
    csv_string_data = body_data.read().decode('utf-8')
    data_target = pd.read_csv(StringIO(csv_string_data),sep=",", index_col=0, parse_dates=True, dayfirst=True)
    
     # read dynamic features (exp smoothing)
    obj_data = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=ENHANCED_DATASET_KEY_EXP_SMOOTH)
    body_data = obj_data['Body']
    csv_string_data = body_data.read().decode('utf-8')
    data_dynamic = pd.read_csv(StringIO(csv_string_data),sep=",", index_col=0, parse_dates=True, dayfirst=True)
    
    # read dynamic features (od labels)
    obj_data = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=ENHANCED_DATASET_KEY_OD_LABEL)
    body_data = obj_data['Body']
    csv_string_data = body_data.read().decode('utf-8')
    data_dynamic_od_label = pd.read_csv(StringIO(csv_string_data),sep=",", index_col=0, parse_dates=True, dayfirst=True)
    
    # read dynamic features (od labels all)
    obj_data = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=ENHANCED_DATASET_KEY_OD_LABEL_ALL)
    body_data = obj_data['Body']
    csv_string_data = body_data.read().decode('utf-8')
    data_dynamic_od_label_all = pd.read_csv(StringIO(csv_string_data),sep=",", index_col=0, parse_dates=True, dayfirst=True)
    
    # read categorical features 
    obj_data = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=ENHANCED_DATASET_CATEGORICAL)
    body_data = obj_data['Body']
    jsonObject = json.loads(body_data.read())
    
    primary_label=np.array(jsonObject['primary_cat'])
    sec_label=np.array(jsonObject['secondary_cat'])

    
    # covert the time series into lists
    num_timeseries = data_target.shape[1]
    
    timeseries = []
    for i in range(num_timeseries):
        timeseries.append(data_target.iloc[:,i])
        
    timeseries_dynamic = []
    timeseries_dynamic_od_label = []
    timeseries_dynamic_od_label_all = []

    for i in range(num_timeseries):
        timeseries_dynamic.append(data_dynamic.iloc[:,i])
        timeseries_dynamic_od_label.append(data_dynamic_od_label.iloc[:,i])
        timeseries_dynamic_od_label_all.append(data_dynamic_od_label_all.iloc[:,i])
        

    start_dataset = pd.Timestamp(TRAIN_START, freq=FREQ)
    forecast_date = pd.Timestamp(FORECAST_DAY, freq=FREQ)

    # compute qunatiles based on selected confidence threshold
    assert(CONFIDENCE > 50 and CONFIDENCE < 100)
    low_quantile = 0.5 - CONFIDENCE * 0.005
    up_quantile = CONFIDENCE * 0.005 + 0.5


    #table = dynamodb_resource.Table(TABLE_NAME)

    target_variable_list = data_target.columns
    
    predictions_list_mean =[]
    predictions_list_up =[]
    predictions_list_low =[]
    
    
    # run through the features list
    for variable_id in range(len(target_variable_list)):

        # prepare the request
        target = timeseries[variable_id][start_dataset:forecast_date + datetime.timedelta(days=PRED_LENGTH)].asfreq(freq=FREQ)
        dynamic_feat = [timeseries_dynamic[variable_id][start_dataset:forecast_date + datetime.timedelta(days=PRED_LENGTH)].asfreq(freq=FREQ).tolist(),
                        timeseries_dynamic_od_label[variable_id][start_dataset:forecast_date + datetime.timedelta(days=PRED_LENGTH)].asfreq(freq=FREQ).tolist(),
                        timeseries_dynamic_od_label_all[variable_id][start_dataset:forecast_date + datetime.timedelta(days=PRED_LENGTH)].asfreq(freq=FREQ).tolist()]
        cat_feat = [int(primary_label[variable_id]), int(sec_label[variable_id])]

        #print(variable_id)
        
        # prepare the input format for inference
        args = {
            "ts": target[:forecast_date],
            "return_samples": ADD_SAMPLES,
            "quantiles": [low_quantile, 0.5, up_quantile],
            "num_samples": NUM_SAMPLES
        }

        args["dynamic_feat"] = dynamic_feat
        args["cat"] = cat_feat

        # invoke the endpoint
        prediction = predict(endpoint=ENDPOINT_NAME, contentType ='application/json', **args)


        # pack the results in a dataframe
        column_name_mean = target_variable_list[variable_id] + '_mean'
        df_predictions_mean = pd.Series(prediction["0.5"], name=column_name_mean, index=prediction.index)
        predictions_list_mean.append(df_predictions_mean)

        column_name_low = target_variable_list[variable_id] + '_low_quantile'
        df_predictions_low = pd.Series(prediction[str(low_quantile)], name=column_name_low, index=prediction.index)
        predictions_list_low.append(df_predictions_low)

        column_name_up = target_variable_list[variable_id] + '_up_quantile'
        df_predictions_up = pd.Series(prediction[str(up_quantile)], name=column_name_up, index=prediction.index)
        predictions_list_up.append(df_predictions_up)
        
        
    predictions_list_mean = pd.concat(predictions_list_mean,axis=1,sort=False)
    predictions_list_low = pd.concat(predictions_list_low,axis=1,sort=False)
    predictions_list_up = pd.concat(predictions_list_up,axis=1,sort=False)

    # read the scaler values
    scale_array_read = io.BytesIO()
    S3_CLIENT.download_fileobj(BUCKET_NAME,SCALER_KEY, scale_array_read)
    scale_array_read.seek(0)
    scale_array_read = pickle.load(scale_array_read)
    #print(scale_array_read)

    center_array_read = io.BytesIO()
    S3_CLIENT.download_fileobj(BUCKET_NAME,CENTER_KEY,center_array_read)
    center_array_read.seek(0)
    center_array_read = pickle.load(center_array_read)
    #print(center_array_read)

    scaler = RobustScaler()
    scaler.scale_ = scale_array_read
    scaler.center_ = center_array_read

    #apply the inverse transform (rescale)
    predictions_list_mean[predictions_list_mean.columns] = scaler.inverse_transform(predictions_list_mean[predictions_list_mean.columns])
    predictions_list_low[predictions_list_low.columns] = scaler.inverse_transform(predictions_list_low[predictions_list_low.columns])
    predictions_list_up[predictions_list_up.columns] = scaler.inverse_transform(predictions_list_up[predictions_list_up.columns])

    # concat all quantile in one frame
    predictions_list = pd.concat([predictions_list_mean,predictions_list_low,predictions_list_up],axis=1,sort=False)
    
    # write results to S3
    csv_buffer = StringIO()
   
    predictions_list.to_csv(csv_buffer)
    
    S3_RESOURCE.Object(BUCKET_NAME, PREDICTIONS_DEEPAR_KEY).put(Body=csv_buffer.getvalue())

    #delete endpoint
    
    #SM_CLIENT.delete_endpoint(EndpointName=ENDPOINT_NAME)

    return {
        'status': "Serving Request Succesful",
        'body': PREDICTIONS_DEEPAR_KEY
    }
