import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')


def lambda_handler(event, context):

    # list all models in a decending order
    response = sm_client.list_models(
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1,
    NameContains='DeepAR'
    )
    
    # retrieve the definition of the latest saved model
    response_model = sm_client.describe_model(
    ModelName=response['Models'][0]['ModelName']
    )

    # get the training job name
    training_job_name = response_model['PrimaryContainer']['ModelDataUrl']
    training_job_name = training_job_name.split(str(event['OutputPathDeepAR']+'/'))[1]
    training_job_name = training_job_name.split('/')[0]

    # retrieve the training job results (incl. the accuracy metrics, RMSE and Qunatile Loss)
    response_training = sm_client.describe_training_job(TrainingJobName=training_job_name)
    
    # convert all datetime objects returned to unix time

    for index, metric in enumerate(response_training['FinalMetricDataList']):
        metric['Timestamp'] = metric['Timestamp'].timestamp()


    if response_training['FinalMetricDataList'][6]['Value'] < float(event['RMSEthresholdDeepAR']): PromotionFlag="True"
    else: PromotionFlag="False"

    return {
        'statusCode': 200,
        'trainingMetrics': response_training['FinalMetricDataList'],
        'PromotionFlag' : PromotionFlag
    }