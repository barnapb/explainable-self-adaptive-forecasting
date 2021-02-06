import logging
import json


def lambda_handler(event, context):

    model_data = event['OutputPathDeepAR'] + "/"+ event['BestTrainingJobName'] + "/output/model.tar.gz"

    return {
        'statusCode': 200,
        'bestmodelARN': str(model_data)
    }