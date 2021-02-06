import json
import boto3
import os
import time

REGION=boto3.Session().region_name

SFN_CLIENT = boto3.client('stepfunctions', region_name=REGION)
SM_CLIENT = boto3.client('sagemaker')

STATE_MACHINE_ARN= os.environ["STATE_MACHINE_NAME"]


def lambda_handler(event, context):
    
    
    ACCOUNT_ID = context.invoked_function_arn.split(":")[4]
    #STATE_MACHINE_ARN = 'arn:aws:states:{}:{}:stateMachine:{}'.format(REGION,ACCOUNT_ID,STATE_MACHINE_NAME)
    
    state_id=0
    fail_details=""
    metrics=""
    
    print(event["StepFunctionAPI"])

    
    if event["StepFunctionAPI"] == "invoke":
        
        response = SFN_CLIENT.list_executions(
            stateMachineArn=STATE_MACHINE_ARN,
            statusFilter='RUNNING'
        )
        
        if response['executions']:
            status= "execution already in progress"
            
        else:
        
            response = SFN_CLIENT.start_execution(
                stateMachineArn= STATE_MACHINE_ARN,
                input=json.dumps(event)
            )
            
            status="execution in progress"
            
            os.environ["SF_EXECUTION_ARN"]=response['executionArn']
        
    elif event["StepFunctionAPI"] == "status":
        
        response = SFN_CLIENT.list_executions(
            stateMachineArn=STATE_MACHINE_ARN,
            statusFilter='RUNNING'
        )
        
        if response['executions']:
            
            execution_arn=response['executions'][0]['executionArn']
            
            
            response_ = SFN_CLIENT.describe_execution(
                                                    executionArn=execution_arn
                                                )
                                                
            
            if response_['status']=='RUNNING':
                
                params = {
                            'executionArn': execution_arn,
                            'reverseOrder': True,
                            'includeExecutionData':False
                        }
      
                response = SFN_CLIENT.get_execution_history(**params)
                  
                state_id = response['events'][0]['id']
                
                status= "execution in progress"
                    
        else:
            
            if os.environ["SF_EXECUTION_ARN"]:
                
                response_ = SFN_CLIENT.describe_execution(
                                                        executionArn=os.environ["SF_EXECUTION_ARN"]
                                                        )
                                          
                if response_['status']=='FAILED':
                
                    status = "execution failed"
                    
                    print(response_)
                
                    fail_details = "Failed. Check your email!"
                    
                elif response_['status']=='ABORTED':
                    
                    status = "execution aborted"
                
                    fail_details = "aborted"
    
                
                elif response_['status']=='SUCCEEDED':
                    
                    status = "execution succeeded"
                    
                
            else:
                
                status = "NO execution in progress"
            
    elif event["StepFunctionAPI"] == "stop":
        
        response = SFN_CLIENT.list_executions(
            stateMachineArn=STATE_MACHINE_ARN,
            statusFilter='RUNNING'
        )
        
        print(response)
        
        if response['executions']:
            
            execution_arn=response['executions'][0]['executionArn']
        
            response = SFN_CLIENT.stop_execution(executionArn=execution_arn)
            
            time.sleep(2)
            
            status = "execution stopped"
            
        else:
            
            status ="NO execution in progress"
            
    elif event["StepFunctionAPI"] == "metrics":
        
        response_deepar_tjlist = SM_CLIENT.list_training_jobs(
                                                                NameContains=event['TrainingJobNameDeepAR'],
                                                                StatusEquals='Completed',
                                                                SortBy='CreationTime',
                                                                SortOrder='Descending'
                                                            )

        
        response = SM_CLIENT.describe_training_job(TrainingJobName=response_deepar_tjlist['TrainingJobSummaries'][0]['TrainingJobName'])
        
        print(response['FinalMetricDataList'])
        metrics=[response['FinalMetricDataList'][0]['MetricName'],response['FinalMetricDataList'][0]['Value'],
                response['FinalMetricDataList'][4]['MetricName'],response['FinalMetricDataList'][4]['Value'],
                response['FinalMetricDataList'][6]['MetricName'],response['FinalMetricDataList'][6]['Value']]
        status="mertics retrieved"
        
    elif event["StepFunctionAPI"] == "delete": 
        
        # list existing endpoints
        response_endpoint = SM_CLIENT.list_endpoints(
                                        SortBy='CreationTime',
                                        SortOrder='Ascending',
                                        MaxResults=1,
                                        NameContains=event['EndpointName']
                                    )
    
        # pick up the latest one
        if response_endpoint['Endpoints']:
            
            ENDPOINT_NAME = response_endpoint['Endpoints'][0]['EndpointName']
        
            SM_CLIENT.delete_endpoint(EndpointName=ENDPOINT_NAME)
        
            status="endpoint deleted"
            
        else:
        
            status="endpoint already deleted"

    return {
        'response': status,
        'metrics':metrics,
        'state_id': int(state_id),
        'fail_details': fail_details
    }