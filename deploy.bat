:: DON'T USE, NOT TESTED YET
@ECHO OFF
SETLOCAL EnableDelayedExpansion

SET stack_name=%1
SET ecr_repository=%2
SET dashboard_ecr_repository=%3
SET region=%4
SET bucket_backend=%5
SET dashboard_name=%6
SET email=%7
SET _tempvar=0

IF %stack_name%=="" SET _tempvar=1
IF %ecr_repository%=="" SET _tempvar=1
IF %dashboard_ecr_repository%=="" SET _tempvar=1
IF %region%=="" SET _tempvar=1
IF %bucket_backend%=="" SET _tempvar=1
IF %dashboard_name%=="" SET _tempvar=1
IF %email%=="" SET _tempvar=1
IF %_tempvar% EQU 1 (
    ECHO "Usage: %0 <stack_name> <ecr_repository> <dashboard_ecr_repository> <region> <bucket_backend> <dashboard_name> <email>"
    SET _tempvar=0
    exit 1
)


:: Upload the preprocessing entrypoint script to s3
echo Upload the preprocessing entrypoint script to s3
aws s3 cp src-backend/preprocessing.py s3://%bucket_backend%/config/code/

:: Upload the GluonTS source code to s3
echo Upload the GluonTS source code to s3
aws s3 cp src-backend/sourcedir.tar.gz s3://%bucket_backend%/config/sourcedir/

SET account=0
for /f %%i in ('aws sts get-caller-identity --query Account --output text') do set account=%%i

echo The AWS Account is %account%
IF %account%==0 (
    ECHO "AWS Account Number cannot be identified."
    exit 255
)

SET processing_repository_uri="%account%.dkr.ecr.%region%.amazonaws.com/%ecr_repository%:latest"
echo %processing_repository_uri%

aws configure set default.region %region%

:: sam build
echo SAM build
sam build -t cloudformation.yml

:: sam deploy
echo SAM deploy
SET output=""
for /f %%i in ('sam deploy --stack-name %stack_name% --s3-bucket %bucket_backend% --s3-prefix %stack_name% --no-confirm-changeset --capabilities "CAPABILITY_IAM" --parameter-overrides region=%region% Environment="EASF" PreProcessingImageURI=%processing_repository_uri%') do set output=%%i

>log.txt echo %output%

SET DeepARLambda=""
for /f %%i in ('python parser.py log.txt -DeepARPredictLambda-') do set DeepARLambda=%%i

SET UtilityLambda=""
for /f %%i in ('python parser.py log.txt -UtilityLambda-') do set UtilityLambda=%%i

SET FailureNotificationTopic=""
for /f %%i in ('python parser.py log.txt -FailureNotificationTopic-') do set FailureNotificationTopic=%%i

SET tempvar=0
IF %DeepARLambda%=="" SET tempvar=1
IF %UtilityLambda%=="" SET tempvar=1
IF %FailureNotificationTopic%=="" SET tempvar=1
IF %tempvar% EQU 1 (
    ECHO SAM deployment failed
    SET tempvar=0
    exit 1
)

:: dashboard NOT tested yet in other regions
set frontend_region="us-west-2"

:: create a bucket for the dashboard CF

SET bucket_dashboard=%dashboard_name%-%date:~6,4%-%date:~3,2%-%date:~0,2%

echo Create a bucket for the dashboard

aws s3api create-bucket --bucket %bucket_dashboard% --region %frontend_region% --create-bucket-configuration LocationConstraint=%frontend_region%

::  Package the dashboard cloud formation
echo Package the dashboard cloud formation
aws cloudformation package --template-file ./dashboard-streamlit/cloudformation/template.yaml --s3-bucket %bucket_dashboard%  --s3-prefix dashboard/build --region %frontend_region% --output-template-file ./dashboard-streamlit/build/packaged.yaml

:: deploy the dashboard cloud formation
echo Deploy the dashboard cloud formation
aws cloudformation deploy --template-file ./dashboard-streamlit/build/packaged.yaml --stack-name %dashboard_name% --parameter-overrides ResourceName=%dashboard_name% AddDevelopmentStack='false' CognitoAuthenticationSampleUserEmail=%email% SageMakerNotebookGitRepository=''  S3BucketUser=%bucket_backend% BackendRegion=%region% UtilityLambdaARN=%UtilityLambda% ServeLambdaARN=%DeepARLambda% DashboardRepositoryName=%dashboard_ecr_repository% --capabilities CAPABILITY_IAM --region %frontend_region%

:: build the dasboard app docker image
echo Build the dasboard app docker image
docker build -t %dashboard_ecr_repository% docker-dashboard

:: Installing AWS Tools for Powershell
:: https://docs.aws.amazon.com/powershell/latest/userguide/pstools-getting-set-up-windows.html
powershell -Command "if (-not (Get-PackageProvider -Name 'NuGet')) {Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 -Force}"
powershell -Command "if (-not (Get-Module -Name 'AWS.Tools.Installer')) {Install-Module -Confirm:$false -Force -Name AWS.Tools.Installer}"
powershell -Command "Install-AWSToolsModule -Confirm:$false AWS.Tools.ECR" 
powershell -Command "if (-not (Get-Module -Name 'AWSPowerShell')) {Install-Package -Confirm:$false -Force -Name AWSPowerShell}"

:: Get the login command from ECR and execute it directly
powershell -Command "(Get-ECRLoginCommand -Region %frontend_region%).Password | docker login --username AWS --password-stdin %account%.dkr.ecr.%frontend_region%.amazonaws.com"

:: Push the docker image to ECR with the full name.
:: docker tag %image% %fullname%
:: docker push %fullname%

SET dashboard_repository_uri="%account%.dkr.ecr.%frontend_region%.amazonaws.com/%dashboard_ecr_repository%:latest"
:: push the image to ECR
echo Push the dashboard image to ECR
docker tag %dashboard_ecr_repository% %dashboard_repository_uri%
docker push %dashboard_repository_uri%

:: subscribe to the topic ARN
echo Subscribe to the topic ARN
SET topicARN="arn:aws:sns:%region%:%account%:%FailureNotificationTopic%"
aws sns subscribe --topic-arn %topicARN% --protocol email --notification-endpoint %email%

:: Upload the config file to s3
echo Upload the config file to s3
aws s3 cp parameters.json s3://%bucket_backend%/config/

:: Upload the sample file to s3
echo Upload the sample file to s3
aws s3 cp Air_Pollution_in_Seoul.csv s3://%bucket_backend%/

aws configure set default.region %frontend_region%





