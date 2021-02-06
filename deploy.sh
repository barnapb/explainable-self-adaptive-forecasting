#!/usr/bin/env bash

stack_name=$1
ecr_repository=$2
dashboard_ecr_repository=$3
region=$4
bucket_backend=$5
dashboard_name=$6
email=$7

if [ "$stack_name" == "" ] || [ "$ecr_repository" == "" ] || [ "$dashboard_ecr_repository" == "" ] || [ "$region" == "" ] || [ "$bucket_backend" == "" ] || [ "$dashboard_name" == "" ] || [ "$email" == "" ]
then
    echo "Usage: $0 <stack_name> <ecr_repository> <dashboard_ecr_repository> <region> <bucket_backend> <dashboard_name> <email>"
    exit 1
fi

aws configure set default.region ${region}

# Upload the preprocessing entrypoint script to s3
echo ""
echo "$(tput setaf 4)Upload the preprocessing entrypoint script to s3$(tput setaf 7)"
aws s3 cp src-backend/preprocessing.py s3://${bucket_backend}/config/code/

# Upload the GluonTS source code to s3
echo ""
echo "$(tput setaf 4)Upload the GluonTS source code to s3$(tput setaf 7)"
aws s3 cp src-backend/sourcedir.tar.gz s3://${bucket_backend}/config/sourcedir/

# Get the account number associated with the current IAM credentials
echo ""
echo "$(tput setaf 4)Get the account number associated with the current IAM credentials"
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

processing_repository_uri="${account}.dkr.ecr.${region}.amazonaws.com/${ecr_repository}:latest"

# # Build the preprocessing image
# echo ""
# echo "$(tput setaf 4)Build the preprocessing image$(tput setaf 7)"
# docker build -t ${ecr_repository} docker-pp

# # If the repository doesn't exist in ECR, create it.
# echo ""
# echo "$(tput setaf 4)Create the repository if it doesn't exist in ECR$(tput setaf 7)"
# aws ecr describe-repositories --region ${region} --repository-names "${ecr_repository}" > /dev/null 2>&1

# if [ $? -ne 0 ]
# then
#     aws ecr create-repository --region ${region} --repository-name "${ecr_repository}" > /dev/null
# fi

# # Get the login command from ECR and execute it directly
# aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

# # Push the docker image to ECR with the full name.
# echo ""
# echo "$(tput setaf 4)Push the preprocessing image to ECR$(tput setaf 7)"
# docker tag ${ecr_repository} ${processing_repository_uri}
# docker push ${processing_repository_uri}

# sam build
echo ""
echo "$(tput setaf 4)SAM build$(tput setaf 7)"
sam build -t cloudformation.yml


# sam deploy
echo ""
echo "$(tput setaf 4)SAM deploy$(tput setaf 7)"
OUTPUT=$(sam deploy --stack-name ${stack_name} --s3-bucket ${bucket_backend} --s3-prefix ${stack_name} --no-confirm-changeset --capabilities "CAPABILITY_IAM" --parameter-overrides region=${region} Environment="EASF" PreProcessingImageURI=${processing_repository_uri})
echo "Outputs SAM"
echo "${OUTPUT}"

declare -a arrayOutputs

sub_rm="Value";

while read -r line
do
    arrayOutputs+=("$line")            
done <<< "$OUTPUT"

for line in "${arrayOutputs[@]}"
do
    if [[ $line == *"-DeepARPredictLambda-"* ]]; then
        DeepARLambda=${line//$sub_rm/}
        DeepARLambda=${DeepARLambda//[[:blank:]]/}
        echo "${DeepARLambda}"
    fi
    if [[ $line == *"-UtilityLambda-"* ]]; then
        UtilityLambda=${line//$sub_rm/}
        UtilityLambda=${UtilityLambda//[[:blank:]]/}
        echo "${UtilityLambda}"
    fi
    if [[ $line == *"-FailureNotificationTopic-"* ]]; then
        FailureNotificationTopic=${line//$sub_rm/}
        FailureNotificationTopic=${FailureNotificationTopic//[[:blank:]]/}
        echo "${FailureNotificationTopic}"
    fi
done

if [ "$DeepARLambda" == "" ] || [ "$UtilityLambda" == "" ] || [ "$FailureNotificationTopic" == "" ]
then
    echo "$(tput setaf 5)SAM deployment failed$(tput setaf 7)"
    exit 1
fi

# dashboard NOT tested yet in other regions
frontend_region="us-west-2";

# create a bucket for the dashboard CF
echo ""
echo "$(tput setaf 5)Create a bucket for the dashboard CF$(tput setaf 7)"
bucket_dashboard="${dashboard_name}-$(date +%Y%m%d)" 
aws s3api create-bucket --bucket ${bucket_dashboard} --region ${frontend_region} --create-bucket-configuration LocationConstraint=${frontend_region}

# Package the dashboard cloud formation
echo ""
echo "$(tput setaf 5)Package the dashboard cloud formation$(tput setaf 7)"
aws cloudformation package --template-file ./dashboard-streamlit/cloudformation/template.yaml --s3-bucket ${bucket_dashboard}  --s3-prefix dashboard/build --region ${frontend_region} --output-template-file ./dashboard-streamlit/build/packaged.yaml

# deploy the dashboard cloud formation
echo ""
echo "$(tput setaf 5)Deploy the dashboard cloud formation$(tput setaf 7)"
aws cloudformation deploy --template-file ./dashboard-streamlit/build/packaged.yaml --stack-name ${dashboard_name} --parameter-overrides ResourceName=${dashboard_name} AddDevelopmentStack='false' CognitoAuthenticationSampleUserEmail=${email} SageMakerNotebookGitRepository=''  S3BucketUser=${bucket_backend} BackendRegion=${region} UtilityLambdaARN=${UtilityLambda} ServeLambdaARN=${DeepARLambda} DashboardRepositoryName=${dashboard_ecr_repository} --capabilities CAPABILITY_IAM --region ${frontend_region}

# build the dasboard app docker image
echo ""
echo "$(tput setaf 5)Build the dasboard app docker image$(tput setaf 7)"
docker build -t ${dashboard_ecr_repository} docker-dashboard

# # If the repository doesn't exist in ECR, create it.
# echo ""
# echo "$(tput setaf 4)Create the repository if it doesn't exist in ECR$(tput setaf 7)"
# aws ecr describe-repositories --region ${frontend_region} --repository-names "${dashboard_ecr_repository}" > /dev/null 2>&1

# if [ $? -ne 0 ]
# then
#     aws ecr create-repository --region ${frontend_region} --repository-name "${dashboard_ecr_repository}" > /dev/null
# fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${frontend_region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${frontend_region}.amazonaws.com

# push the image to ECR
echo ""
echo "$(tput setaf 5)Push the dashboard image to ECR$(tput setaf 7)"
dashboard_repository_uri="${account}.dkr.ecr.${frontend_region}.amazonaws.com/${dashboard_ecr_repository}:latest"

docker tag ${dashboard_ecr_repository} ${dashboard_repository_uri}
docker push ${dashboard_repository_uri}

#subscribe to the topic ARN
echo ""
echo "$(tput setaf 5)Subscribe to the topic ARN$(tput setaf 7)"
topicARN="arn:aws:sns:${region}:${account}:${FailureNotificationTopic}"
aws sns subscribe --topic-arn ${topicARN} --protocol email --notification-endpoint ${email}

# Upload the config file to s3
echo ""
echo "$(tput setaf 5)Upload the config file to s3$(tput setaf 7)"
aws s3 cp parameters.json s3://${bucket_backend}/config/

# Upload the sample file to s3
echo ""
echo "$(tput setaf 5)Upload the sample file to s3$(tput setaf 7)"
aws s3 cp Air_Pollution_in_Seoul.csv s3://${bucket_backend}/

aws configure set default.region ${frontend_region}
