stack_name=$1
ecr_repository=$2
region=$3
bucket_backend=$4

# enable bucket versioning
aws s3api put-bucket-versioning --bucket ${bucket_backend} --versioning-configuration Status=Enabled

# upload the docker file to S3
aws s3 cp docker-pp/preprocessing.zip s3://${bucket_backend}/

# package the docker pipeline
aws cloudformation package --template-file ./docker-pipeline/cloudformation_docker.yml --s3-bucket ${bucket_backend}  --s3-prefix dashboard/build --region ${region} --output-template-file ./docker-pipeline/build/packaged.yaml

# deploy the docker pipeline
aws cloudformation deploy --template-file ./docker-pipeline/build/packaged.yaml --stack-name ${stack_name} --parameter-overrides SourceType="S3" RepositoryBucket=${bucket_backend} RepositoryNameECR=${ecr_repository} RepositoryObjectKey="preprocessing.zip" ComputeType="BUILD_GENERAL1_LARGE" --capabilities CAPABILITY_IAM --region ${region}
