
dashboard_ecr_repository=$1

if [ "$dashboard_ecr_repository" == "" ]
then
    echo "Usage: $0 <dashboard_ecr_repository>"
    exit 1
fi

# dashboard NOT tested yet in other regions
region="us-west-2"

account=$(aws sts get-caller-identity --query Account --output text)

# build the dasboard app docker image
echo ""
echo "$(tput setaf 5)Build the dasboard app docker image$(tput setaf 7)"
docker build -t ${dashboard_ecr_repository} docker-dashboard

# push the image to ECR
echo ""
echo "$(tput setaf 5)Push the dashboard image to ECR$(tput setaf 7)"
dashboard_repository_uri="${account}.dkr.ecr.${region}.amazonaws.com/${dashboard_ecr_repository}:latest"

docker tag ${dashboard_ecr_repository} ${dashboard_repository_uri}
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com
#eval $(aws ecr get-login --no-include-email --region ${frontend_region})
docker push ${dashboard_repository_uri}