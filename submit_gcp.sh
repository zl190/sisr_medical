GCS_BUCKET="gs://<your_bucket>"
IMAGE_URI=gcr.io/<PATH_STORE_DOCKER_IMAGE>
REGION=us-east1

docker build -f Dockerfile -t $IMAGE_URI .
docker push $IMAGE_URI

JOB_NAME=SISR_CT_$(date +%Y_%m_%d_%H%M%S)
JOB_DIR=$GCS_BUCKET"/"$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
  --master-image-uri $IMAGE_URI \
  --scale-tier custom \
  --master-machine-type standard_p100 \
  --region $REGION \
  --job-dir $JOB_DIR

gcloud ai-platform jobs describe $JOB_NAME