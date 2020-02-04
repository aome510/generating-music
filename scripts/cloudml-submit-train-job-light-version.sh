#!/bin/bash

echo "Submitting an AI Platform job..."

REGION="us-central1" # choose a gcp region from https://cloud.google.com/ml-engine/docs/tensorflow/regions
TIER=BASIC_GPU # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
BUCKET="aome-bucket-01" # change to your bucket name

MODEL_NAME="generating_music" # change to your model name

PACKAGE_PATH=trainer # this can be a gcs location to a zipped and uploaded package
MODEL_DIR=gs://${BUCKET}/${MODEL_NAME}

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${CURRENT_DATE}

gcloud ai-platform jobs submit training ${JOB_NAME} \
        --job-dir=${MODEL_DIR} \
        --python-version 3.5 \
        --runtime-version=1.13 \
        --region=${REGION} \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --num-epochs=25 \
        --batch-size=64 \

# notes:
# use --packages instead of --package-path if gcs location
# add --reuse-job-dir to resume training
