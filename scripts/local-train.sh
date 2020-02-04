#!/bin/bash

echo "Training local ML model"

MODEL_NAME="generating_music" # change to your model name

PACKAGE_PATH=trainer
MODEL_DIR='.'


gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --num-epochs=10 \
        --batch-size=64 \
        --job-dir=${MODEL_DIR}
