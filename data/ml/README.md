# Data ML
Deep Learning Model and Inference for Data Team

## Base Setup
### .env
```.dotenv
AWS_INFRA_PROFILE="{{ YOUR_AWS_INFRA_PROFILE }}"
AWS_REGION="us-east-1"
DATA_ACCOUNT="prod-data"
USER="{{ YOUR_NAME }}"
EXP_NAME="{{ YOUR_EXP_NAME }}"
SLACK_LOGGING_URL="{{ data-batch-stage URL }}"
```

### base image
see data/docker projects
https://github.com/ridi/ridi/tree/master/data/docker. 

## Model Setup and Running
https://github.com/aws/deep-learning-containers/blob/master/available_images.md

### .env.model
예시
```.dotenv
IMAGE_URI="801714584815.dkr.ecr.us-east-1.amazonaws.com/data-ml-recsys:model-soohyun"
INSTANCE_TYPE="ml.g5.4xlarge"
INSTANCE_COUNT="1"
GIN_FILE="data/ml/model_runner/configs/config/users/soohyun/vae/model.gin"
GIN_PARAMS_KEY="bl_novel"
GIN_PARAMS_JSON_PATH="env/model_gin_params.json"
WAIT="False"
```

- gin params json 파일 예시
    ```json
    {
        "bl_novel": {
            "TRAIN": true,
            "PREDICT": true,
            "USER": "soohyun",
            "DATE": "2024-06-16",
            "PARTITION": "date=2024-06-16/interaction_type=purchase/version_tag=book2user_3year_20240408/genre=bl_novel",
            "MODEL_NAME": "vae",
            "DATA_TAG": "default",
            "MODEL_TAG": "default",
            "TABLE_NAME": "feature_user_book_interaction_partitioned",
            "PREDICT_TOP_K": 200
        }
    }
    ```


### deploy image
첫 1회만 하면 됩니다. 이후 requirements.model.txt에 수정사항이 있을 때만 다시 실행하면 됩니다.
```shell
$ make deploy-image-model
```

### run
소스코드를 tar 파일로 압축하여 s3에 업로드하고, SageMaker Training Job을 실행합니다.
```shell
$ make py-run-model  # deploy and run
$ make run-model  # run only
```

## Data Setup and Running
data etl project와 동일하게 AWS emr-containers 서비스를 사용합니다.

### .env.data
예시
```.dotenv
# Infrastructure Configurations
TARGET_ZONE="us-east-1a"
# Task Configurations
GIN_FILE="data/ml/configs/config/users/hyesung/manta_vae_data_processor.gin"
TABLE_NAME="manta_user_item_interaction_partitioned"
DATABASE="stage.db"  # .db suffix를 꼭 붙여주세요.
DATA_TAG=""  # 식별할 tag를 자유롭게 입력해주세요.
PARTITION_PATH="date=2023-11-22"

# EMR Configurations
DRIVER_MEMORY=2G
EXECUTOR_INSTANCE_TYPE="m6g.xlarge"
EXECUTOR_MEMORY=9G
NUM_EXECUTOR_CORES=3
NUM_EXECUTORS=1
MAX_ATTEMPTS=1
VOLUME_SIZE="8Gi"
```

### deploy image
```shell
$ make deploy-image-data
```

### run
```shell
$ make py-run-data  # deploy and run
$ make run-data  # run only
```

## Inference Setup and Running
batch inference, real-time inference 두 가지 방법으로 실행할 수 있습니다. 
팀에서 사용하는 Prediction Model을 SageMaker Endpoint로 배포하고, Inference를 실행합니다.
이는 실험적 기능으로 로컬에서만 실행 가능합니다.

### .env.inference
예시
```.dotenv
IMAGE_URI="801714584815.dkr.ecr.us-east-1.amazonaws.com/data-ml-recsys:inference-hyesung"
INFERENCE_INSTANCE_TYPE="ml.c4.2xlarge"
INFERENCE_INSTANCE_COUNT="1"
HANDLER_PATH="data/ml/sagemaker/inference/handler/vae_cf.py"
PYTORCH_MODEL_PATH="s3://sagemaker-us-east-1-368316345532/data-ml-recsys-2024-04-03-08-44-02-219/output/model.tar.gz"
OUTPUT_PATH="s3://sagemaker-us-east-1-368316345532/batch_transform_output"
SAMPLE_INFERENCE_DATA_DIR="s3://sagemaker-us-east-1-368316345532/batch_transform_input"
SAMPLE_INFERENCE_META_DIR="s3://sagemaker-us-east-1-368316345532"
BATCH_MODE="False"
WAIT="False"
```

### deploy image
마찬가지로 최초 1회 실행 필요합니다.
```shell
$ make deploy-image-inference
```

### run
```shell
$ make py-run-inference  # deploy and run
$ make run-inference  # run only
```