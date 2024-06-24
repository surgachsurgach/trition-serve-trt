"""Run model_runner_sagemaker on Sagemaker."""
from __future__ import annotations

import dataclasses
import datetime
import enum
import json
from typing import Any

import boto3
from loguru import logger
import pydantic
import pydantic_settings
import sagemaker
from sagemaker import debugger
from sagemaker import estimator
from sagemaker.experiments import run as exp_run

logger.level("INFO")

_RESERVED_SAGEMAKER_CODE_PATH = "/opt/ml/code"
_RESERVED_SAGEMAKER_OUTPUT_LOCAL_PATH = "/opt/ml/output"
_RESERVED_SAGEMAKER_TENSORBOARD_OUTPUT_LOCAL_PATH = f"{_RESERVED_SAGEMAKER_OUTPUT_LOCAL_PATH}/tensorboard"
_ENTRY_POINT = "data/ml/model_runner/main.py"
# TODO(hyesung): inject rol arn sagemaker service will assume from environment variable..?
_SAGEMAKER_ROLE_ARN = "arn:aws:iam::368316345532:role/service-role/AmazonSageMaker-ExecutionRole-20231213T174537"
_SAGEMAKER_MODEL_OUTPUT_S3_PATH = "s3://sagemaker-us-east-1-368316345532"
_CODE_TAR = "code.tar.gz"


def _to_gin_arg(value: Any):
    if isinstance(value, str):
        if not (value.startswith("@") or value.startswith("%")):
            return f"'{value}'"
    return str(value).strip().replace(" ", "")


@dataclasses.dataclass
class GPU:
    cpu: int
    gpu: int


class GPUResource(enum.Enum):
    G5_XLARGE = GPU(cpu=3, gpu=1)
    G5_2XLARGE = GPU(cpu=7, gpu=1)
    G5_4XLARGE = GPU(cpu=14, gpu=1)
    G5_8XLARGE = GPU(cpu=30, gpu=1)
    G5_16XLARGE = GPU(cpu=60, gpu=1)

    @classmethod
    def parse(cls, value: str) -> GPUResource:
        # value : ml.g5.4xlarge -> G5_4XLARGE
        # remove ml
        value = "_".join(value.split(".")[1:])
        return cls[value.upper()]

    @property
    def gin_params(self):
        return {
            "DEVICES": self.value.gpu,
            "NUM_WORKERS": self.value.cpu,
        }


class Config(pydantic_settings.BaseSettings):
    aws_region: str = pydantic.Field(env="AWS_REGION", default="us-east-1")
    exp_name: str = pydantic.Field(env="EXP_NAME")
    gin_file: str = pydantic.Field(env="GIN_FILE")
    gin_params_key: str = pydantic.Field(default="", env="GIN_PARAMS_KEY")
    gin_params_json_path: str = pydantic.Field(default="", env="GIN_PARAMS_JSON_PATH")
    image_uri: str = pydantic.Field(env="IMAGE_URI")
    instance_type: str = pydantic.Field(env="INSTANCE_TYPE")
    instance_count: int = pydantic.Field(default=1, env="INSTANCE_COUNT")
    source_dir: str = pydantic.Field(env="SOURCE_DIR")
    slack_logging_url: str = pydantic.Field(env="SLACK_LOGGING_URL")
    user: str = pydantic.Field(env="USER")
    wait: bool = pydantic.Field(env="WAIT", default=True)

    class Config:
        env_file = (".env", ".env.model")
        extra = "allow"

    @property
    def run_datetime(self):
        return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    @property
    def gin_params(self):
        params = GPUResource.parse(self.instance_type).gin_params

        params.update(
            {
                "USER": self.user,
            }
        )

        if self.gin_params_json_path:
            with open(self.gin_params_json_path, "r", encoding="utf-8") as json_file:
                gin_params_dict = json.load(json_file)[self.gin_params_key]
            params.update(gin_params_dict)

        return [f"{key}={_to_gin_arg(value)}" for key, value in params.items()]

    @property
    def env_vars(self):
        return {
            "GIN_FILE": f"{_RESERVED_SAGEMAKER_CODE_PATH}/{self.gin_file}",
            "GIN_PARAMS": "|".join(self.gin_params),
            "EXP_NAME": self.exp_name,
            "TENSORBOARD_LOG_DIR": _RESERVED_SAGEMAKER_TENSORBOARD_OUTPUT_LOCAL_PATH,
            "SLACK_BATCH_LOGGING_URL": self.slack_logging_url,
        }


def main():
    config = Config()

    logger.info(
        f"aws_region: {config.aws_region}\n"
        f"exp_name: {config.exp_name}\n"
        f"gin_file: {config.gin_file}\n"
        f"gin_params_key: {config.gin_params_key}\n"
        f"gin_params_json_path: {config.gin_params_json_path}\n"
        f"image_uri: {config.image_uri}\n"
        f"instance_type: {config.instance_type}\n"
        f"instance_count: {config.instance_count}\n"
        f"source_dir: {config.source_dir}\n"
        f"user: {config.user}\n"
        f"wait: {config.wait}\n"
    )

    boto3_session = boto3.Session(region_name=config.aws_region)
    sagemaker_session = sagemaker.Session(boto3_session)

    # Debugger TensorBoard output config location:
    # f"{_SAGEMAKER_MODEL_OUTPUT_S3_PATH}/data-ml-recsys-2023-12-19-06-51-18-664/tensorboard-output/..."
    tensorboard_output_config = debugger.TensorBoardOutputConfig(
        s3_output_path=_SAGEMAKER_MODEL_OUTPUT_S3_PATH,
        container_local_output_path=_RESERVED_SAGEMAKER_TENSORBOARD_OUTPUT_LOCAL_PATH,
    )

    with exp_run.Run(
        experiment_name=config.exp_name,
        run_name=config.run_datetime,
        sagemaker_session=sagemaker_session,
    ):

        est = estimator.Estimator(
            image_uri=config.image_uri,
            source_dir=f"{config.source_dir}/{_CODE_TAR}",
            environment=config.env_vars,
            entry_point=_ENTRY_POINT,
            role=_SAGEMAKER_ROLE_ARN,
            instance_count=config.instance_count,
            instance_type=config.instance_type,
            # model artifact location: e.g. f"{_SAGEMAKER_MODEL_OUTPUT_S3_PATH}/2023-12-19-06-51-18-664/output/model.tar.gz"
            output_path=_SAGEMAKER_MODEL_OUTPUT_S3_PATH,
            sagemaker_session=sagemaker_session,
            tensorboard_output_config=tensorboard_output_config,
        )

        est.fit(
            wait=config.wait,
            job_name=f"{config.user}-{config.exp_name}-{config.run_datetime}",
        )


if __name__ == "__main__":
    main()
