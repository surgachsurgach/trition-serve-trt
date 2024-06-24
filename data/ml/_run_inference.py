"""Run Torch Serve on Sagemaker for model inferenece.
We use batch-inference mode for now.
Find more detail about
- inference options: https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html#deploy-model-options.
- TorchServe: https://pytorch.org/serve/
"""

import argparse

import boto3
from loguru import logger
import pydantic
import pydantic_settings
import sagemaker
from sagemaker import model as sagemaker_model

logger.level("INFO")

_SAGEMAKER_ROLE_ARN = "arn:aws:iam::368316345532:role/service-role/AmazonSageMaker-ExecutionRole-20231213T174537"
_ENTRY_POINT = "inference.py"
_CODE_TAR = "code.tar.gz"


class Config(pydantic_settings.BaseSettings):
    aws_region: str = pydantic.Field(default="us-east-1", env="AWS_REGION")
    output_path: str = pydantic.Field(env="OUTPUT_PATH")
    source_dir: str = pydantic.Field(env="SOURCE_DIR")
    image_uri: str = pydantic.Field(env="IMAGE_URI")
    inference_instance_type: str = pydantic.Field(default="ml.c4.2xlarge", env="INFERENCE_INSTANCE_TYPE")
    inference_instance_count: int = pydantic.Field(default=1, env="INFERENCE_INSTANCE_COUNT")
    sample_inference_data_dir: str = pydantic.Field(env="SAMPLE_INFERENCE_DATA_DIR")
    sample_inference_meta_dir: str = pydantic.Field(env="SAMPLE_INFERENCE_META_DIR")
    slack_logging_url: str = pydantic.Field(env="SLACK_LOGGING_URL")
    pytorch_model_path: str = pydantic.Field(env="PYTORCH_MODEL_PATH")
    sagemaker_triton_default_model_name: str = pydantic.Field(default=None, env="SAGEMAKER_TRITON_DEFAULT_MODEL_NAME")
    batch_mode: bool = pydantic.Field(default=True, env="BATCH_MODE")
    wait: bool = pydantic.Field(default=True, env="WAIT")

    class Config:
        env_file = (".env", ".env.inference")
        extra = "allow"


def run_torchserve(config: Config):
    logger.info(
        f"region: {config.aws_region}\n"
        f"source_dir: {config.source_dir}\n"
        f"image_uri: {config.image_uri}\n"
        f"instance_type: {config.inference_instance_type}\n"
        f"instance_count: {config.inference_instance_count}\n"
        f"sample_inference_data_dir: {config.sample_inference_data_dir}\n"
        f"sample_inference_meta_dir: {config.sample_inference_meta_dir}\n"
        f"role_arn: {_SAGEMAKER_ROLE_ARN}\n"
        f"pytorch_model_path: {config.pytorch_model_path}\n"
    )
    boto3_session = boto3.Session(region_name=config.aws_region)
    sagemaker_session = sagemaker.Session(boto3_session)

    model = sagemaker_model.Model(
        image_uri=config.image_uri,
        model_data=config.pytorch_model_path,
        sagemaker_session=sagemaker_session,
        env={
            "SLACK_BATCH_LOGGING_URL": config.slack_logging_url,
            "META_PATH": config.sample_inference_meta_dir,  # meta.json must be in this directory.
        },
        role=_SAGEMAKER_ROLE_ARN,
        # It does not actually replace Docker entrypoint, but it is used to set SAGEMAKER_PROGRAM env
        # which is used by MMS Transformer to replace default handler method.
        # For more detail:
        # https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_inference/transformer.py#L200
        entry_point=_ENTRY_POINT,
        source_dir=f"{config.source_dir}/{_CODE_TAR}",
    )

    if config.batch_mode:
        logger.info("Running batch-inference mode.")

        transformer = model.transformer(
            instance_count=config.inference_instance_count,
            instance_type=config.inference_instance_type,
            output_path=config.output_path,
        )

        transformer.transform(
            data=config.sample_inference_data_dir,
            data_type="S3Prefix",
            content_type="application/json",
            wait=config.wait,
        )
    else:
        logger.info("Deploying model in real-time mode.")
        model.deploy(
            initial_instance_count=config.inference_instance_count,
            instance_type=config.inference_instance_type,
            wait=config.wait,
        )

        logger.info(
            "Model deployed successfully."
            "You can now use the endpoint to get inference from the model."
            "Here is an example code to get inference from the model:"
        )
        logger.info(
            "\nimport boto3\n"
            f"boto3_session = boto3.Session(region_name='{config.aws_region}')\n"
            f"sagemaker_runtime = boto3_session.client('sagemaker-runtime')\n"
            f"endpoint_name = '{model.endpoint_name}'\n"
            f"response = sagemaker_runtime.invoke_endpoint(\n"
            f"   EndpointName='{model.endpoint_name}',\n"
            "   ContentType='application/json',\n"
            '   Body="{"item_id": ["1255", "1915", "1828", "2240", "2205", "1386", "1252"]}\',\n'
            ")\n"
        )


parser = argparse.ArgumentParser(description="Run Torch Serve on Sagemaker for model inferenece.")
parser.add_argument("--run-torchserve", action="store_true")
parser.add_argument("--run-trition", action="store_true")


def main():
    args = parser.parse_args()
    config = Config()

    if args.run_torchserve:
        run_torchserve(config)

    if args.run_trition:
        raise NotImplementedError("Triton is not implemented yet.")


if __name__ == "__main__":
    main()
