""" Run EMR Container on EKS """

import http
import json
import os
from typing import Any

import boto3
from loguru import logger
import pydantic
import pydantic_settings

from data.pylib import aws_utils
from data.pylib import strenum

_CLOUDWATCH_LOG_GROUP = "/aws/eks/emr"
_EBS_STORAGE_CLASS_NAME = "gp3"
_EMR_ENV_VARS_JSON = "emr_env_vars.json"
# _JARS_FILE = "jars.txt"
# _JARS_LOCAL_PATH = "local:///opt/spark/jars"
# _PACKAGES_FILE = "packages.txt"
_RELEASE_LABEL = "emr-6.10.0-20230421"  # The Amazon EMR release version to use for the job run.
_IRSA_NAME = "eks-ridi-data-serviceaccounts"
_AWS_REGION = "us-east-1"
_ENTRY_POINT = "data/ml/data_runner/main.py"

_AVAILABLE_ZONES = [
    f"{_AWS_REGION}a",
    f"{_AWS_REGION}b",
    f"{_AWS_REGION}c",
]

_PROD_DATA_SUBNETS = [
    "subnet-046207d41cbbf5137",
    "subnet-080d4dbea45b84815",
    "subnet-01c6ba9dee54e4d21",
]


def _get_valid_pod_name_prefix(name: str) -> str:
    """It must conform the rules defined by the Kubernetes DNS Label Names.

    executor pod names in the form of podNamePrefix−exec−id, where the `id` is a positive int value,
    so the length of the `podNamePrefix` needs to be less than or equal to 47(= 63 - 10 - 6).
    """

    padding = "0"  # prevent pod name from being ended with `-`.
    return name[:45].lower().replace("_", "-").replace(".", "-") + padding


def _to_gin_arg(value: Any):
    if isinstance(value, str):
        if not (value.startswith("@") or value.startswith("%")):
            return f"'{value}'"
    return str(value).strip().replace(" ", "")


class EksVirtualCluster(strenum.StrEnum):
    # TODO(hyesung): modify here.
    # create virtual cluster, see https://docs.aws.amazon.com/ko_kr/emr/latest/EMR-on-EKS-DevelopmentGuide/setting-up-registration.html
    # we need to register emr-containers to eks cluster https://github.com/ridi/ridi-devops/pull/5350.
    PROD_DATA = "sv65iwkxaznrv44txnx4xqrx1"


class EmrContainerJobConfig(pydantic_settings.BaseSettings):
    """
    - To use pvc for dynamic allocation (Currently not use dynamic allocation), add below configs to spark default properties
        "spark.kubernetes.driver.volumes.persistentVolumeClaim.data.mount.path": "/data",
        "spark.kubernetes.driver.volumes.persistentVolumeClaim.data.mount.readOnly": "false",
        "spark.kubernetes.driver.volumes.persistentVolumeClaim.data.options.claimName": "OnDemand",
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.data.mount.path": "/data",
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.data.mount.readOnly": "false",
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.data.options.claimName": "OnDemand",
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.data.options.sizeLimit": self.volume_size,
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.data.options.storageClass": _EBS_STORAGE_CLASS_NAME,
    - To use pvc for checkpoint, add below configs to spark default properties
        "spark.kubernetes.driver.volumes.persistentVolumeClaim.checkpointpvc.mount.path": "/checkpoint"
        "spark.kubernetes.driver.volumes.persistentVolumeClaim.checkpointpvc.mount.readOnly": "false"
        "spark.kubernetes.driver.volumes.persistentVolumeClaim.checkpointpvc.mount.subPath": "checkpoint"
        "spark.kubernetes.driver.volumes.persistentVolumeClaim.checkpointpvc.options.claimName": "OnDemand",
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.checkpointpvc.mount.path": "/checkpoint"
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.checkpointpvc.mount.readOnly": "false"
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.checkpointpvc.mount.subPath": "checkpoint"
        "spark.kubernetes.executor.volumes.persistentVolumeClaim.checkpointpvc.options.claimName": "OnDemand",
    """

    # Task Configurations
    data_account: str = pydantic.Field("stage-data", env="DATA_ACCOUNT")
    user: str = pydantic.Field(..., env="USER")
    # gin_params
    data_tag: str = pydantic.Field(env="DATA_TAG")
    gin_file: str = pydantic.Field(env="GIN_FILE")
    gin_params: list[str] = pydantic.Field([], env="GIN_PARAMS")
    gin_params_json_path: str = pydantic.Field("", env="GIN_PARAMS_JSON_PATH")
    table_name: str = pydantic.Field(env="TABLE_NAME")
    partition_path: str = pydantic.Field(env="PARTITION_PATH")
    database: str = pydantic.Field("", env="DATABASE")
    # EMR Container Spec
    driver_instance_type: str = pydantic.Field("c6g.xlarge", env="DRIVER_INSTANCE_TYPE")
    driver_memory: str = pydantic.Field("1G", env="DRIVER_MEMORY")
    executor_instance_type: str = pydantic.Field("c6g.xlarge", env="EXECUTOR_INSTANCE_TYPE")
    executor_memory: str = pydantic.Field("1G", env="EXECUTOR_MEMORY")
    num_driver_cores: int = pydantic.Field(1, env="NUM_DRIVER_CORES")
    num_executor_cores: int = pydantic.Field(1, env="NUM_EXECUTOR_CORES")
    num_executors: int = pydantic.Field(1, env="NUM_EXECUTORS")
    volume_size: str = pydantic.Field("", env="VOLUME_SIZE")
    target_zone: str = pydantic.Field("", env="TARGET_ZONE")
    # ETC
    max_attempts: int = pydantic.Field(1, env="MAX_ATTEMPTS")
    use_instance_family: bool = pydantic.Field(False, env="USE_INSTANCE_FAMILY")

    class Config:
        env_file = (".env", ".env.data")
        extra = "allow"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if self.data_account == "prod-data":
            self.execution_account_id = aws_utils.Account.prod_data
            self.s3_bucket = "ridi-ml-batch"
            self.virtual_cluster_id = EksVirtualCluster.PROD_DATA
            self.subnet_ids = _PROD_DATA_SUBNETS
        else:
            raise ValueError(f"Unknown data account: {self.data_account}")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: pydantic_settings.BaseSettings,
        init_settings: pydantic_settings.PydanticBaseSettingsSource,
        env_settings: pydantic_settings.PydanticBaseSettingsSource,
        dotenv_settings: pydantic_settings.PydanticBaseSettingsSource,
        file_secret_settings: pydantic_settings.PydanticBaseSettingsSource,
    ) -> tuple[pydantic_settings.PydanticBaseSettingsSource, ...]:
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    @property
    def emr_env_vars_json(self) -> dict[str, dict[str, str]]:
        if os.path.isfile(_EMR_ENV_VARS_JSON):
            with open(_EMR_ENV_VARS_JSON, "r", encoding="utf-8") as env_file:
                return json.load(env_file)
        return {}

    @property
    def emr_envs(self) -> dict[str, str]:
        return self.emr_env_vars_json.get("envs", {})

    @property
    def emr_secrets(self) -> dict[str, str]:
        return self.emr_env_vars_json.get("secrets", {})

    @property
    def execution_role_arn(self) -> str:
        return f"arn:aws:iam::{self.execution_account_id}:role/{_IRSA_NAME}"

    @property
    def s3_deployment_path(self) -> str:
        return f"s3://{self.s3_bucket}/env/users/{self.user}"

    @property
    def image_uri(self) -> str:
        return f"{aws_utils.Account.infra}.dkr.ecr.{_AWS_REGION}.amazonaws.com/data-ml-recsys:data-{self.user}"

    @property
    def target_script(self) -> str:
        return self.entrypoint.split("/")[-1]

    @property
    def tags(self) -> dict[str, str]:
        return {
            "user": self.user,
            "app": self.entrypoint.split("/")[-1],
        }

    @property
    def entrypoint(self):
        return f"{self.s3_deployment_path}/{_ENTRY_POINT}"

    @property
    def py_files(self):
        return f"{self.s3_deployment_path}/src.zip"

    def _get_spark_submit_params(self, catalog_id: str) -> str:
        params = [
            "--py-files",
            self.py_files,
            "--conf",
            f"spark.driver.memory={self.driver_memory}",
            "--conf",
            f"spark.driver.cores={self.num_driver_cores}",
            "--conf",
            f"spark.executor.cores={self.num_executor_cores}",
            "--conf",
            f"spark.executor.instances={self.num_executors}",
            "--conf",
            f"spark.executor.memory={self.executor_memory}",
            "--conf",
            "spark.hive.metastore.client.factory.class=com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory",
            "--conf",
            f"spark.hive.metastore.glue.catalogid={catalog_id}",
            "--conf",
            # Without "BucketOwnerFullControl", the object created by prod/stage-data EMR cannot be read/writen
            # by other account except creator. Since s3://ridi-emr bucket ownership setting is "Bucket owner preferred",
            # the bucket owner also owns the ownership of the object, regardless of the object's creator,
            # https://docs.aws.amazon.com/ko_kr/emr/latest/ManagementGuide/emr-s3-acls.html
            # https://docs.aws.amazon.com/AmazonS3/latest/userguide/about-object-ownership.html
            "spark.hadoop.fs.s3.canned.acl=BucketOwnerFullControl",
        ]

        return " ".join(params)

    def get_job_name(self, suffix: str | None = None) -> str:
        base_job_name = f"{self.user}-{self.target_script}"
        if suffix:
            return f"{base_job_name}-{suffix}"
        return base_job_name

    def get_job_driver_params(self, catalog_id: str) -> dict:
        """Get job driver parameters for EMR Container

        Args:
            - catalog_id: Catalog ID of glue in which to get hive metastore. (= AWS account ID)

        Returns:
            - entryPoint: The path to the script that is executed when the container starts.
            - entryPointArguments: The arguments to pass to the script that is executed when the container starts.
            - sparkSubmitParameters: The parameters for spark submit.
        """
        return {
            "sparkSubmitJobDriver": {
                "entryPoint": f"{self.entrypoint}",
                "sparkSubmitParameters": self._get_spark_submit_params(catalog_id),
            },
        }

    @property
    def _gin_params(self):
        params = {
            "DATA_TAG": self.data_tag,
            "USER": self.user,
            "TABLE_NAME": self.table_name,
            "PARTITION": self.partition_path,
            "DATABASE": self.database,
        }

        if self.gin_params_json_path:
            with open(self.gin_params_json_path, "r", encoding="utf-8") as f:
                params.update(json.load(f))

        gin_params = [f"{key}={_to_gin_arg(value)}" for key, value in params.items()]
        gin_params.extend(self.gin_params)
        param_keys = [param.split("=")[0] for param in gin_params]
        key_counts = {key: param_keys.count(key) for key in param_keys}
        duplicated_keys = [key for key, count in key_counts.items() if count > 1]
        if duplicated_keys:
            logger.warning(f"Duplicated gin keys: {duplicated_keys}")
        return gin_params

    def get_configuration_overrides_params(self, dynamic_allocation: bool = False) -> dict:
        assert self.target_zone in _AVAILABLE_ZONES
        # pylint: disable=line-too-long
        spark_default_properties = {
            "spark.dynamicAllocation.enabled": str(dynamic_allocation).lower(),
            "spark.kubernetes.container.image": self.image_uri,
            "spark.kubernetes.driver.node.selector.karpenter.sh/capacity-type": "on-demand",
            "spark.kubernetes.driver.node.selector.node.kubernetes.io/instance-type": self.driver_instance_type,
            "spark.kubernetes.executor.podNamePrefix": f"{_get_valid_pod_name_prefix(self.user + '-' + self.target_script)}",
            "spark.kubernetes.executor.node.selector.karpenter.sh/capacity-type": "spot",
            "spark.kubernetes.node.selector.kubernetes.io/arch": "arm64",
            "spark.kubernetes.node.selector.kubernetes.io/os": "linux",
            "spark.kubernetes.node.selector.topology.kubernetes.io/zone": self.target_zone,
            # TODO(hyesung): default secrets and envs
            "spark.kubernetes.driverEnv.GIN_PARAMS": "|".join(self._gin_params),
            "spark.kubernetes.driverEnv.GIN_FILE": self.gin_file,
            "spark.kubernetes.driver.secretKeyRef.SLACK_BATCH_LOGGING_URL": "data-secrets:SLACK_BATCH_STAGE_LOGGING_URL",
            # "spark.kubernetes.driver.secretKeyRef.SLACK_DATA_REPORT_URL": "data-secrets:SLACK_DATA_REPORT_URL",
            # configuration for prometheus service discovery and metric endpoint
            "spark.ui.prometheus.enabled": "true",
            "spark.executor.processTreeMetrics.enabled": "true",
            "spark.kubernetes.driver.annotation.prometheus.io/scrape": "true",
            "spark.kubernetes.driver.annotation.prometheus.io/path": "/metrics/executors/prometheus/",
            "spark.kubernetes.driver.annotation.prometheus.io/port": "4040",
            "spark.kubernetes.driver.service.annotation.prometheus.io/scrape": "true",
            "spark.kubernetes.driver.service.annotation.prometheus.io/path": "/metrics/driver/prometheus/",
            "spark.kubernetes.driver.service.annotation.prometheus.io/port": "4040",
            "spark.metrics.conf.*.sink.prometheusServlet.class": "org.apache.spark.metrics.sink.PrometheusServlet",
            "spark.metrics.conf.*.sink.prometheusServlet.path": "/metrics/driver/prometheus/",
            "spark.metrics.conf.master.sink.prometheusServlet.path": "/metrics/master/prometheus/",
            "spark.metrics.conf.applications.sink.prometheusServlet.path": "/metrics/applications/prometheus/",
            "spark.sql.hive.convertMetastoreParquet": "true",
            "spark.sql.hive.convertMetastoreOrc": "true",
            "spark.sql.parquet.fs.optimized.committer.optimization-enabled": "true",
            "spark.sql.parquet.output.committer.class": "com.amazon.emr.committer.EmrOptimizedSparkSqlParquetOutputCommitter",
            "spark.sql.sources.commitProtocolClass": "org.apache.spark.sql.execution.datasources.SQLEmrOptimizedCommitProtocol",
            "spark.sql.storeAssignmentPolicy": "LEGACY",
            "spark.sql.legacy.timeParserPolicy": "LEGACY",
            "spark.sql.parquet.int96RebaseModeInWrite": "LEGACY",
            "spark.sql.parquet.int96RebaseModeInRead": "LEGACY",
            "spark.sql.parquet.datetimeRebaseModeInWrite": "CORRECTED",
            "spark.sql.parquet.datetimeRebaseModeInRead": "CORRECTED",
        }

        if self.use_instance_family:
            # To use AWS Spot instances, it is recommended to have a wide range of instance types available for use.
            instance_family = self.executor_instance_type.split(".")[0]
            spark_default_properties.update(
                {
                    "spark.kubernetes.executor.node.selector.karpenter.k8s.aws/instance-family": instance_family,
                }
            )
        if not self.use_instance_family:
            spark_default_properties.update(
                {
                    "spark.kubernetes.executor.node.selector.node.kubernetes.io/instance-type": self.executor_instance_type,
                }
            )

        # Environment variable & Secret
        for key, value in self.emr_envs.items():
            spark_default_properties[f"spark.kubernetes.driverEnv.{key}"] = value
        for key, value in self.emr_secrets.items():
            spark_default_properties[f"spark.kubernetes.driver.secretKeyRef.{key}"] = value

        # Add EBS volume as a persistent volume
        # https://spark.apache.org/docs/latest/running-on-kubernetes.html#using-kubernetes-volumes
        # You can also use custom pvc.
        if self.volume_size:
            spark_default_properties.update(
                {
                    # Use pvc for spill
                    "spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-local-dir-spill.mount.path": "/var/data/spill",
                    "spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-local-dir-spill.mount.readOnly": "false",
                    "spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-local-dir-spill.options.claimName": "OnDemand",
                    "spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-local-dir-spill.options.sizeLimit": self.volume_size,
                    "spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-local-dir-spill.options.storageClass": _EBS_STORAGE_CLASS_NAME,
                }
            )

        # https://aws.github.io/aws-emr-containers-best-practices/node-placement/docs/eks-node-placement/#job-submitter-pod-placement
        # To reduce node provisioning time, unify the settings for driver and job submitter that require fewer resources.
        emr_job_submitter_properties = {
            "jobsubmitter.node.selector.karpenter.sh/capacity-type": "on-demand",
            "jobsubmitter.node.selector.node.kubernetes.io/instance-type": self.driver_instance_type,
            "jobsubmitter.node.selector.topology.kubernetes.io/zone": self.target_zone,
        }
        return {
            "applicationConfiguration": [
                {
                    "classification": "spark-defaults",
                    "properties": spark_default_properties,
                },
                {
                    "classification": "emr-job-submitter",
                    "properties": emr_job_submitter_properties,
                },
            ],
            "monitoringConfiguration": {
                "persistentAppUI": "ENABLED",
                "cloudWatchMonitoringConfiguration": {
                    "logGroupName": _CLOUDWATCH_LOG_GROUP,
                },
                "s3MonitoringConfiguration": {"logUri": f"s3://{self.s3_bucket}/emr_container/logs"},
            },
        }

    def get_retry_policy_configuration(self) -> dict:
        return {"maxAttempts": self.max_attempts}


def main() -> None:
    emr_container_job_config = EmrContainerJobConfig()

    job_name = emr_container_job_config.get_job_name()
    job_driver_params = emr_container_job_config.get_job_driver_params(aws_utils.Account.prod)
    configuration_overrides_params = emr_container_job_config.get_configuration_overrides_params(False)
    tags = emr_container_job_config.tags
    retry_policy_configuration = emr_container_job_config.get_retry_policy_configuration()

    logger.info(f"Job Name: {job_name}")
    logger.info(f"Job Driver: {json.dumps(job_driver_params, indent=4)}")
    logger.info(f"Configuration Overrides: {json.dumps(configuration_overrides_params, indent=4)}")
    logger.info(f"Tags: {json.dumps(tags, indent=4)}")
    logger.info(f"Retry Policy Configuration: {json.dumps(retry_policy_configuration, indent=4)}")

    emr_client = boto3.client("emr-containers", region_name=_AWS_REGION)
    response = emr_client.start_job_run(
        name=job_name[:63],
        virtualClusterId=emr_container_job_config.virtual_cluster_id,
        executionRoleArn=emr_container_job_config.execution_role_arn,
        releaseLabel=_RELEASE_LABEL,
        jobDriver=job_driver_params,
        configurationOverrides=configuration_overrides_params,
        tags=tags,
        retryPolicyConfiguration=retry_policy_configuration,
    )

    status_code = response["ResponseMetadata"]["HTTPStatusCode"]
    if status_code != http.HTTPStatus.OK:
        raise RuntimeError(f"Failed to start job: {json.dumps(response, indent=4)}")

    logger.info(f"Status Code: {status_code}")
    logger.info(f"Job Run ID: {response['id']}")
    logger.info(f"Response detail: {json.dumps(response, indent=4)}")


if __name__ == "__main__":
    main()
