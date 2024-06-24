import abc
import dataclasses
import os
from typing import Any, Type

import boto3
from sagemaker import session as sagemaker_session
from sagemaker.experiments import run as sagemaker_run


class Run:
    """To be used for compatibility when not running in SageMaker."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_metric(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def log_artifacts(self, *args, **kwargs):
        pass

    def log_parameters(self, *args, **kwargs):
        pass


def is_in_sagemaker_env() -> bool:
    return os.environ.get("SM_HOSTS") is not None


def init_run(region="us-east-1") -> sagemaker_run.Run | Run:
    exp_name = os.environ.get("EXP_NAME")
    if exp_name:
        session = sagemaker_session.Session(boto3.session.Session(region_name=region))
        # https://sagemaker.readthedocs.io/en/v2.125.0/experiments/sagemaker.experiments.html#run
        return sagemaker_run.load_run(sagemaker_session=session)
    return Run()


def guard_none(instance: dict[str, any]):
    """Member must satisfy regular expression pattern: .* in sagemaker"""

    copy_dict = instance.copy()

    for key, value in instance.items():
        if value is None:
            copy_dict[key] = "None"
        if isinstance(value, dict):
            copy_dict[key] = guard_none(value)
        if isinstance(value, list):
            copy_list = value.copy()
            for i, v in enumerate(value):
                if isinstance(v, dict):
                    copy_list[i] = guard_none(v)
                if v is None:
                    copy_list[i] = "None"
            copy_dict[key] = copy_list
    return copy_dict


def dataclass_to_dict(instance: dataclasses.dataclass):
    # Convert the instance to a dictionary
    instance_dict = dataclasses.asdict(instance)

    # Recursively convert nested data classes to dictionaries
    for key, value in instance_dict.items():
        if dataclasses.is_dataclass(value):
            instance_dict[key] = dataclass_to_dict(value)
    return guard_none(instance_dict)


class MetricLoggingMixin(abc.ABC):
    @abc.abstractmethod
    def _collect_metric_on_train_step(self, metrics: dict[str, Any]) -> None:
        """Collect metrics on train step.
        Must be called in the training_step method.
        """

    @abc.abstractmethod
    def _log_metric_on_train_epoch_end(self) -> None:
        """Log metrics on train epoch end."""

    @abc.abstractmethod
    def _collect_metric_on_validation_step(self, metrics: dict[str, Any]) -> None:
        """Collect metrics on validation step.
        Must be called in the validation_step method.
        """

    @abc.abstractmethod
    def _log_metric_on_validation_epoch_end(self) -> None:
        """Log metrics on validation epoch end."""


RunningEnv = Type[sagemaker_run.Run | Run]

RESERVED_CHECKPOINT = "best_model.pth"


def get_sagemaker_checkpoint():
    return f"{os.environ.get('SM_MODEL_DIR')}/{RESERVED_CHECKPOINT}"


def get_sagemaker_tensorboard_path():
    return os.getenv("TENSORBOARD_LOG_DIR")
