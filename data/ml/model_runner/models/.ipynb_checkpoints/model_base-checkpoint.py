from typing import Any

from data.pylib.constant import recsys as common
from data.ml.model_runner.base import model_base
from data.ml.model_runner.generators import base as base_generator
from data.ml.model_runner.utils import sagemaker_utils
from data.ml.utils import metadata


class SageMakerModelLoggingMixin(model_base.ModelBase, sagemaker_utils.MetricLoggingMixin):
    def _collect_metric_on_train_step(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if key not in self.training_step_outputs:
                self.training_step_outputs[key] = value
            else:
                self.training_step_outputs[key] += value

    def _collect_metric_on_validation_step(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if key not in self.validation_step_outputs:
                self.validation_step_outputs[key] = value
            else:
                self.validation_step_outputs[key] += value

    def _log_metric_on_train_epoch_end(self) -> None:
        for key, value in self.training_step_outputs.items():
            epoch_mean = value / self.train_steps_per_epoch
            self.run.log_metric(key, epoch_mean, step=self.current_epoch)
        self.training_step_outputs.clear()

    def _log_metric_on_validation_epoch_end(self, **kwargs) -> None:
        for key, value in self.validation_step_outputs.items():
            epoch_mean = value / self.validation_steps_per_epoch
            self.run.log_metric(key, epoch_mean, step=self.current_epoch)
        self.validation_step_outputs.clear()


class RecsysModelBase(model_base.ModelBase):
    def __init__(
        self,
        meta: metadata.Meta,
        generator: base_generator.Generator | None = None,
        predict_top_k: int | None = None,  # Deprecated
        exclude_inputs_from_predictions: bool = True,  # Deprecated
        **kwargs,
    ):
        """Model base class.

        Args:
            meta: metadata of the model.
        """
        super().__init__(**kwargs)

        self._meta = meta
        self._generator = generator
        self._predict_top_k = min(self._item_size, predict_top_k) if predict_top_k else None
        self.exclude_inputs_from_predictions = exclude_inputs_from_predictions
        self.save_hyperparameters(ignore="meta")
        self._print_hparams()

    @property
    def _item_size(self) -> int:
        return self._meta.get_meta_size(common.ITEM_ID_COL)

    @property
    def _user_size(self):
        return self._meta.get_meta_size(common.USER_ID_COL)


class RecsysSageMakerModelBase(RecsysModelBase, SageMakerModelLoggingMixin):
    def on_train_epoch_end(self):
        self._log_metric_on_train_epoch_end()

    def on_validation_epoch_end(self):
        self._log_metric_on_validation_epoch_end()
