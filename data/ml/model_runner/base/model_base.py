from lightning import pytorch as pl
from lightning.pytorch.core import mixins
from loguru import logger

from data.ml.model_runner.utils import sagemaker_utils as run_utils


class ModelBase(pl.LightningModule, mixins.HyperparametersMixin):
    def __init__(
        self,
        total_training_steps: int | None = None,
        lr: float | list[float] = 1e-4,
        weight_decay: float = 0.0,
        auto_optimization: bool = True,
        run: run_utils.RunningEnv = run_utils.Run(),
        **kwargs,
    ):
        """Model base class.

        Args:
            lr: Learning rate. (list for mutliple optimizations)
        Attributes:
            automatic_optimization: False if set mulitple optimizers like GAN.
        """
        super().__init__(**kwargs)

        self._lr = lr
        self._weight_decay = weight_decay
        self.automatic_optimization = auto_optimization  # LightningModule property
        self._run = run
        self._total_training_steps = total_training_steps
        self._train_steps_per_epoch = None
        self._validation_steps_per_epoch = None
        self.training_step_outputs = {}  # must be cleared at each epoch
        self.validation_step_outputs = {}  # must be cleared at each epoch

    def get_name(self):
        return self.__class__.__name__

    def _print_hparams(self):
        logger.info(f"{self.__class__.__name__} configurations:")

        for k, v in self.hparams.items():
            logger.info(f"\t{k}: {v}")

    @property
    def run(self):
        return self._run

    @run.setter
    def run(self, run: run_utils.RunningEnv):
        self._run = run

    @property
    def total_training_steps(self):
        return self._total_training_steps

    @total_training_steps.setter
    def total_training_steps(self, value: int):
        self._total_training_steps = value

    @property
    def train_steps_per_epoch(self):
        return self._train_steps_per_epoch

    @train_steps_per_epoch.setter
    def train_steps_per_epoch(self, value: int):
        self._train_steps_per_epoch = value

    @property
    def validation_steps_per_epoch(self):
        return self._validation_steps_per_epoch

    @validation_steps_per_epoch.setter
    def validation_steps_per_epoch(self, value: int):
        self._validation_steps_per_epoch = value
