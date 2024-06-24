import os

from lightning.pytorch import loggers
from lightning.pytorch.core import saving
from lightning.pytorch.utilities import rank_zero


class FastTensorboardLogger(loggers.TensorBoardLogger):
    """
    lightning.fabric.loggers.TensorboardLogger flushes logs everytime when save() is called.
    This generates huge delay in training/validation steps.

    Tensorboard SummaryWriter has its own flush options which are `max_queue` and `flush_secs`.
    Overriding save() function to not flush on every save() and leave flush ops to SummaryWriter.
    """

    @rank_zero.rank_zero_only
    def save(self) -> None:
        dir_path = self.log_dir

        # prepare the file path
        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist and the log directory exists
        if self._fs.isdir(dir_path) and not self._fs.isfile(hparams_file):
            saving.save_hparams_to_yaml(hparams_file, self.hparams)
