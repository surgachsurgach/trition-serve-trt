import os
import tempfile

from absl.testing import absltest

from data.ml.model_runner.configs import config_parser


class ConfigParserTest(absltest.TestCase):
    def test_config_parser(self):
        expected = config_parser.TrainConfig(
            accelerator="cpu",
            batch_size=32,
            checkpoints=[
                config_parser.CheckpointConfig(monitor="train_loss/G_content", dirpath="test/path", mode="min", every_n_epochs=1),
                config_parser.CheckpointConfig(dirpath="test/path", every_n_epochs=1),
            ],
            devices=2,
            log_dir="log/path",
            max_epoch=1,
        )
        actual = config_parser.TrainConfig.from_dict(
            {
                "accelerator": "cpu",
                "batch_size": 32,
                "checkpoints": [
                    {
                        "monitor": "train_loss/G_content",
                        "dirpath": "test/path",
                        "mode": "min",
                        "every_n_epochs": 1,
                    },
                    {
                        "dirpath": "test/path",
                        "every_n_epochs": 1,
                    },
                ],
                "devices": 2,
                "log_dir": "log/path",
                "max_epoch": 1,
            }
        )
        assert expected == actual

        expected = config_parser.PredictConfig(
            accelerator="gpu",
            batch_size=256,
            devices=2,
            checkpoint_dir="test/path",
            output_path="test/outpath",
        )
        actual = config_parser.PredictConfig.from_dict(
            {"accelerator": "gpu", "batch_size": 256, "devices": 2, "checkpoint_dir": "test/path", "output_path": "test/outpath"}
        )
        assert expected == actual

        expected = config_parser.RecsysPredictConfig(accelerator="gpu", batch_size=256, devices=2, checkpoint_dir="test/path")
        actual = config_parser.RecsysPredictConfig.from_dict(
            {"accelerator": "gpu", "batch_size": 256, "devices": 2, "checkpoint_dir": "test/path"}
        )
        assert expected == actual

    def test_default(self):
        expected = config_parser.TrainConfig(
            accelerator="cpu",
            batch_size=32,
            checkpoints=None,
            max_epoch=50,
            devices="auto",
            log_dir=None,
        )

        actual = config_parser.TrainConfig.from_dict({})

        assert expected == actual

    def test_get_best_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i in range(5):
                with open(os.path.join(tmpdirname, f"epoch={i + 1}.ckpt"), "wb"):
                    pass

            with open(os.path.join(tmpdirname, "best_model.ckpt"), "wb"):
                pass

            expected = os.path.join(tmpdirname, "best_model.ckpt")
            actual = config_parser.RecsysPredictConfig(checkpoint_dir=tmpdirname).get_best_checkpoint_path()

            assert expected == actual

    def test_get_best_checkpoint_path_best_not_exists(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(os.path.join(tmpdirname, "last.ckpt"), "wb"):
                pass

            expected = os.path.join(tmpdirname, "last.ckpt")
            actual = config_parser.RecsysPredictConfig(checkpoint_dir=tmpdirname).get_best_checkpoint_path()

            assert expected == actual

    def test_get_best_checkpoint_path_ckpt_not_exists(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(RuntimeError):
                config_parser.RecsysPredictConfig(checkpoint_dir=tmpdirname).get_best_checkpoint_path()


if __name__ == "__main__":
    absltest.main()
