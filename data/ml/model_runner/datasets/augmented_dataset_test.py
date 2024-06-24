import os
import random
import tempfile

from absl.testing import absltest
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.datasets import augmented_dataset
from data.ml.model_runner.datasets import parquet_utils
from data.ml.model_runner.utils import test_utils
from data.ml.utils import metadata


class AugmentedDatasetTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)

        self.meta = metadata.Meta.from_json({"col1": [str(i) for i in range(10)], "col2": [str(i) for i in range(10)]})

    def test_item_drop(self):
        seq = torch.tensor([62, 1868, 1309, 1167, 578, 3280, 2787, 1216, 3689, 1206])
        length = 10
        output_seq, output_len = augmented_dataset._item_crop(seq, length)  # pylint: disable=protected-access

        assert output_len == 6
        torch.testing.assert_close(output_seq, torch.tensor([1167, 578, 3280, 2787, 1216, 3689, 0, 0, 0, 0], dtype=torch.long))

        seq2 = torch.tensor([1, 17, 62, 141, 628, 0, 0, 0, 0, 0])
        length2 = 5
        output_seq2, output_len2 = augmented_dataset._item_crop(seq2, length2)  # pylint: disable=protected-access

        assert output_len2 == 3
        torch.testing.assert_close(output_seq2, torch.tensor([17, 62, 141, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long))

    def test_item_mask(self):
        seq = torch.tensor([62, 1868, 1309, 1167, 578, 3280, 2787, 1216, 3689, 1206])
        length = 10

        output_seq, output_len = augmented_dataset._item_mask(seq, length, 10154)  # pylint: disable=protected-access

        assert output_len == 10
        torch.testing.assert_close(
            output_seq, torch.tensor([10155, 1868, 1309, 1167, 578, 3280, 10155, 1216, 3689, 10155], dtype=torch.long)
        )

        seq2 = torch.tensor([1, 17, 62, 141, 628, 0, 0, 0, 0, 0])
        length2 = 5
        output_seq2, output_len2 = augmented_dataset._item_mask(seq2, length2, 10154)  # pylint: disable=protected-access

        assert output_len2 == 5
        torch.testing.assert_close(output_seq2, torch.tensor([1, 17, 10155, 141, 628, 0, 0, 0, 0, 0], dtype=torch.long))

    def test_item_reorder(self):
        seq = torch.tensor([62, 1868, 1309, 1167, 578, 3280, 2787, 1216, 3689, 1206])
        length = 10

        output_seq, output_len = augmented_dataset._item_reorder(seq, length)  # pylint: disable=protected-access

        assert output_len == 10
        torch.testing.assert_close(output_seq, torch.tensor([62, 1868, 1309, 1216, 578, 3689, 3280, 1167, 2787, 1206], dtype=torch.long))

        seq2 = torch.tensor([1, 17, 62, 141, 628, 0, 0, 0, 0, 0])
        length2 = 5

        output_seq2, output_len2 = augmented_dataset._item_reorder(seq2, length2)  # pylint: disable=protected-access

        assert output_len2 == 5
        torch.testing.assert_close(output_seq2, torch.tensor([1, 17, 141, 62, 628, 0, 0, 0, 0, 0], dtype=torch.long))

    def test_augment_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col1": [[4, 9, 5, 2, 1, 7, 6, 3, 1, 3]], "col2": [list(range(10))]},
                os.path.join(tmpdirname, "test1.snappy.parquet"),
            )

            dataset = augmented_dataset.AugmentedDataset(
                parquet_dir=tmpdirname,
                max_seq_len=10,
                target_col="col1",
                cols=["col1", "col2"],
                meta=self.meta,
                reserved_labels={"col1": 1, "col2": 1},
            )

            expected = [
                {
                    common.INPUTS_COL: {
                        "col1": torch.tensor([5, 10, 6, 3, 2, 8, 7, 4, 2, 0]),
                        "col2": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
                        "augmented_seq_0": torch.tensor([12, 10, 6, 3, 12, 8, 7, 4, 2, 0]),
                        "augmented_seq_len_0": torch.tensor(9),
                        "augmented_seq_1": torch.tensor([5, 10, 6, 3, 2, 7, 8, 2, 4, 0]),
                        "augmented_seq_len_1": torch.tensor(9),
                    },
                    common.TARGETS_COL: {"col1": torch.tensor(4), "col2": torch.tensor(10)},
                    common.SEQ_LEN_COL: torch.tensor(9),
                    common.ALL_TARGETS_COL: torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
                    common.NEXT_TARGET_COL: torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                }
            ]

            for i, datum in enumerate(dataset):
                test_utils.assert_dict_equals(datum, expected[i])


if __name__ == "__main__":
    absltest.main()
