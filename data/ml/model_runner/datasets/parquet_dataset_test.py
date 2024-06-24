# pylint: disable=protected-access
import os
import random
import tempfile

from absl.testing import absltest
import numpy as np
import torch
from torch.utils import data

from data.pylib.constant import recsys as common
from data.ml.model_runner.datasets import parquet_dataset
from data.ml.model_runner.datasets import parquet_utils
from data.ml.model_runner.utils import test_utils
from data.ml.utils import metadata


class _TestParquetDataSet(parquet_dataset.ParquetIterableDataSet):
    def _transform_features(self, features):
        feature_dict = {}
        for col in self._cols:
            feature_dict[col] = features[col]

        return feature_dict


class ParquetDatasetTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_split_by_rank_and_worker(self):
        inputs = list(range(20))

        num_gpus = 2
        num_workers = 4

        expected = [
            [  # gpu 0
                [0, 8, 16],
                [2, 10, 18],
                [4, 12],
                [6, 14],
            ],
            [  # gpu 1
                [1, 9, 17],
                [3, 11, 19],
                [5, 13],
                [7, 15],
            ],
        ]

        for i in range(num_gpus):
            for j in range(num_workers):
                result = list(parquet_dataset._split_by_gpu_and_worker(inputs, i, num_gpus, j, num_workers))
                self.assertEqual(result, expected[i][j])

        num_gpus = 1
        num_workers = 8

        expected2 = [
            [
                [0, 8, 16],
                [1, 9, 17],
                [2, 10, 18],
                [3, 11, 19],
                [4, 12],
                [5, 13],
                [6, 14],
                [7, 15],
            ]
        ]

        for i in range(num_gpus):
            for j in range(num_workers):
                result = list(parquet_dataset._split_by_gpu_and_worker(inputs, i, num_gpus, j, num_workers))
                self.assertEqual(result, expected2[i][j])

    def test_invalid_parquet_dir(self):
        with self.assertRaises(AssertionError):
            _TestParquetDataSet(parquet_dir="")

        with self.assertRaises(AssertionError):
            _TestParquetDataSet(parquet_dir=None)

        with self.assertRaises(AssertionError):
            _TestParquetDataSet(parquet_dir=[])

    def test_parquet_dataset_loader(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]},
                os.path.join(tmpdirname, "test.snappy.parquet"),
            )

            expected = [
                {"col1": torch.tensor([1, 2]), "col2": torch.tensor([5, 6])},
                {"col1": torch.tensor([3, 4]), "col2": torch.tensor([7, 8])},
            ]

            dataset = _TestParquetDataSet(
                parquet_dir=tmpdirname,
                cols=["col1", "col2"],
                shuffle_buffer_size=1,
                shuffle=False,
            )
            data_loader = data.DataLoader(dataset, batch_size=2)
            for batch_idx, batch_data in enumerate(data_loader):
                test_utils.assert_dict_equals(batch_data, expected[batch_idx])

    def test_parquet_dataset_loader_multiple_worker(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col1": [1, 2], "col2": [5, 6]},
                os.path.join(tmpdirname, "test1.snappy.parquet"),
            )
            parquet_utils.make_parquet_file(
                {"col1": [3, 4], "col2": [7, 8]},
                os.path.join(tmpdirname, "test2.snappy.parquet"),
            )

            expected = [
                {"col1": torch.tensor([1, 2]), "col2": torch.tensor([5, 6])},
                {"col1": torch.tensor([3, 4]), "col2": torch.tensor([7, 8])},
            ]

            dataset = _TestParquetDataSet(
                parquet_dir=tmpdirname,
                cols=["col1", "col2"],
                shuffle_buffer_size=1,
                shuffle=False,
            )
            data_loader = data.DataLoader(dataset, batch_size=2, num_workers=2)
            for batch_idx, batch_data in enumerate(data_loader):
                test_utils.assert_dict_equals(batch_data, expected[batch_idx])

    def test_parquet_dataset_shuffle(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]},
                os.path.join(tmpdirname, "test.snappy.parquet"),
            )

            expected = [
                {"col1": torch.tensor(4), "col2": torch.tensor(8)},
                {"col1": torch.tensor(2), "col2": torch.tensor(6)},
                {"col1": torch.tensor(3), "col2": torch.tensor(7)},
                {"col1": torch.tensor(1), "col2": torch.tensor(5)},
            ]

            dataset = _TestParquetDataSet(
                parquet_dir=tmpdirname,
                cols=["col1", "col2"],
                shuffle_buffer_size=10,
                shuffle=True,
                seed=0,
            )

            for i, datum in enumerate(dataset):
                test_utils.assert_dict_equals(datum, expected[i])

    def test_parquet_sparse_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col1": [1, 2], "col2": [3, 4]},
                os.path.join(tmpdirname, "test1.snappy.parquet"),
            )
            parquet_utils.make_parquet_file(
                {"col1": [5, 6], "col2": [7, 8]},
                os.path.join(tmpdirname, "test2.snappy.parquet"),
            )

            dataset = parquet_dataset.ParquetIterableSparseDataSet(
                meta=metadata.Meta.from_json({common.ITEM_ID_COL: list(map(str, range(10)))}),
                ignore=["col1"],
                parquet_dir=tmpdirname,
                cols=["col1", "col2"],
                shuffle_buffer_size=1,
                shuffle=False,
            )

            expected = [
                {
                    "col1": 1,
                    "col2": torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                },
                {
                    "col1": 2,
                    "col2": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                },
                {
                    "col1": 5,
                    "col2": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                },
                {
                    "col1": 6,
                    "col2": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                },
            ]

            for i, datum in enumerate(dataset):
                test_utils.assert_dict_equals(datum, expected[i])

    def test_parquet_identity_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col1": [1, 2], "col2": [3, 4]},
                os.path.join(tmpdirname, "test1.snappy.parquet"),
            )
            parquet_utils.make_parquet_file(
                {"col1": [5, 6], "col2": [7, 8]},
                os.path.join(tmpdirname, "test2.snappy.parquet"),
            )

            dataset = parquet_dataset.ParquetIterableIdentityDataSet(
                parquet_dir=tmpdirname,
                cols=["col1", "col2"],
                shuffle_buffer_size=1,
                shuffle=False,
            )

            expected = [
                {"col1": 1, "col2": 3},
                {"col1": 2, "col2": 4},
                {"col1": 5, "col2": 7},
                {"col1": 6, "col2": 8},
            ]

            for i, datum in enumerate(dataset):
                test_utils.assert_dict_equals(datum, expected[i])

    def test_parquet_sequential_dataset(self):
        random.seed(0)
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col0": "a", "col1": [[1, 2]], "col2": [[3, 4]]},
                os.path.join(tmpdirname, "test1.snappy.parquet"),
            )
            parquet_utils.make_parquet_file(
                {"col0": "b", "col1": [[5, 6, 7, 8, 9, 10]], "col2": [[1, 2, 3, 4, 5, 6]]},
                os.path.join(tmpdirname, "test2.snappy.parquet"),
            )
            parquet_utils.make_parquet_file(
                {"col0": "c", "col1": [list(range(0, 100))], "col2": [list(range(100, 0, -1))]},
                os.path.join(tmpdirname, "test3.snappy.parquet"),
            )

            dataset = parquet_dataset.ParquetIterableSequentialDataSet(
                max_seq_len=5,
                cols=["col0", "col1", "col2"],
                reserved_labels={"col1": 1},
                ignore=["col0"],
                parquet_dir=tmpdirname,
                shuffle_buffer_size=1,
                shuffle=False,
            )

            expected = [
                {
                    common.INPUTS_COL: {"col1": torch.tensor([2, 0, 0, 0, 0]), "col2": torch.tensor([3, 0, 0, 0, 0])},
                    common.TARGETS_COL: {"col1": torch.tensor(3), "col2": torch.tensor(4)},
                    common.SEQ_LEN_COL: torch.tensor(1),
                    "col0": "a",
                },
                {
                    common.INPUTS_COL: {"col1": torch.tensor([6, 7, 8, 9, 10]), "col2": torch.tensor([1, 2, 3, 4, 5])},
                    common.TARGETS_COL: {"col1": torch.tensor(11), "col2": torch.tensor(6)},
                    common.SEQ_LEN_COL: torch.tensor(5),
                    "col0": "b",
                },
                {
                    common.INPUTS_COL: {"col1": torch.tensor([54, 55, 56, 57, 58]), "col2": torch.tensor([47, 46, 45, 44, 43])},
                    common.TARGETS_COL: {"col1": torch.tensor(59), "col2": torch.tensor(42)},
                    common.SEQ_LEN_COL: torch.tensor(5),
                    "col0": "c",
                },
            ]

            for i, datum in enumerate(dataset):
                test_utils.assert_dict_equals(datum, expected[i])

    def test_parquet_sequential_dev_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col0": "a", "col1": [[1, 2]], "col2": [[3, 4]]},
                os.path.join(tmpdirname, "test1.snappy.parquet"),
            )

            dataset = parquet_dataset.ParquetIterableSequentialDataSet(
                target_col="col2",
                max_seq_len=5,
                reserved_labels={"col1": 1, "col2": 0},
                ignore=["col0"],
                parquet_dir=tmpdirname,
                cols=["col0", "col1", "col2"],
                shuffle_buffer_size=1,
                shuffle=False,
                meta=(metadata.Meta.from_json({"col2": list(range(5))})),
            )

            expected = [
                {
                    common.INPUTS_COL: {"col1": torch.tensor([2, 0, 0, 0, 0]), "col2": torch.tensor([3, 0, 0, 0, 0])},
                    common.TARGETS_COL: {"col1": torch.tensor(3), "col2": torch.tensor(4)},
                    common.SEQ_LEN_COL: torch.tensor(1),
                    "col0": "a",
                    common.ALL_TARGETS_COL: torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
                    common.NEXT_TARGET_COL: torch.tensor([0, 0, 0, 0, 1]),
                },
            ]

            for i, datum in enumerate(dataset):
                test_utils.assert_dict_equals(datum, expected[i])

            assert dataset._item_size == 5

            dataset2 = parquet_dataset.ParquetIterableSequentialDataSet(
                target_col="col2",
                max_seq_len=5,
                reserved_labels={"col1": 1, "col2": 1},
                ignore=["col0"],
                parquet_dir=tmpdirname,
                cols=["col0", "col1", "col2"],
                shuffle_buffer_size=1,
                shuffle=False,
                meta=(metadata.Meta.from_json({"col2": list(range(5))})),
            )

            assert dataset2._item_size == 6

    def test_parquet_sequential_group_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col0": "a", common.ITEM_ID_COL: [[0, 1]], "item_genre": [[0, 0]]},
                os.path.join(tmpdirname, "test1.snappy.parquet"),
            )

            dataset = parquet_dataset.ParquetIterableSequentialGroupedDataSet(
                target_col=common.ITEM_ID_COL,
                max_seq_len=5,
                reserved_labels={},
                ignore=["col0"],
                parquet_dir=tmpdirname,
                cols=["col0", common.ITEM_ID_COL, "item_genre"],
                shuffle_buffer_size=1,
                shuffle=False,
                meta=(metadata.Meta.from_json({common.ITEM_ID_COL: ["a/1", "a/2"], "item_genre": ["a", "b"]})),
                target_genre="a",
            )

            expected = [
                {
                    common.INPUTS_COL: {common.ITEM_ID_COL: torch.tensor([0, 0, 0, 0, 0]), "item_genre": torch.tensor([0, 0, 0, 0, 0])},
                    common.TARGETS_COL: {common.ITEM_ID_COL: torch.tensor(1), "item_genre": torch.tensor(0)},
                    common.SEQ_LEN_COL: torch.tensor(1),
                    "col0": "a",
                    common.NEXT_TARGET_COL: torch.tensor([0, 1]),
                },
            ]

            for i, datum in enumerate(dataset):
                test_utils.assert_dict_equals(datum, expected[i])

    def test_parquet_sequential_masked_dataset(self):
        random.seed(0)
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col0": "b", "col1": [[5, 6, 7, 8, 9, 10]]},
                os.path.join(tmpdirname, "test2.snappy.parquet"),
            )

            dataset = parquet_dataset.ParquetIterableMaskedDataset(
                meta=metadata.Meta.from_json({"col1": list(map(str, range(11)))}),
                max_seq_len=10,
                cols=["col0", "col1"],
                reserved_labels={"col1": 1},
                ignore=["col0"],
                parquet_dir=tmpdirname,
                target_col="col1",
                shuffle_buffer_size=1,
                shuffle=False,
            )

            expected = [
                {
                    common.INPUTS_COL: {"col1": torch.tensor([6, 7, 8, 9, 10, 12, 0, 0, 0, 0])},
                    common.TARGETS_COL: torch.tensor([-100, -100, -100, -100, -100, 11, -100, -100, -100, -100]),
                    common.SEQ_LEN_COL: torch.tensor(5),
                    "col0": "b",
                },
            ]

            for i, datum in enumerate(dataset):
                test_utils.assert_dict_equals(datum, expected[i])


if __name__ == "__main__":
    absltest.main()
