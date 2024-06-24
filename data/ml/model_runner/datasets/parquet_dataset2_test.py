import os
import tempfile

from absl.testing import absltest
import numpy as np
import torch
from torch.utils import data as td

from data.pylib.constant import recsys as common
from data.ml.model_runner.datasets import data_transformer
from data.ml.model_runner.datasets import parquet_dataset2
from data.ml.model_runner.datasets import parquet_utils
from data.ml.model_runner.utils import test_utils
from data.ml.utils import metadata


class _TestDataTransformer(data_transformer.DataTransformer):
    def transform_features(self, features, dataset_type=None):
        feature_dict = {}
        for col in self._cols:
            feature_dict[col] = features[col]
        return feature_dict


class ParquetDataset2Test(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_parquet_dataset2(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]},
                os.path.join(tmpdirname, "test.snappy.parquet"),
            )

            dataset = parquet_dataset2.ParquetDataset2(
                parquet_dir=tmpdirname,
                transformer=_TestDataTransformer(
                    meta=metadata.Meta.from_json({common.ITEM_ID_COL: list(map(str, range(10)))}),
                    cols=["col1", "col2"],
                ),
            )

            expected = [
                {"col1": torch.tensor([1, 2]), "col2": torch.tensor([5, 6])},
                {"col1": torch.tensor([3, 4]), "col2": torch.tensor([7, 8])},
            ]

            data_loader = td.DataLoader(dataset, batch_size=2)
            for batch_idx, batch_data in enumerate(data_loader):
                test_utils.assert_dict_equals(batch_data, expected[batch_idx])

    def test_parquet_iterable_dataset2(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]},
                os.path.join(tmpdirname, "test.snappy.parquet"),
            )

            dataset = parquet_dataset2.ParquetIterableDataset2(
                parquet_dir=tmpdirname,
                shuffle=False,
                transformer=_TestDataTransformer(
                    meta=metadata.Meta.from_json({common.ITEM_ID_COL: list(map(str, range(10)))}),
                    cols=["col1", "col2"],
                ),
            )

            expected = [
                {"col1": torch.tensor([1, 2]), "col2": torch.tensor([5, 6])},
                {"col1": torch.tensor([3, 4]), "col2": torch.tensor([7, 8])},
            ]

            data_loader = td.DataLoader(dataset, batch_size=2)
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
                {"col1": torch.tensor([2, 1]), "col2": torch.tensor([6, 5])},
                {"col1": torch.tensor([4, 3]), "col2": torch.tensor([8, 7])},
            ]

            dataset = parquet_dataset2.ParquetIterableDataset2(
                parquet_dir=tmpdirname,
                transformer=_TestDataTransformer(
                    meta=metadata.Meta.from_json({common.ITEM_ID_COL: list(map(str, range(10)))}),
                    cols=["col1", "col2"],
                ),
                shuffle_buffer_size=10,
                shuffle=True,
                seed=0,
            )

            data_loader = td.DataLoader(dataset, batch_size=2, num_workers=2)
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

            dataset = parquet_dataset2.ParquetIterableDataset2(
                parquet_dir=tmpdirname,
                transformer=_TestDataTransformer(
                    meta=metadata.Meta.from_json({common.ITEM_ID_COL: list(map(str, range(10)))}),
                    cols=["col1", "col2"],
                ),
                shuffle_buffer_size=10,
                shuffle=True,
                seed=0,
            )

            for i, datum in enumerate(dataset):
                test_utils.assert_dict_equals(datum, expected[i])


if __name__ == "__main__":
    absltest.main()
