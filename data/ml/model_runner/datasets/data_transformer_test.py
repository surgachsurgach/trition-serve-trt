import random

from absl.testing import absltest
import numpy as np
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.datasets import data_transformer
from data.ml.model_runner.datasets import dataset_type as datasets
from data.ml.model_runner.datasets import sampling_utils
from data.ml.model_runner.utils import test_utils
from data.ml.utils import metadata


class DataTransformerTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    def test_sparse_data_transformer(self):
        data = {"col1": [1, 2, 3, 4], "col2": [5]}

        transformer = data_transformer.SparseDataTransformer(
            meta=metadata.Meta.from_json({common.ITEM_ID_COL: list(map(str, range(10)))}),
            cols=["col1", "col2"],
        )

        actual = transformer.transform_features(data)
        expected = {
            "col1": torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "col2": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        }

        test_utils.assert_dict_equals(actual, expected)

    def test_sparse_weighted_data_transformer(self):
        transformer = data_transformer.SparseWeightedDataTransformer(
            meta=metadata.Meta.from_json({common.ITEM_ID_COL: list(map(str, range(10)))}),
            cols=["col1", "weight_of_col1"],
            sparse_cols=["col1"],
            weight_cols=["weight_of_col1"],
            meta_keys=[common.ITEM_ID_COL],
        )
        data = {"col1": [1, 2, 3, 4], "weight_of_col1": [4, 3, 2, 1]}

        actual = transformer.transform_features(data)
        expected = {"col1": torch.tensor([0.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        test_utils.assert_dict_equals(actual, expected)

    def test_sampling_data_transformer(self):
        sampler = sampling_utils.NegativeSampler(
            meta=metadata.Meta.from_json(
                {
                    common.ITEM_ID_COL: list(map(str, range(5))),
                    common.ITEM_FREQUENCY_META_KEY: [1, 2, 3, 1, 2],
                }
            ),
            num_negative_samples=2,
        )
        transformer = data_transformer.SamplingDataTransformer(
            meta=metadata.Meta.from_json(
                {
                    common.ITEM_ID_COL: list(map(str, range(5))),
                    common.ITEM_FREQUENCY_META_KEY: [1, 2, 3, 1, 2],
                }
            ),
            cols=[
                common.INPUTS_COL,
                common.TARGETS_COL,
                common.POSITIVE_CONTEXTS_COL,
            ],
            sampler=sampler,
        )

        data = {
            common.INPUTS_COL: [2, 0],
            common.TARGETS_COL: [2, 0],
            common.POSITIVE_CONTEXTS_COL: [[0], [2]],
        }
        actual = transformer.transform_features(data)
        expected = {
            common.TARGETS_COL: torch.tensor([2, 0]),
            common.POSITIVE_CONTEXTS_COL: torch.tensor([[0], [2]]),
            common.NEGATIVE_CONTEXTS_COL: torch.tensor([1, 3, 4]),
        }

        test_utils.assert_dict_equals(actual, expected)
        actual = transformer.postprocess(actual)
        expected = {
            common.TARGETS_COL: torch.tensor([2, 0]),
            common.POSITIVE_CONTEXTS_COL: torch.tensor([[0], [2]]),
            common.NEGATIVE_CONTEXTS_COL: torch.tensor([[3, 4], [3, 3]]),
        }
        test_utils.assert_dict_equals(actual, expected)

    def test_sequential_data_transformer(self):
        transformer = data_transformer.SequentialDataTransformer(
            meta=metadata.Meta.from_json({"item_id": list(map(str, range(10)))}),
            cols=[common.USER_ID_COL, common.INPUTS_COL, common.TARGETS_COL],
            input_col=common.INPUTS_COL,
            target_col=common.TARGETS_COL,
            meta_key=common.ITEM_ID_COL,
            max_seq_len=10,
        )

        data = {
            common.USER_ID_COL: 1,
            common.INPUTS_COL: [1, 2, 3, 4],
            common.TARGETS_COL: [5],
        }
        actual = transformer.transform_features(data)
        expected = {
            common.INPUTS_COL: torch.tensor([1, 2, 3, 10, 10, 10, 10, 10, 10, 10]),
            common.TARGETS_COL: torch.tensor(4),
            common.TARGET_IDX_COL: torch.tensor(3),
            common.USER_ID_COL: 1,
        }
        test_utils.assert_dict_equals(actual, expected)

        actual = transformer.transform_features(data, dataset_type=datasets.DatasetType.DEV_DATASET)
        expected = {
            common.INPUTS_COL: torch.tensor([1, 2, 3, 10, 10, 10, 10, 10, 10, 10]),
            common.TARGETS_COL: torch.tensor(4),
            common.TARGET_IDX_COL: torch.tensor(3),
            common.NEXT_TARGET_COL: torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            common.ALL_TARGETS_COL: torch.tensor([0, 0, 0, 0, 1, 0.99, 0, 0, 0, 0]),
            common.USER_ID_COL: 1,
        }
        test_utils.assert_dict_equals(actual, expected)

        actual = transformer.transform_features(data, dataset_type=datasets.DatasetType.PREDICT_DATASET)
        expected = {
            common.INPUTS_COL: torch.tensor([1, 2, 3, 4, 10, 10, 10, 10, 10, 10]),
            common.TARGET_IDX_COL: torch.tensor(4),
            common.USER_ID_COL: 1,
        }
        test_utils.assert_dict_equals(actual, expected)

    def test_masked_sequential_data_transformer(self):
        transformer = data_transformer.MaskedSequentialDataTransformer(
            meta=metadata.Meta.from_json({common.ITEM_ID_COL: list(map(str, range(10)))}),
            cols=[common.INPUTS_COL, common.TARGETS_COL],
            input_col=common.INPUTS_COL,
            target_col=common.TARGETS_COL,
            meta_key=common.ITEM_ID_COL,
            max_seq_len=5,
            overlength_handing_method="latest",
            mask_prob=0.3,
        )

        data = {
            common.USER_ID_COL: 1,
            common.INPUTS_COL: [0, 4, 3, 1, 2],
            common.TARGETS_COL: [5],
        }

        actual = transformer.transform_features(data)
        actual = transformer.postprocess(actual)
        expected = {
            common.INPUTS_COL: torch.tensor([0, 4, 11, 1, 2]),
            common.TARGETS_COL: torch.tensor([-100, -100, 3, -100, -100]),
        }
        test_utils.assert_dict_equals(actual, expected)

        actual = transformer.transform_features(data, dataset_type=datasets.DatasetType.DEV_DATASET)
        actual = transformer.postprocess(actual, dataset_type=datasets.DatasetType.DEV_DATASET)
        expected = {
            common.INPUTS_COL: torch.tensor([0, 4, 3, 1, 11]),
            common.TARGETS_COL: torch.tensor([-100, -100, -100, -100, 2]),
            common.TARGET_IDX_COL: torch.tensor(4),
            common.NEXT_TARGET_COL: torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            common.ALL_TARGETS_COL: torch.tensor([0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.9900, 0.0000, 0.0000, 0.0000, 0.0000]),
        }
        test_utils.assert_dict_equals(actual, expected)

        actual = transformer.transform_features(data, dataset_type=datasets.DatasetType.PREDICT_DATASET)
        actual = transformer.postprocess(actual, dataset_type=datasets.DatasetType.PREDICT_DATASET)
        expected = {
            common.INPUTS_COL: torch.tensor([4, 3, 1, 2, 11]),
            common.TARGET_IDX_COL: torch.tensor(4),
            common.EXCLUSION_COL: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        test_utils.assert_dict_equals(actual, expected)


if __name__ == "__main__":
    absltest.main()
