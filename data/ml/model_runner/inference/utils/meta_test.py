import os
import tempfile

from absl.testing import absltest

from data.ml.model_runner.inference.utils import meta as meta_utils
from data.ml.utils import metadata


class MetaTest(absltest.TestCase):
    _TEST_MODEL = "test_model"

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()  # pylint:disable=consider-using-with
        os.environ["META_PATH"] = cls.tmpdir.name

        meta_utils.MetaLazyFactory.clear()
        meta = metadata.Meta()
        meta.add_meta("item_id", ["a", "b", "c", "d", "e"])
        meta.save(cls.tmpdir.name)

    def test_get_item_size(self):
        assert meta_utils.get_item_size(self._TEST_MODEL) == 5

    def test_convert_id_to_idx(self):
        assert meta_utils.convert_id_to_idx(self._TEST_MODEL, ["a", "b", "c"]) == [0, 1, 2]

    def test_convert_idx_to_id(self):
        assert meta_utils.convert_idx_to_id(self._TEST_MODEL, [0, 1, 2]) == ["a", "b", "c"]

    def test_is_singleton(self):
        meta1 = meta_utils.MetaLazyFactory.get_instance(self._TEST_MODEL)
        meta2 = meta_utils.MetaLazyFactory.get_instance(self._TEST_MODEL)
        assert meta1 is meta2

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()
