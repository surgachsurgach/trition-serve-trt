import os

from absl.testing import absltest

from data.ml.model_runner.generators import item2user
from data.ml.model_runner.inference.utils import meta as meta_utils
from data.ml.model_runner.inference.utils import model as model_utils
from data.ml.model_runner.models import vae_cf

_TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data/vae_cf")


class ModelLoaderTest(absltest.TestCase):
    def setUp(self):
        self.model_path = _TEST_DATA_DIR
        self.meta_path = _TEST_DATA_DIR
        self.model_klass = vae_cf.VAE
        self.device = "cpu"
        meta_utils.MetaLazyFactory.clear()

    def test_load_model(self):
        model = model_utils.load_model(
            model_path=self.model_path,
            model_klass=self.model_klass,
            generator_klass=item2user.ExclusionGenerator,
            device=self.device,
            meta_path=self.meta_path,
        )
        self.assertIsNotNone(model)

    def test_is_singleton(self):
        model1 = model_utils.load_model(
            self.model_path,
            self.model_klass,
            item2user.ExclusionGenerator,
            self.device,
            self.meta_path,
        )
        model2 = model_utils.load_model(
            self.model_path,
            self.model_klass,
            item2user.ExclusionGenerator,
            self.device,
            self.meta_path,
        )
        self.assertIs(model1, model2)
