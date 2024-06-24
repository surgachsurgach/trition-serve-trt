import os
from typing import Type

from loguru import logger

from data.ml.model_runner.base import model_base
from data.ml.model_runner.generators import item2user
from data.ml.model_runner.inference.utils import meta as meta_utils


class _ModelLazyFactory:
    _instance = {}

    @classmethod
    def get_instance(
        cls,
        model_path: str,
        model_klass: Type[model_base.ModelBase],
        generator_klass: Type[item2user.Item2UserGenerator],
        device: str,
        meta_path: str | None = None,
    ):
        model_name = model_klass.__name__
        logger.info(f"Loading model from {model_path}, model class: {model_name}")

        if cls._instance.get(model_name) is None:
            path = os.path.join(model_path, "best_model.pth")
            meta = meta_utils.MetaLazyFactory.get_instance(model_name, meta_path)

            cls._instance[model_name] = model_klass.load_from_checkpoint(
                path,
                meta=meta,
                generator=generator_klass(meta=meta),
                map_location=device,
            )
        return cls._instance[model_name]

    @classmethod
    def clear(cls):
        cls._instance.clear()


def load_model(
    model_path: str,
    model_klass: Type[model_base.ModelBase],
    generator_klass: Type[item2user.Item2UserGenerator],
    device: str = "cpu",
    meta_path: str = None,
):
    return _ModelLazyFactory.get_instance(
        model_path,
        model_klass,
        generator_klass,
        device,
        meta_path,
    )
