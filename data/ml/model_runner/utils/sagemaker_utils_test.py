import dataclasses

from absl.testing import absltest

from data.ml.model_runner.utils import sagemaker_utils


class SagemakerUtilsTest(absltest.TestCase):
    def test_init_sagemaker_run(self):
        run = sagemaker_utils.init_run()
        assert isinstance(run, sagemaker_utils.Run)

    def test_guard_none(self):
        guarded = sagemaker_utils.guard_none(
            {
                "a": None,
                "b": {
                    "c": None,
                    "d": "hello",
                },
                "e": [1, 2, 3, 4, 5, None],
            }
        )

        assert guarded == {
            "a": "None",
            "b": {
                "c": "None",
                "d": "hello",
            },
            "e": [1, 2, 3, 4, 5, "None"],
        }

    def test_dataclass_to_dict(self):
        @dataclasses.dataclass
        class Son:
            name: str
            age: int

        @dataclasses.dataclass
        class Parent:
            name: str
            son: Son

        parent = Parent(name="john", son=Son(name="tom", age=12))

        d = sagemaker_utils.dataclass_to_dict(parent)

        assert d == {
            "name": "john",
            "son": {
                "name": "tom",
                "age": 12,
            },
        }


if __name__ == "__main__":
    absltest.main()
