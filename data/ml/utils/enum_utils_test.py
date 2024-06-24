import enum

from absl.testing import absltest

from data.ml.utils import enum_utils


class FileUtilsTest(absltest.TestCase):
    def test_str_enum(self):
        class TestStrEnum(enum_utils.StrEnum):
            A = enum.auto()
            B = enum.auto()

        assert TestStrEnum.A == "A"
        assert TestStrEnum.B == "B"

        for member in TestStrEnum:
            assert member.lower() in ("a", "b")


if __name__ == "__main__":
    absltest.main()
