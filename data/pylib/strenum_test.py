import enum

from data.pylib import strenum


def test_str_enum_auto():
    # pylint: disable=invalid-name
    class Example(strenum.StrEnum):
        UPPER_CASE = enum.auto()
        lower_case = enum.auto()
        MixedCase = enum.auto()

    assert Example.UPPER_CASE == "UPPER_CASE"
    assert Example.lower_case == "lower_case"
    assert Example.MixedCase == "MixedCase"


def test_str_enum_manual():
    # pylint: disable=invalid-name
    class Example(strenum.StrEnum):
        UPPER_CASE = "UPPER_CASE"
        lower_case = "lower_case"
        MixedCase = "MixedCase"

    assert Example.UPPER_CASE == "UPPER_CASE"
    assert Example.lower_case == "lower_case"
    assert Example.MixedCase == "MixedCase"
