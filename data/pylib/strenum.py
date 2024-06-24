import enum

# The first argument to the `_generate_next_value_` function of the `enum.Enum`
# class is documented to be the name of the enum member, not the enum class:
#
#     https://docs.python.org/3.6/library/enum.html#using-automatic-values
#
# Pylint, though, doesn't know about this so we need to disable it's check for
# `self` arguments.


class StrEnum(str, enum.Enum):
    """
    StrEnum is a Python ``enum.Enum`` that inherits from ``str``. The default
    ``auto()`` behavior uses the member name as its value.

    Example usage::
        class Example(StrEnum):
            UPPER_CASE = auto()
            lower_case = auto()
            MixedCase = auto()
        assert Example.UPPER_CASE == "UPPER_CASE"
        assert Example.lower_case == "lower_case"
        assert Example.MixedCase == "MixedCase"
    """

    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, (str, enum.auto)):
            raise TypeError(f"Values of StrEnums must be strings: {value!r} is a {type(value)}")
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self):
        return str(self.value)

    def _generate_next_value_(name, *_):  # pylint: disable=no-self-argument
        return name
