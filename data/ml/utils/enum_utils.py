import enum


class StrEnum(str, enum.Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """Determine enum.auto(). it's ignored if value is set explicitly, .

        e.g) class Example(StrEnum):
                A = enum.auto()  # 'A'
                B = 'Explicit'  # 'Explicit'
        """
        return name

    def __str__(self):
        return self.name

    @classmethod
    def get_case_insensitive(cls, name):
        for member in cls:
            if member.lower() == name.lower():
                return member
        raise Exception(f"Invalid KeyError: {name}")
