import abc
from typing import Any


class HiveTable(abc.ABC):
    @abc.abstractmethod
    def create(self) -> None:
        pass

    @abc.abstractmethod
    def create_database(self) -> None:
        pass

    @abc.abstractmethod
    def drop(self, delete_data: bool) -> None:
        pass

    @abc.abstractmethod
    def save(self, df: Any) -> None:
        pass

    @abc.abstractmethod
    def load(self) -> Any:
        pass

    @abc.abstractmethod
    def check_if_exists(self) -> bool:
        pass

    @abc.abstractmethod
    def check_schema_changed(self) -> bool:
        pass
