from abc import ABC, abstractmethod

class IListener(ABC):
    @abstractmethod
    def update(self, level: str, message: str, *args, **kwargs) -> None:
        ...
