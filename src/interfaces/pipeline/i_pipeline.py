from abc import ABC, abstractmethod

class IPipeline(ABC):
    @abstractmethod
    def run(self, context):
        pass