from abc import ABC, abstractmethod

class IPipelineStep(ABC):
    @abstractmethod
    def run(self, context):
        pass