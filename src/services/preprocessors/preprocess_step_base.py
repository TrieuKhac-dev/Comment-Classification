from enum import Enum

class StepPriority(Enum):
    LOWEST = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    HIGHEST = 5


class PreprocessStepBase:
    conflicts: list[str] = []
    priority: StepPriority = StepPriority.LOWEST
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        self._conflicts = conflicts if conflicts is not None else self.__class__.conflicts
        self._priority = priority if priority is not None else self.__class__.priority
    
    def name(self) -> str:
        return self.__class__.__name__
    
    def get_conflicts(self) -> list[str]:
        return self._conflicts
    
    def get_priority(self) -> StepPriority:
        return self._priority
    
    def apply(self, text: str) -> str:
        raise NotImplementedError
