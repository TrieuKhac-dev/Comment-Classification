from src.interfaces.processing import IPreprocessor
from src.services.preprocessors.preprocess_step_base import PreprocessStepBase

class TextPreprocessorService(IPreprocessor):

    # Check 2 chiều: step bị conflict hoặc step conflict với step đã add
    def _is_conflicted(
        self, 
        step_name: str, 
        step_conflicts: list[str], 
        all_conflicts: set
    ) -> bool:
        return step_name in all_conflicts or any(c in all_conflicts for c in step_conflicts)
    
    def _filter_conflicted_steps(
        self, 
        steps: list[PreprocessStepBase], 
        config: dict
    ) -> tuple[list[PreprocessStepBase], dict]:
        final_steps = []
        all_conflicts = set()
        final_config = {}
        
        for step in steps:
            step_name = step.name()
            step_conflicts = step.get_conflicts()
            
            if self._is_conflicted(step_name, step_conflicts, all_conflicts):
                continue
            
            final_steps.append(step)
            all_conflicts.add(step_name)
            all_conflicts.update(step_conflicts)
            
            if step_name in config:
                final_config[step_name] = config[step_name]
        
        return final_steps, final_config
    
    def __init__(self, steps: list[PreprocessStepBase], config: dict):
        sorted_steps = sorted(steps, 
                              key=lambda s: s.get_priority().value, 
                              reverse=True)
        
        self.steps, self.config = self._filter_conflicted_steps(sorted_steps, config)
    
    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        for step in self.steps:
            text = step.apply(text)
        return text
    
    def preprocess_batch(self, texts: list[str]) -> list[str]:
        return [self.preprocess(t) for t in texts]
    
    def get_config_summary(self) -> dict:
        return self.config.copy()
