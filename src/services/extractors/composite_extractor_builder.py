from typing import List, Dict

from src.containers.container import Container
from .composite_extractor import CompositeExtractor
from src.validators import CompositeExtractorValidator

class CompositeExtractorBuilder:
    def __init__(self, container: Container):
        self.container = container
        self.extractors_keys: List[str] = []
        self.weights: Dict[str, float] = {}
        self.combine_mode: str = "concat"

    def add_extractor(self, key: str, weight: float = 1.0):
        self.extractors_keys.append(key)
        self.weights[key] = weight
        return self

    def set_combine_mode(self, mode: str):
        self.combine_mode = mode
        return self

    def build(self) -> CompositeExtractor:
        extractors = {
            key: self.container.resolve(key)
            for key in self.extractors_keys
        }

        CompositeExtractorValidator.validate_extractors(extractors)

        return CompositeExtractor(
            extractors=extractors,
            weights=self.weights,
            combine_mode=self.combine_mode
        )
