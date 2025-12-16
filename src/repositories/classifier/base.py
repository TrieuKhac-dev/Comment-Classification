from src.interfaces.repositories import IClassifierRepository
from .factory import ClassifierRepositoryFactory

class ClassifierRepository(IClassifierRepository):
    def save(self, model, path: str, save_type: str = "joblib"):
        try:
            if save_type is None:
                raise ValueError("save_type must be provided")
            if path is None:
                raise ValueError("path must be provided")
            if model is None:
                raise ValueError("model must be provided")

            provider = ClassifierRepositoryFactory.get(save_type)
            provider.save(model, path)
        except Exception as e:
            print(f"[ClassifierRepository] Error when save model: {e}")

    def load(self, path: str, load_type: str = "joblib"):
        try:
            if load_type is None:
                raise ValueError("load_type must be provided")
            if path is None:
                raise ValueError("path must be provided")
            
            provider = ClassifierRepositoryFactory.get(load_type)
            return provider.load(path)
        except Exception as e:
            print(f"[ClassifierRepository] Error when load model: {e}")
            return None