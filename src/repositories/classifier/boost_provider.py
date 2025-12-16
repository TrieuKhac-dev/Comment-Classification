import lightgbm as lgb
from src.interfaces.repositories.i_classifier_repository_provider import IClassifierRepoProvider

class BoostRepositoryProvider:
    def save(self, model, path: str) -> None:
        try:
            model.booster_.save_model(path)
            print(f"[BoostRepositoryProvider] Model saved using LightGBM booster at: {path}")
        except Exception as e:
            print(f"[BoostRepositoryProvider] Error saving model: {e}")

    def load(self, path: str):
        try:
            booster = lgb.Booster(model_file=path)
            print(f"[BoostRepositoryProvider] Model loaded using LightGBM booster from: {path}")
            return booster
        except Exception as e:
            print(f"[BoostRepositoryProvider] Error loading model: {e}")
            return None
