import joblib
from src.interfaces.repositories import IClassifierRepoProvider

class JoblibRepositoryProvider(IClassifierRepoProvider):
    def save(self, model, path: str) -> None:
        try:
            joblib.dump(model, path)
            print(f"[JoblibSaveStrategy] Đã lưu mô hình bằng joblib tại: {path}")
        except Exception as e:
            print(f"[JoblibSaveStrategy] Lỗi lưu mô hình: {e}")

    def load(self, path: str):
        try:
            model = joblib.load(path)
            print(f"[JoblibSaveStrategy] Đã load mô hình bằng joblib từ: {path}")
            return model
        except Exception as e:
            print(f"[JoblibSaveStrategy] Lỗi load mô hình: {e}")
            return None


