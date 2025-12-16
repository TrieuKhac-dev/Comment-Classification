class ClassifierModelConfig:
    """Config LightGBM."""

    # Các tham số cơ bản cho LightGBM
    LIGHTGBM_PARAMS_BASE = {
        "n_estimators": 200,         # Số lượng cây quyết định (số vòng lặp boosting)
        "learning_rate": 0.05,      # Tốc độ học (learning rate)
        "num_leaves": 31,           # Số lượng lá tối đa cho mỗi cây
        "random_state": 42,         # Seed cho random (tái lập kết quả)
    }

    # Các tham số mở rộng cho LightGBM
    LIGHTGBM_PARAMS_EXTRA = {
        "max_depth": -1,            # Độ sâu tối đa của cây (-1: không giới hạn)
        "min_child_samples": 20,    # Số lượng mẫu tối thiểu ở một leaf
        "subsample": 0.8,           # Tỷ lệ mẫu lấy ngẫu nhiên cho mỗi cây (bagging)
        "colsample_bytree": 0.8,    # Tỷ lệ cột (feature) lấy ngẫu nhiên cho mỗi cây
        "reg_alpha": 0.1,           # Tham số regularization L1
        "reg_lambda": 0.1,          # Tham số regularization L2
        "n_jobs": -1,               # Số luồng CPU sử dụng (-1: dùng tối đa)
        "verbose": -1,              # Mức độ log (-1: tắt log)
    }

    @property
    def merged_params(self):
        """Trả về params LightGBM đã gộp."""
        return {**self.LIGHTGBM_PARAMS_BASE, **self.LIGHTGBM_PARAMS_EXTRA}

    DEFAULT_SAVE_FORMAT = "pipeline"


classifier_config = ClassifierModelConfig()
