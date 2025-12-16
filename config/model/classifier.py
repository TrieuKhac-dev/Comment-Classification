class ClassifierModelConfig:
    """Config LightGBM."""

    LIGHTGBM_PARAMS_BASE = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
    }

    LIGHTGBM_PARAMS_EXTRA = {
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbose": -1,
    }

    @property
    def merged_params(self):
        """Trả về params LightGBM đã gộp."""
        return {**self.LIGHTGBM_PARAMS_BASE, **self.LIGHTGBM_PARAMS_EXTRA}

    DEFAULT_SAVE_FORMAT = "pipeline"


classifier_config = ClassifierModelConfig()
