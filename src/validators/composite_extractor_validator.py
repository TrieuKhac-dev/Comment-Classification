import numpy as np

class CompositeExtractorValidator:
    @staticmethod
    def validate_extractors(extractors: dict):
        if not extractors:
            raise ValueError("No extractors provided")
        for name, ext in extractors.items():
            if ext is None:
                raise ValueError(f"Extractor '{name}' is None")

    @staticmethod
    def validate_weights(extractors, weights):
        if weights is None:
            return
        if len(weights) != len(extractors):
            raise ValueError("Số lượng trọng số phải bằng số lượng extractor")
        for name in weights.keys():
            if name not in extractors:
                raise ValueError(f"Weight '{name}' not found in extractors")

    @staticmethod
    def validate_row_counts(features_list):
        rows = [f.shape[0] for f in features_list]
        if len(set(rows)) != 1:
            raise ValueError("All extractors must produce the same number of rows")

    @staticmethod
    def validate_dimensions(extractors, combine_mode):
        if combine_mode == 'mean':
            dims = [ext.get_dimension() for ext in extractors.values()]
            if len(set(dims)) != 1:
                raise ValueError("For 'mean' mode, all extractors must have the same dimension")
