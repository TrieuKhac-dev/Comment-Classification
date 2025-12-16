class DataConfig:
    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.2
    VAL_RATIO = 0.0
    
    RANDOM_SEED = 42

    TEXT_COLUMN = "comment"
    LABEL_COLUMNS = [
        "is_violation"
    ]

    DROP_DUPLICATES = True
    DROP_NA = True

    STRATIFY = True

    MIN_TEXT_LENGTH = 1
    MAX_TEXT_LENGTH = 1000


data_config = DataConfig()
