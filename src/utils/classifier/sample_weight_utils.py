import numpy as np

def weight_by_class_balance(y: np.ndarray) -> np.ndarray:
    """
    Cân bằng theo tỷ lệ lớp (positive vs negative).
    """
    n = len(y)
    pos = y.sum()
    neg = n - pos
    if pos == 0 or neg == 0:
        return np.ones(n)
    w_pos = n / (2 * pos)
    w_neg = n / (2 * neg)
    return np.where(y == 1, w_pos, w_neg)


def weight_inverse_frequency(y: np.ndarray) -> np.ndarray:
    """
    Trọng số ngược với tần suất xuất hiện.
    """
    n = len(y)
    pos = y.sum()
    neg = n - pos
    if pos == 0 or neg == 0:
        return np.ones(n)
    freq_pos = pos / n
    freq_neg = neg / n
    w_pos = 1.0 / freq_pos
    w_neg = 1.0 / freq_neg
    return np.where(y == 1, w_pos, w_neg)


def weight_cost_sensitive(y: np.ndarray, cost_pos: float = 5.0, cost_neg: float = 1.0) -> np.ndarray:
    """
    Trọng số theo chi phí sai nhãn (ví dụ: sai lớp dương tốn kém hơn).
    """
    return np.where(y == 1, cost_pos, cost_neg)


def weight_hard_example(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Trọng số theo độ khó (mẫu nào dự đoán sai thì tăng trọng số).
    """
    return np.where(y_true != y_pred, 2.0, 1.0)
