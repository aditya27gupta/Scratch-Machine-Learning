import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)
