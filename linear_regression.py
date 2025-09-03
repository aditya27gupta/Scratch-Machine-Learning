import numpy as np
from metrics import r_squared, root_mean_squared_error


class LinearRegression:
    def __init__(self, lr: float, con_tol: float = 1e-6) -> None:
        self.coefficients = np.array([])
        self.intercept = 0
        self.learning_rate = lr
        self.convergence_tol = con_tol

    def standardize(self, arr: np.ndarray) -> np.ndarray:
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        return (arr - mean) / std

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return np.dot(arr, self.coefficients) + self.intercept

    def compute_cost(self, predictions: np.ndarray, y: np.ndarray) -> float:
        m = len(predictions)
        errors = predictions - y
        return np.sum(errors**2) / (2 * m)

    def compute_gradients(self, predictions: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        m = len(predictions)
        dw = np.dot(predictions - y, x) / m
        db = np.sum(predictions - y) / m
        return dw, db

    def fit(self, arr: np.ndarray, y: np.ndarray, iterations: int) -> None:
        std_arr = self.standardize(arr)
        _, n_features = std_arr.shape
        self.coefficients = np.zeros(n_features)

        for _ in range(iterations):
            predictions = self.forward(std_arr)
            cost = self.compute_cost(predictions, y)
            dw, db = self.compute_gradients(predictions, std_arr, y)
            self.coefficients -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
            if cost < self.convergence_tol:
                break

    def predict(self, arr: np.ndarray) -> np.ndarray:
        std_arr = self.standardize(arr)
        return self.forward(std_arr)


def main() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(1, 10, size=100)
    x = y * rng.normal(1, 0.1, size=100)
    x = x.reshape(-1, 1)
    linear_regression = LinearRegression(lr=0.01)
    linear_regression.fit(x, y, 10000)
    print(f"Coefficient: {linear_regression.coefficients} and Intercept: {linear_regression.intercept}")
    predictions = linear_regression.predict(x)
    print(f"RMSE: {root_mean_squared_error(y, predictions)}")
    print(f"R^2: {r_squared(y, predictions)}")


if __name__ == "__main__":
    main()
