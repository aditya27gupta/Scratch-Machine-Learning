# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    from statistics import correlation, standard_deviation, mean, de_mean


@app.function
def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha


@app.function
def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return predict(alpha, beta, x_i) - y_i


@app.function
def sum_of_sq_error(
    alpha: float, beta: float, x: list[float], y: list[float]
) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


@app.function
def least_square_fit(x: list[float], y: list[float]) -> tuple[float, float]:
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


@app.cell
def _():
    x = [i for i in range(-100, 110, 10)]
    y = [3 * i - 5 for i in x]
    least_square_fit(x, y)
    return x, y


@app.function
def total_sum_of_squares(y: list[float]) -> float:
    return sum(v**2 for v in de_mean(y))


@app.function
def r_squared(
    alpha: float, beta: float, x: list[float], y: list[float]
) -> float:
    return 1 - (sum_of_sq_error(alpha, beta, x, y) / total_sum_of_squares(y))


@app.cell
def _(x, y):
    r_squared(-5.0, 3.000545603346367, x, y)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
