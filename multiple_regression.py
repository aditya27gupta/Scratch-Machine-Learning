# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "tqdm==4.67.3",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    from vector_algebra import dot, Vector, vector_mean, add
    from gradient_descent import gradient_step
    from linear_regression import total_sum_of_squares
    import tqdm
    import random


@app.function
def predict(x: Vector, beta: Vector) -> float:
    return dot(x, beta)


@app.function
def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y


@app.function
def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


@app.function
def squared_error_gradient(x: Vector, y: float, beta: Vector) -> float:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


@app.function
def least_square_fit(
    xs: list[Vector],
    ys: list[Vector],
    num_steps: int,
    learning_rate: float = 0.001,
    batch_size: int = 1,
) -> Vector:
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="Least Squares Fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start : start + batch_size]
            batch_ys = ys[start : start + batch_size]
            gradient = vector_mean(
                [
                    squared_error_gradient(x, y, guess)
                    for x, y in zip(batch_xs, batch_ys)
                ]
            )
            guess = gradient_step(guess, gradient, -learning_rate)
            guess = [round(g, 4) for g in guess]

    return guess


@app.cell
def _():
    xs = [
        [
            random.randint(1, 20),
            random.randint(1, 20),
            random.randint(1, 20),
            5,
        ]
        for _ in range(10_000)
    ]
    ys = [2 * x[0] + 0.2 * x[1] + 0.6 * x[2] - x[3] for x in xs]

    beta1 = least_square_fit(xs, ys, batch_size=100, num_steps=1000)
    beta1
    return beta1, xs, ys


@app.function
def multiple_r_squared(xs: list[Vector], ys: Vector, beta: Vector) -> float:
    sum_of_sq_errors = sum(error(x, y, beta) ** 2 for x, y in zip(xs, ys))
    return 1.0 - sum_of_sq_errors / total_sum_of_squares(ys)


@app.cell
def _(beta1, xs, ys):
    multiple_r_squared(xs, ys, beta1)
    return


@app.function
def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])


@app.function
def squaredd_error_ridge(
    x: Vector, y: float, beta: Vector, alpha: float
) -> float:
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)


@app.function
def lasso_penalty(beta: Vector, alpha: float) -> float:
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])


@app.function
def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
    return [0.0] + [2 * alpha * beta_j for beta_j in beta[1:]]


@app.cell
def _(sqerror_gradient):
    def sqerror_ridge_gradient(
        x: Vector, y: float, beta: Vector, alpha: float
    ) -> Vector:
        return add(
            sqerror_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha)
        )

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
