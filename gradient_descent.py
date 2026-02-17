# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "ruff==0.15.0",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App()

with app.setup:
    from typing import Callable, TypeVar, List, Iterator
    import matplotlib.pyplot as plt
    from vector_algebra import Vector, add, scalar_multiply, vector_mean
    import random


@app.function
def differential_quotient(
    f: Callable[[float], float], x: float, h: float
) -> float:
    return (f(x + h) - f(x)) / h


@app.function
def square(x: float) -> float:
    return x * x


@app.function
def derivative(x: float) -> float:
    return 2 * x


@app.cell
def _():
    xs = range(-10, 11)
    actuals = [derivative(x) for x in xs]
    estimates = [differential_quotient(square, x, h=0.001) for x in xs]
    plt.title("Actual Derivatives vs Estimates")
    plt.plot(xs, actuals, "rx", label="Actual")
    plt.plot(xs, estimates, "b+", label="Estimates")
    plt.legend(loc=9)
    plt.show()
    return


@app.function
def partial_difference_quotient(
    f: Callable[[Vector], float], v: Vector, i: int, h: float
) -> float:
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


@app.function
def estimate_gradient(
    f: Callable[[Vector], float], v: Vector, h: float = 0.0001
):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


@app.function
def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


@app.function
def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]


@app.cell
def _():
    v = [random.uniform(-10, 10) for i in range(3)]
    for epoch in range(1_000):
        grad = sum_of_squares_gradient(v)
        v = gradient_step(v, grad, -0.01)
        if sum((abs(v_i) for v_i in v)) < 1e-6:
            print(f"Broke at Epoch: {epoch}")
            break

    print([int(v_i) for v_i in v])
    return


@app.cell
def _(error):
    def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
        slope, intercept = theta
        predicted = slope * x + intercept
        squared_error = (predicted - y) ** 2
        grad = [2 * error * x, 2 * error]
        return grad

    return (linear_gradient,)


@app.cell
def _(linear_gradient):
    inputs = [(x, 20 * x + 5) for x in range(-50, 50)]
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    learning_rate = 0.001

    for iteration in range(5_001):
        grad_ = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
        theta = gradient_step(theta, grad_, -learning_rate)
        if iteration % 500 == 0:
            print(iteration, theta)
    return inputs, learning_rate, theta


@app.cell
def _():
    T = TypeVar('T')

    def minibatches(dataset: List[T], batch_size: int, shuffle: bool) -> Iterator[List[T]]:
        batch_starts = [start for start in range(0, len(dataset), batch_size)]
        if shuffle:
            random.shuffle(batch_starts)
        for start in batch_starts:
            end = start + batch_size
            yield dataset[start: end]

    return (minibatches,)


@app.cell
def _(inputs, learning_rate, linear_gradient, minibatches, theta):
    theta_ = [random.uniform(-1, 1), random.uniform(-1, 1)]

    for it in range(1_001):
        for batch in minibatches(inputs, 20, True):
            gra_ = vector_mean([linear_gradient(x, y, theta_) for x, y in inputs])
            theta_ = gradient_step(theta_, gra_, -learning_rate)
        if it % 500 == 0:
            print(it, theta)
    return


@app.cell
def _(inputs, learning_rate, linear_gradient):
    thet_ = [random.uniform(-1, 1), random.uniform(-1, 1)]

    for ep in range(101):
        for x, y in inputs:
            gra = linear_gradient(x, y, thet_)
            thet_ = gradient_step(thet_, gra, -learning_rate)
        if ep % 20 == 0:
            print(ep, thet_)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
