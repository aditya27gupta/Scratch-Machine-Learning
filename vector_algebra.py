# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ruff==0.14.5",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    from typing import List, Tuple, Callable
    import math

    Vector = List[float]
    Matrix = List[List[float]]


@app.function
def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)
    return [v_i + w_i for v_i, w_i in zip(v, w)]


@app.cell
def _():
    add([1, 2, 3], [4, 5, 6])
    return


@app.function
def substract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)
    return [v_i - w_i for v_i, w_i in zip(v, w)]


@app.cell
def _():
    substract([5, 6, 7], [1, 2, 3])
    return


@app.function
def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors, "no vectors provided!"

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


@app.cell
def _():
    vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]])
    return


@app.function
def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]


@app.cell
def _():
    scalar_multiply(2, [1, 2, 3])
    return


@app.function
def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


@app.cell
def _():
    vector_mean([[1, 2], [3, 4], [5, 6]])
    return


@app.function
def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "vector must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


@app.cell
def _():
    dot([1, 2, 3], [4, 5, 6])
    return


@app.function
def sum_of_squares(v: Vector) -> float:
    return dot(v, v)


@app.cell
def _():
    sum_of_squares([1, 2, 3])
    return


@app.function
def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))


@app.cell
def _():
    magnitude([1, 2, 3])
    return


@app.function
def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(substract(v, w))


@app.function
def distance(v: Vector, w: float) -> float:
    return magnitude(substract(v, w))


@app.function
def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


@app.cell
def _():
    shape([[1, 2, 3], [4, 5, 6]])
    return


@app.function
def get_rows(A: Matrix, i: int) -> Vector:
    return A[i]


@app.function
def get_column(A: Matrix, j: int) -> Vector:
    return [A_i[j] for A_i in A]


@app.function
def make_matrix(
    num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]
) -> Matrix:
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]


@app.function
def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


@app.cell
def _():
    identity_matrix(3)
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
