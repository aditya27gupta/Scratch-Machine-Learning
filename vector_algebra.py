# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ruff==0.14.5",
# ]
# ///

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import List, Tuple, Callable
    import math
    return Callable, List, Tuple, math


@app.cell
def _(List):
    Vector = List[float]
    return (Vector,)


@app.cell
def _(Vector):
    def add(v: Vector, w: Vector) -> Vector:
        assert len(v) == len(w)
        return [v_i + w_i for v_i, w_i in zip(v, w)]
    return (add,)


@app.cell
def _(add):
    add([1, 2, 3], [4, 5, 6])
    return


@app.cell
def _(Vector):
    def substract(v: Vector, w: Vector) -> Vector:
        assert len(v) == len(w)
        return [v_i - w_i for v_i, w_i in zip(v, w)]
    return (substract,)


@app.cell
def _(substract):
    substract([5, 6, 7], [1, 2, 3])
    return


@app.cell
def _(List, Vector):
    def vector_sum(vectors: List[Vector]) -> Vector:
        assert vectors, "no vectors provided!"

        num_elements = len(vectors[0])
        assert all(len(v) == num_elements for v in vectors), "different sizes!"

        return [sum(vector[i] for vector in vectors) for i in range(num_elements)]
    return (vector_sum,)


@app.cell
def _(vector_sum):
    vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]])
    return


@app.cell
def _(Vector):
    def scalar_multiply(c: float, v: Vector) -> Vector:
        return [c * v_i for v_i in v]
    return (scalar_multiply,)


@app.cell
def _(scalar_multiply):
    scalar_multiply(2, [1, 2, 3])
    return


@app.cell
def _(List, Vector, scalar_multiply, vector_sum):
    def vector_mean(vectors: List[Vector]) -> Vector:
        n = len(vectors)
        return scalar_multiply(1 / n, vector_sum(vectors))
    return (vector_mean,)


@app.cell
def _(vector_mean):
    vector_mean([[1, 2], [3, 4], [5, 6]])
    return


@app.cell
def _(Vector):
    def dot(v: Vector, w: Vector) -> float:
        assert len(v) == len(w), "vector must be same length"
        return sum(v_i * w_i for v_i, w_i in zip(v, w))
    return (dot,)


@app.cell
def _(dot):
    dot([1, 2, 3], [4, 5, 6])
    return


@app.cell
def _(Vector, dot):
    def sum_of_squares(v: Vector) -> float:
        return dot(v, v)
    return (sum_of_squares,)


@app.cell
def _(sum_of_squares):
    sum_of_squares([1, 2, 3])
    return


@app.cell
def _(Vector, math, sum_of_squares):
    def magnitude(v: Vector) -> float:
        return math.sqrt(sum_of_squares(v))
    return (magnitude,)


@app.cell
def _(magnitude):
    magnitude([1, 2, 3])
    return


@app.cell
def _(Vector, substract, sum_of_squares):
    def squared_distance(v: Vector, w: Vector) -> float:
        return sum_of_squares(substract(v, w))
    return


@app.cell
def _(Vector, magnitude, substract):
    def distance(v: Vector, w: float) -> float:
        return magnitude(substract(v, w))
    return


@app.cell
def _(List):
    Matrix = List[List[float]]
    return (Matrix,)


@app.cell
def _(Matrix, Tuple):
    def shape(A: Matrix) -> Tuple[int, int]:
        num_rows = len(A)
        num_cols = len(A[0]) if A else 0
        return num_rows, num_cols
    return (shape,)


@app.cell
def _(shape):
    shape([[1, 2, 3], [4, 5, 6]])
    return


@app.cell
def _(Matrix, Vector):
    def get_rows(A: Matrix, i: int) -> Vector:
        return A[i]


    def get_column(A: Matrix, j: int) -> Vector:
        return [A_i[j] for A_i in A]
    return


@app.cell
def _(Callable, Matrix):
    def make_matrix(
        num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]
    ) -> Matrix:
        return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]
    return (make_matrix,)


@app.cell
def _(Matrix, make_matrix):
    def identity_matrix(n: int) -> Matrix:
        return make_matrix(n, n, lambda i, j: 1 if i == j else 0)
    return (identity_matrix,)


@app.cell
def _(identity_matrix):
    identity_matrix(3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
