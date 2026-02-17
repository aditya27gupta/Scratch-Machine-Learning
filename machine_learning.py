# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App()

with app.setup:
    import random
    from typing import TypeVar, List, Tuple

    X = TypeVar("X")
    Y = TypeVar("Y")


@app.function
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]


@app.cell
def _():
    data = [n for n in range(1_000)]
    train, test = split_data(data, 0.75)
    len(train), len(test)
    return


@app.function
def train_test_split(
    xs: List[X], ys: List[Y], test_pct: float
) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idx = [i for i in range(len(xs))]
    train_ids, test_ids = split_data(idx, 1 - test_pct)
    return (
        [xs[i] for i in train_ids],
        [xs[i] for i in test_ids],
        [ys[i] for i in train_ids],
        [ys[i] for i in test_ids],
    )


@app.function
def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


@app.function
def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)


@app.function
def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)


@app.function
def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
