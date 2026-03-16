import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import requests
    from typing import NamedTuple
    import csv
    import matplotlib.pyplot as plt
    from collections import defaultdict, Counter
    import random
    from machine_learning import split_data
    from vector_algebra import distance


@app.cell
def _():
    dataset_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    )
    data = requests.get(dataset_url)
    with open("iris.data", "w") as f:
        f.write(data.text)
    return


@app.class_definition
class LabeledPoint(NamedTuple):
    point: list[float]
    label: str


@app.function
def parse_iris_row(row: list[str]) -> LabeledPoint:
    measurements = [float(val) for val in row[:-1]]
    label = row[-1].split("-")[-1]
    return LabeledPoint(measurements, label)


@app.cell
def _():
    with open("iris.data", "r") as file:
        reader = csv.reader(file)
        iris_data = [parse_iris_row(row) for row in reader if row]
    return (iris_data,)


@app.cell
def _(iris_data):
    len(iris_data)
    return


@app.cell
def _(iris_data):
    points_by_species: dict[str, list[float]] = defaultdict(list)
    for iris in iris_data:
        points_by_species[iris.label].append(iris.point)
    return (points_by_species,)


@app.cell
def _(points_by_species: dict[str, list[float]]):
    metrics = ["sepal length", "sepal width", "petal length", "petal width"]
    pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
    marks = ["+", ".", "x"]

    fig, ax = plt.subplots(2, 3)

    for row in range(2):
        for col in range(3):
            i, j = pairs[3 * row + col]
            ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
            for mark, (species, points) in zip(marks, points_by_species.items()):
                xs = [point[i] for point in points]
                ys = [point[j] for point in points]
                ax[row][col].scatter(xs, ys, marker=mark, label=species)

    ax[-1][-1].legend(loc="lower right", prop={"size": 6})
    plt.show()
    return


@app.function
def majority_vote(labels: list[str]) -> str:
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(n=1)[0]
    return winner


@app.function
def knn_classify(
    k: int, labeled_points: list[LabeledPoint], new_point: list[float]
) -> str:
    by_distance = sorted(
        labeled_points, key=lambda lp: distance(lp.point, new_point)
    )
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    return majority_vote(k_nearest_labels)


@app.cell
def _(iris_data):
    confusion_matrix: dict[tuple[str, str], int] = defaultdict(int)
    num_correct = 0

    for iris_row in iris_data:
        predicted = knn_classify(5, iris_data, iris_row.point)
        if predicted == iris_row.label:
            num_correct += 1
        confusion_matrix[(predicted, iris_row.label)] += 1
    pct_correct = num_correct / len(iris_data)
    print(pct_correct * 100)

    for val, num in confusion_matrix.items():
        print(val, num)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
