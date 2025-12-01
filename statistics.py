import marimo

__generated_with = "0.18.1"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    from typing import List
    import math
    from collections import Counter
    from vector_algebra import sum_of_squares, dot


@app.cell
def _():
    friends = [95,42,64,63,43,46,61,36,32,94,31,1,94,57,11,93,55,21,30,16,75,99,65,75,31,33,71,17,65,63,50,60,92,60,32,37,72,37,1,46,78,80,25,18,31,77,33,94,19,79,58,79,59,10,32,37,21,36,67,45,31,47,15,25,18,5,72,5,74,76,37,1,23,78,93,44,57,52,37,64,27,2,99,49,34,46,56,57,55,75,30,8,25,65,17,38,34,98,33,91,7,2,87,30,23,74,21,53,53,90,25,67,67,98,72,85,48,78,94,22,15,87,31,14,19,81,97,66,90,70,59,93,5,13,63,3,84,59,46,11,67,90,59,38,51,92,59,71,14,62,38,51,97,63,32,73,49,92,72,78,61,68,74,28,96,80,64,45,75,52,88,18,37,14,10,33,90,20,28,14,17,76,66,79,77,10,8,79,88,74,2,82,40,88,98,43,92,20,27,3,83,25,89,82,67,26,62,60,25,45,98,19,64,69,65,90,40,4,19,73,58,40,81,3,73,76,4,46,49,77,29,13,9,97,60,36,73,61,97,64,69,16,17,50,78,21,51,96,16,67,99,98,80,69,75,79,21,91,91,87,5,90,2,91,30,67,23,67,66,13,99,34,66,52,68,98,9,87,28,49,82,7,68,91,27,92,3,92,46,74,32,45,71,39,39,13,14,24,92,69,13,35,84,97,70,50,42,100,16,86,19,32,85,60,44,96,92,39,31,54,22,27,40,63,91,9,8,60,67,69,92,27,64,4,3,81,73,42,19,40,18,50,3,23,62,23,48,78,71,64,82,58,24,49,64,2,49,49,28,94,96,5,35,42,56,8,76,9,99,64,40,8,78,20,69,1,66,66,1,67,99,47,4,58,27,69,37,72,31,25,18,63,90,1,7,74,37,21,27,75,7,93,56,42,74,2,52,59,95,94,37,4,41,3,1,49,85,1,19,75,86,79,36,3,76,61,99,37,28,52,72,33,10,67,96,97,74,14,48,95,36,49,73,59,86,98,32,84,54,47,98,96,17,23,70,5,33,62,42,12,72,48,31,48,3,72,97,93,85,43,22,23,82,61,58,79,59,79,15,50,5,97,8,53,43,93,66,62,9,50,63,64,39,97,71,11,31,58,98,86,48,82,45,1,27,37,49,24,32,1,77,38,46,15,51,59,26,16,90,58,56,36,70,23,47,23,75,88,60,50,97,63,4,16,98,81,8,72,38,86,76,58,23,2,14,62,8,93,19,43,64,56,60,57,46,91,87,55,3,95,20,46,99,75,30,6,92,71,42,33,30,62,2,14,12,10,61,4,36,56,71,78,54,35,21,71,97,15,44,82,89,24,22,34,88,34,6,15,7,62,64,25,69,28,87,83,1,93,15,42,17,69,74,87,16,67,48,4,68,12,57,35,1,69,64,34,11,72,73,94,29,78,40,23,44,52,78,44,18,19,66,22,2,16,29,61,92,21,43,33,56,78,7,62,18,63,3,13,84,77,65,37,3,28,41,23,21,37,57,41,94,16,76,69,28,44,58,51,92,85,79,58,55,59,24,11,12,54,56,14,8,80,33,14,39,54,93,6,51,64,14,86,56,86,3,23,77,2,32,26,84,11,40,49,31,75,66,11,29,95,24,22,38,19,17,69,8,11,49,51,37,82,33,82,97,26,72,48,81,46,43,32,17,90,30,67,99,81,15,96,39,75,99,12,58,85,37,48,58,84,73,1,2,77,19,64,46,99,14,44,39,60,26,7,37,41,6,37,37,25,28,39,41,37,19,10,57,19,66,44,4,30,68,60,40,39,94,36,45,14,40,99,85,73,32,59,70,7,34,100,25,94,47,9,81,55,4,70,20,79,72,68,49,97,45,28,2,28,17,61,97,49,2,39,11,19,32,28,94,90,53,18,73,56,16,60,35,21,28,23,34,84,64,93,95,63,66,10,71,41,51,97,34,23,2,25,31,18,2,74,99,64,73,66,71,39,8,68,64,17,86,64,18,17,31,5,45,72,28,92,67,69,49,28,38,55,24,13,3,59,11,54,47,67,37,54,97,29,61,85,63,14,12,20,26,33,50,78,72,7,96,15,8,91,4,79,39,71,79,38,39,51,17,61,94,53,12,11,82,93,3,83,42,84,59,28,81,78,75,51,53,20,9,4,23,15,40,33,8,35,92,6,83,76,3,88,95,20,12,77,39,65,63,11,100,78,14,70,26,89,3,48,39,60,35,58,48,10,14,88,95,90,39,78,70,68,72,27,89,61]
    return (friends,)


@app.function
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


@app.cell
def _(friends):
    mean(friends)
    return


@app.function
def median(xs: List[float]) -> float:
    sorted_xs = sorted(xs)
    n = len(xs)
    if n % 2 != 0:
        return sorted_xs[n // 2]
    else:
        return (sorted_xs[n // 2 - 1] + sorted_xs[n // 2]) / 2


@app.cell
def _(friends):
    median(friends)
    return


@app.function
def quantile(xs: List[float], p: float) -> float:
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


@app.cell
def _(friends):
    quantile(friends, 0.5)
    return


@app.function
def mode(xs: List[float]) -> List[float]:
    counts = Counter(xs)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


@app.cell
def _(friends):
    mode(friends)
    return


@app.function
def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)


@app.cell
def _(friends):
    data_range(friends)
    return


@app.function
def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


@app.function
def variance(xs: List[float]) -> float:
    assert len(xs) >= 2, "variance requires atleast two elements"
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / n - 1


@app.cell
def _(friends):
    variance(friends)
    return


@app.function
def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))


@app.cell
def _(friends):
    standard_deviation(friends)
    return


@app.function
def interquantile_range(xs: List[float]) -> float:
    return quantile(xs, 0.75) - quantile(xs, 0.25)


@app.cell
def _(friends):
    interquantile_range(friends)
    return


@app.function
def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys should have number of elements"
    return dot(de_mean(xs), de_mean(ys)) / len(xs)-1


@app.cell
def _(friends):
    covariance(friends, friends)
    return


@app.function
def correlation(xs: List[float], ys: List[float]) -> float:
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / (stdev_x * stdev_y)
    return 0


@app.cell
def _(friends):
    correlation(friends, friends)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
