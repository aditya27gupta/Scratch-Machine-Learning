# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib==3.10.7",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    import enum, random
    import math


@app.class_definition
class Kid(enum.Enum):
    BOY = 0
    GIRL = 1


@app.function
def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])


@app.cell
def _():
    both_girls = older_girl = either_girl = 0
    random.seed(0)

    for _ in range(10_000):
        younger = random_kid()
        older = random_kid()

        if older == Kid.GIRL:
            older_girl += 1
        if older == Kid.GIRL and younger == Kid.GIRL:
            both_girls += 1
        if older == Kid.GIRL or younger == Kid.GIRL:
            either_girl += 1

    print("P(both | older):", both_girls / older_girl)
    print("P(both | either): ", both_girls / either_girl)
    return


@app.function
def uniform_pdf(x: float) -> int:
    return 1 if 0 <= x <= 1 else 0


@app.function
def uniform_cdf(x: float) -> int:
    if x < 0:
        return 0
    elif x < 1:
        return x
    return 1


@app.function
def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    first = 1 / (math.sqrt(2 * math.pi) * sigma)
    second = -1 * math.pow(x - mu, 2) / (2 * math.pow(sigma, 2))
    return first * math.exp(second)


@app.cell
def _():
    import matplotlib.pyplot as plt

    xs = [x / 10.0 for x in range(-50, 50)]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], "-", label="mu=0,sigma=1")
    plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], "--", label="mu=0,sigma=2")
    plt.plot(
        xs, [normal_pdf(x, sigma=0.5) for x in xs], ":", label="mu=0,sigma=0.5"
    )
    plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], "-.", label="mu=-1,sigma=1")
    plt.legend()
    plt.title("Various Normal pdfs")
    plt.show()
    return plt, xs


@app.function
def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


@app.cell
def _(plt, xs):
    plt.figure(figsize=(8, 4))
    plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], "-", label="mu=0,sigma=1")
    plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], "--", label="mu=0,sigma=2")
    plt.plot(
        xs, [normal_cdf(x, sigma=0.5) for x in xs], ":", label="mu=0,sigma=0.5"
    )
    plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], "-.", label="mu=-1,sigma=1")
    plt.legend(loc=4)  # bottom right
    plt.title("Various Normal cdfs")
    plt.show()
    return


@app.function
def inverse_normal_cdf(
    p: float, mu: float = 0, sigma: float = 1, tol: float = 1e-6
):
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tol=tol)

    low_z, hi_z = -10.0, 10.0
    while hi_z - low_z > tol:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z
    return mid_z


@app.function
def bernoulli_trial(p: float) -> int:
    return 1 if random.random() < p else 0


@app.function
def binomial(n: int, p: float) -> int:
    return sum(bernoulli_trial(p) for _ in range(n))


@app.cell
def _(plt):
    from collections import Counter

    p = 0.75
    n = 100
    num_points = 10_000

    data = [binomial(n, p) for _ in range(num_points)]
    # use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar(
        [x - 0.4 for x in histogram.keys()],
        [v / num_points for v in histogram.values()],
        0.8,
        color="0.75",
    )
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    # use a line chart to show the normal approximation
    x_s = range(min(data), max(data) + 1)
    ys = [
        normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in x_s
    ]
    plt.plot(x_s, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
