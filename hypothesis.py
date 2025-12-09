import marimo

__generated_with = "0.18.3"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    from typing import Tuple
    import math
    from probability import normal_cdf, inverse_normal_cdf


@app.function
def normal_approx_binom(n: int, p: float) -> Tuple[float, float]:
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma


@app.cell
def _():
    normal_approx_binom(1000, 0.5)
    return


@app.function
def normal_proba_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    return 1 - normal_cdf(lo, mu, sigma)


@app.cell
def _():
    normal_proba_above(0.2, 500, 15.811)
    return


@app.function
def normal_proba_between(
    lo: float, hi: float, mu: float = 0, sigma: float = 1
) -> float:
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)


@app.cell
def _():
    normal_proba_between(0.2, 0.6, 500, 15.811)
    return


@app.function
def normal_proba_outside(
    lo: float, hi: float, mu: float = 0, sigma: float = 1
) -> float:
    return 1 - normal_proba_between(lo, hi, mu, sigma)


@app.cell
def _():
    normal_proba_outside(0.2, 0.6, 500, 15.811)
    return


@app.function
def normal_upper_bound(prob: float, mu: float = 0, sigma: float = 1) -> float:
    return inverse_normal_cdf(prob, mu, sigma)


@app.cell
def _():
    normal_upper_bound(0.1, 500, 15.811)
    return


@app.function
def normal_lower_bound(prob: float, mu: float = 0, sigma: float = 1) -> float:
    return inverse_normal_cdf(1 - prob, mu, sigma)


@app.cell
def _():
    normal_lower_bound(0.1, 500, 15.811)
    return


@app.function
def normal_two_sided_bounds(
    prob: float, mu: float = 0, sigma: float = 1
) -> Tuple[float, float]:
    tail_prob = (1 - prob) / 2
    upper_bound = normal_lower_bound(tail_prob, mu, sigma)
    lower_bound = normal_upper_bound(tail_prob, mu, sigma)
    return lower_bound, upper_bound


@app.cell
def _():
    normal_two_sided_bounds(0.95, 500, 15.811)
    return


@app.cell
def _():
    lo, hi = normal_two_sided_bounds(0.95, 500, 15.811)
    mu1, sigma1 = normal_approx_binom(1000, 0.55)
    type_2_prob = normal_proba_between(lo, hi, mu1, sigma1)
    lo, hi, mu1, sigma1, 1 - type_2_prob
    return


@app.function
def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    if x >= mu:
        return 2 * normal_proba_above(x, mu, sigma)
    return 2 * normal_cdf(x, mu, sigma)


@app.cell
def _():
    two_sided_p_value(529.5, 500, 15.811)
    return


@app.function
def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma


@app.function
def a_b_stats(Na: int, na: int, Nb: int, nb: int) -> float:
    pa, sigma_a = estimated_parameters(Na, na)
    pb, sigma_b = estimated_parameters(Nb, nb)
    return (pb - pa) / math.sqrt(sigma_a**2 + sigma_b**2)


@app.cell
def _():
    z = a_b_stats(1000, 200, 1000, 150)
    print(z)
    two_sided_p_value(z)
    return


@app.function
def B(alpha: float, beta: float) -> float:
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)


@app.function
def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)


@app.cell
def _():
    beta_pdf(0.33, 20, 20)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
