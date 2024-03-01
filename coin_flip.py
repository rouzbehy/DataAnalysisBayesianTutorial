from enum import Enum
from os import chown
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
from scipy.special import comb
from scipy.integrate import trapezoid

"""
    For the parameter estimation example of Chapter 2.
    Determining if a coin is fair.
"""


def uniform_prior(x: float) -> float:
    """
    unifrom prior for when we are keeping as open a mind as
    possible.
    x : probability that we get _head_ in a given toss of the coin.
    """
    if 0 <= x <= 1:
        return 1.0
    else:
        return 0


def gaussian_prior(x: float, mu: float, sigma: float) -> float:
    """
    prior for when we have (or think we have) some idea about
    the coin.
    """
    return np.exp(-0.5 * (x - mu) ** 2 / sigma**2) / (sigma * np.sqrt(2 * np.pi))


def likelihood_coin(x: np.longdouble, nheads: int, ntotal: int) -> np.longdouble:
    """
    likelihood function. For the coin flip example, it's just the
    binomial distribution.
    - x : probability of getting head in a toss
    - nheads: how many heads we got in _ntotal_ tosses
    """
    # comb(ntotal, nheads) *
    return pow(x, nheads * 1.0) * pow(1.0 - x, ntotal * 1.0 - nheads)


def posterior(
    x: float, nheads: int, ntotal: int, prior: Callable[[float], float]
) -> float:
    return likelihood_coin(x, nheads, ntotal) * prior(x)


if __name__ == "__main__":
    choices_of_samples = np.array(
        [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    )
    num_rows, num_cols = 5, 3
    true_H = 0.3
    ## choose which prior we want to use
    prior = lambda x: gaussian_prior(x, mu=0.2, sigma=0.5)
    poster_vec = np.vectorize(posterior)
    hvalues = np.linspace(0.0, 1.0, 300)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        sharex=True,
        gridspec_kw={"top": 0.99, "bottom": 0.1},
        figsize=(16, 9),
    )
    for i in range(num_rows):
        for j in range(num_cols):
            itotal = i * num_cols + j
            ntotal = choices_of_samples[itotal]
            ax = axes[i, j]
            ax.axvline(true_H, color="red")
            nheads = sum(np.random.binomial(n=1, p=true_H, size=ntotal) == 1)
            posts = poster_vec(hvalues, nheads, ntotal, prior)
            normalization = trapezoid(posts, hvalues)
            posts /= normalization
            nheads *= 1.0
            ax.text(0.1, 0.8, f"{nheads/ntotal=:0.3f}", transform=ax.transAxes)
            ax.plot(hvalues, posts)
    plt.show()
