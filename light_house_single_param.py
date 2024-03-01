import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


def convert_theta_to_x(theta: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    converts theta values to x, location of observation.
    -- theta, List[float]: angles of emission of light by the lighthouse
    -- alpha, float: distance of lighthouse relative to some fixed
    point along the coast.
    -- beta, float: distance of the lighthouse out in the water.
    """
    return alpha + beta * np.tan(theta)


def log_likelihood_posterior(x: np.ndarray, alpha: float, beta: float) -> float:
    return sum(-np.log(beta**2 + (x - alpha) ** 2))


def main(alpha_0: float, beta_0: float) -> None:
    """
    alpha_0, float: the true value of alpha
    beta_0, float, the true value of beta
    """
    alpha_min, alpha_max = -5, 5
    alpha_vals = np.linspace(alpha_min, alpha_max, 500)
    num_samples = [1, 2, 4, 8, 64, 512]
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
    for i, num in enumerate(num_samples):
        ax_row = i % 3
        ax_col = i // 3
        ax = axes[ax_row, ax_col]
        sample_thetas = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=num)
        xvals = convert_theta_to_x(sample_thetas, alpha_0, beta_0)
        posterior = []
        for alpha in alpha_vals:
            curr_post = log_likelihood_posterior(xvals, alpha, beta_0)
            posterior.append(curr_post)
        ## normalize the posterior:
        posterior = np.exp(posterior - max(posterior))
        norm = trapezoid(posterior, alpha_vals)
        posterior /= norm
        ax.plot(alpha_vals, posterior, color="blue")
        ax.text(0.8, 0.8, f"{num=}", transform=ax.transAxes)
        ax.axvline(alpha_0, color="red")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p\left(\alpha|{x_k},\beta,I)\right)$")
    for ax in axes[2]:
        ax.set_xlabel(r"$\alpha$ (km)")
    plt.show()


if __name__ == "__main__":
    main(1, 1)
