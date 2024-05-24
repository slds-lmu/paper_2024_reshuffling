from functools import partial

import numpy as np
import torch


def simulate_gp(x: torch.Tensor, mu: callable, cov: callable) -> torch.Tensor:
    """
    Simulate a Gaussian process.

    :param x: A tensor of shape (n, d) where n is the number of points and d is the dimensionality of each point.
    :param mu: Function to compute the mean of the Gaussian process.
    :param cov: Function to compute the covariance matrix of the Gaussian process.

    :return: A tensor of shape (n, d) representing the simulated Gaussian process.
    """
    sigma_2 = 1e-5
    n = x.shape[0]
    d = x.shape[1]
    mu_x = mu(x)
    K_x = cov(x, x)
    K_x += sigma_2 * torch.eye(n)
    eigenvalues, eigenvectors = torch.linalg.eigh(K_x)
    positive_eigenvalues = torch.clamp(eigenvalues, min=0)
    sqrt_K_x = eigenvectors @ torch.diag(positive_eigenvalues.sqrt()) @ eigenvectors.T
    z = torch.normal(0, 1, (n, d))
    return mu_x + sqrt_K_x @ z


def mu_factory(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    The mean function of the Gaussian process.

    :param x: A tensor of shape (n, d) where n is the number of points and d is the dimensionality of each point.
    :param alpha: The alpha parameter of the mean function.

    :return: A tensor of shape (n, 1) representing the mean of the Gaussian process at each point.
    """
    values = alpha * (x - 0.5).pow(2)
    return values


def cov_factory(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscale: float,
    tau: float = None,
    shuffled: bool = False,
) -> torch.Tensor:
    """
    Vectorized computation of the covariance matrix of the Gaussian process.

    :param x1: A tensor of shape (n, d) where n is the number of points and d is the dimensionality of each point.
    :param x2: A tensor of shape (n, d) where n is the number of points and d is the dimensionality of each point.
    :param lengthscale: The lengthscale parameter of the covariance function.
    :param shuffled: Whether to assume a shuffled version of the covariance function.
    :param tau: The tau parameter of the shuffled covariance function.

    :return: A tensor of shape (n, n) representing the covariance matrix of the Gaussian process.
    """
    sq_dist = torch.sum((x1[:, None, :] - x2[None, :, :]) ** 2, dim=-1)
    K = torch.exp(-sq_dist / (2 * (lengthscale**2)))
    if shuffled:
        K = (1 - torch.eye(K.shape[0])) * (tau**2) * K + torch.eye(K.shape[0]) * K
    return K


def kernel_factory(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscale: float,
    tau: float = None,
    shuffled: bool = False,
) -> torch.Tensor:
    """
    Computation of the covariance between two points.

    :param x1: A tensor of shape (1, d) where d is the dimensionality.
    :param x2: A tensor of shape (1, d) where d is the dimensionality.
    :param lengthscale:  The lengthscale parameter of the covariance function.
    :param shuffled: Whether to assume a shuffled version of the covariance function.
    :param tau: The tau parameter of the shuffled covariance function.

    :return: A scalar tensor representing the covariance between the two points.
    """
    sq_dist = torch.sum((x1 - x2) ** 2, dim=-1)
    K = torch.exp(-sq_dist / (2 * (lengthscale**2)))
    if shuffled:
        if x1 == x2:
            K = (tau**2) * K
    return K


if __name__ == "__main__":
    import argparse
    import os

    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--lengthscale", type=float, default=0.5)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    results_path = "results"
    d = 1
    alpha = args.alpha
    lengthscale = args.lengthscale
    n_replicates = 10000

    x = torch.linspace(0, 1, 101).reshape(-1, d)
    mu = partial(mu_factory, alpha=alpha)
    kernel = partial(kernel_factory, lengthscale=lengthscale)
    cov = partial(cov_factory, lengthscale=lengthscale)
    taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    taus.sort()
    results = {tau: {"y": [], "y_shuffled": [], "y_diff": []} for tau in taus}
    for tau in taus:
        y_mu_y_list = []
        y_mu_y_shuffled_list = []
        y_diff_list = []
        for _ in tqdm(range(n_replicates)):
            y = simulate_gp(x, mu, cov)
            y_shuffled = simulate_gp(x, mu, partial(cov, shuffled=True, tau=tau))
            y_mu_y = mu(x[y.argmin(dim=0)])
            y_mu_y_shuffled = mu(x[y_shuffled.argmin(dim=0)])
            y_mu_y_list.append(y_mu_y)
            y_mu_y_shuffled_list.append(y_mu_y_shuffled)
            y_diff_list.append(y_mu_y - y_mu_y_shuffled)
        results[tau]["y"] = torch.tensor(y_mu_y_list)
        results[tau]["y_shuffled"] = torch.tensor(y_mu_y_shuffled_list)
        results[tau]["y_diff"] = torch.tensor(y_diff_list)

    results_df = pd.DataFrame(results[taus[0]])
    results_df["replicate"] = range(n_replicates)
    results_df["tau"] = taus[0]
    for tau in taus[1:]:
        results_df_tmp = pd.DataFrame(results[tau])
        results_df_tmp["replicate"] = range(n_replicates)
        results_df_tmp["tau"] = tau
        results_df = pd.concat([results_df, results_df_tmp], axis=0)
    results_df["alpha"] = alpha
    results_df["lengthscale"] = lengthscale
    results_df["scenario"] = str(alpha) + "_" + str(lengthscale)
    results_df = results_df.reset_index(drop=True)
    results_df.to_csv(
        os.path.join(
            results_path,
            f"results_alpha_{alpha}_lengthscale_{lengthscale}.csv",
        ),
        index=False,
    )
