"""Utilities for Quasi-Monte Carlo sampling permutations."""

import numpy as np
from scipy.stats.qmc import MultivariateNormalQMC, Sobol


def permutohedron_samples(p: int, num_perms: int, seed: int = 42) -> np.ndarray:
    """Generate quasi-Monte Carlo permutations via permutohedron projection.

    Samples points on the (p-1)-dimensional unit sphere using QMC, then projects
    them onto the permutohedron to obtain low-discrepancy permutations.

    Args:
        p (int): Number of features.
        num_perms (int): Number of permutations to generate.
        seed (int, optional): Random seed for QMC sampler. Defaults to 42.

    Returns:
        np.ndarray: Array of permutations, shape (num_perms, p).
    """
    qmc = MultivariateNormalQMC(np.zeros(p - 1), seed=seed, inv_transform=False)
    samples = qmc.random(num_perms)
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)

    # Project onto permutohedron
    tril_part = np.tril(np.ones((p - 1, p)))
    diag_part = np.diag(-np.arange(1, p), 1)[:-1]
    U = tril_part + diag_part
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    samples = samples @ U
    return np.argsort(samples, axis=1)


def argsort_samples(p: int, num_perms: int, seed: int = 42) -> np.ndarray:
    """Generate quasi-Monte Carlo permutations via argsort of Sobol sequence.

    Samples low-discrepancy points from the Sobol sequence and converts them to
    permutations by sorting (argsort) each sample.

    Args:
        p (int): Number of features.
        num_perms (int): Number of permutations to generate.
        seed (int, optional): Random seed for Sobol sampler. Defaults to 42.

    Returns:
        np.ndarray: Array of permutations, shape (num_perms, p).
    """
    qmc = Sobol(p, seed=seed)
    samples = qmc.random(num_perms)
    return np.argsort(samples, axis=1)
