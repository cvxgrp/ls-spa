"""Utilities for Quasi-Monte Carlo sampling permutations."""

import numpy as np
from scipy.stats.qmc import MultivariateNormalQMC, Sobol


def permutohedron_samples(p: int, num_perms: int, seed: int = 42) -> np.ndarray:
    """Sample on surface of sphere.

    Args:
        p (int): The number of features.
        num_perms (int): The number of permutations to sample.
        seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
        np.ndarray: The permutations.
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
    """Sample on surface of sphere.

    Args:
        p (int): The number of features.
        num_perms (int): The number of permutations to sample.
        seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
        np.ndarray: The permutations.
    """
    qmc = Sobol(p, seed=seed)
    samples = qmc.random(num_perms)
    return np.argsort(samples, axis=1)
