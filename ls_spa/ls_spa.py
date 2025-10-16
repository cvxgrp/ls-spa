# Copyright 2024 Logan Bell, Nikhil Devanathan, and Stephen Boyd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains a method to efficiently estimate a
Shapley attribution for least squares problems.

This method is described in the paper Efficient Shapley Performance
Attribution for Least-Squares Regression (arXiv:2310.19245) by Logan
Bell, Nikhil Devanathan, and Stephen Boyd.
"""  # noqa: D205

import itertools as it
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy as sp
from numpy import random

# The maximum number of features for which we can display the full attribution.
MAX_ATTR_DISP = 5

# The maximum number of features for which we can feasibly compute the exact Shapley values.
MAX_FEAS_EXACT_FEATS = 9


@dataclass
class ShapleyResults:
    """Contains the results of the LS-SPA algorithm."""

    attribution: np.ndarray
    theta: np.ndarray
    overall_error: float
    attribution_errors: np.ndarray
    r_squared: float
    error_history: np.ndarray | None
    attribution_history: np.ndarray | None

    def __repr__(self) -> str:
        """Makes printing the dataclass look nice."""
        attr_str = ""
        coefs_str = ""

        if len(self.attribution) <= MAX_ATTR_DISP:
            attr_str = "(" + "".join(f"{a:.2f}, " for a in self.attribution.flatten())[:-2] + ")"
            coefs_str = "(" + "".join(f"{c:.2f}, " for c in self.theta.flatten())[:-2] + ")"
        else:
            attr_str = (
                "("
                + "".join(f"{a:.2f}, " for a in self.attribution.flatten()[:MAX_ATTR_DISP])[:-2]
                + ", ...)"
            )
            coefs_str = (
                "("
                + "".join(f"{c:.2f}, " for c in self.theta.flatten()[:MAX_ATTR_DISP])[:-2]
                + ", ...)"
            )

        return f"""
        p = {len(self.attribution.flatten())}
        Out-of-sample R^2 with all features: {self.r_squared:.2f}

        Shapley attribution: {attr_str}
        Estimated error in Shapley attribution: {self.overall_error:.2E}

        Fitted coefficients with all features: {coefs_str}
        """


class SizeIncompatibleError(Exception):
    """Raised when the size of the data is incompatible with the function."""

    def __init__(self, message: str) -> None:
        """Initializes the SizeIncompatibleError."""
        self.message = message
        super().__init__(self.message)


# TODO(ndevanathan): remove this in the next major update
# This is here for backwards compatibility
SizeIncompatible = SizeIncompatibleError


def validate_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Validates the data. Raises an exception if the data is incompatible.

    Args:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The test data.
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The test labels.

    Raises:
        SizeIncompatibleError: If the data is incompatible.
    """
    if X_train.shape[1] != X_test.shape[1]:
        raise SizeIncompatibleError(
            "X_train and X_test should have the same number of columns (features)."
        )

    if X_train.shape[0] != y_train.shape[0]:
        raise SizeIncompatibleError(
            "X_train should have the same number of rows as y_train has entries (observations)."
        )

    if X_test.shape[0] != y_test.shape[0]:
        raise SizeIncompatibleError(
            "X_test should have the same number of rows as y_test has entries (observations)."
        )

    if X_train.shape[1] > X_train.shape[0]:
        raise SizeIncompatibleError(
            "The function works only if the number of "
            "features is at most the number of "
            "observations."
        )


def merge_sample_mean(
    old_mean: np.ndarray,
    new_mean: np.ndarray,
    old_N: int,
    new_N: int,
) -> np.ndarray:
    """Merges the means of two samples.

    Args:
        old_mean (np.ndarray): The old mean.
        new_mean (np.ndarray): The new mean.
        old_N (int): The number of old samples.
        new_N (int): The number of new samples.

    Returns:
        np.ndarray: The merged mean.
    """
    N = old_N + new_N
    adj_old_mean = (old_N / N) * old_mean
    adj_new_mean = (new_N / N) * new_mean
    return adj_old_mean + adj_new_mean


def merge_sample_cov(
    old_mean: np.ndarray,
    new_mean: np.ndarray,
    old_cov: np.ndarray,
    new_cov: np.ndarray,
    old_N: int,
    new_N: int,
) -> np.ndarray:
    """Merges the covariance matrices of two samples.

    Args:
        old_mean (np.ndarray): The old mean.
        new_mean (np.ndarray): The new mean.
        old_cov (np.ndarray): The old covariance matrix.
        new_cov (np.ndarray): The new covariance matrix.
        old_N (int): The number of old samples.
        new_N (int): The number of new samples.

    Returns:
        np.ndarray: The merged covariance matrix.
    """
    N = old_N + new_N
    mean_diff = old_mean - new_mean
    adj_old_cov = (old_N / N) * old_cov
    adj_new_cov = (new_N / N) * new_cov
    delta = (old_N / N) * (new_N / N) * np.outer(mean_diff, mean_diff)
    return adj_old_cov + adj_new_cov + delta


def ls_spa(
    X_train: np.ndarray | pd.DataFrame,
    X_test: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    y_test: np.ndarray | pd.Series,
    reg: float = 0.0,
    max_samples: int = 2**13,
    batch_size: int = 2**8,
    tolerance: float = 1e-2,
    seed: int = 42,
    perms: np.ndarray | None = None,
    antithetical: bool = True,
    return_attribution_history: bool = False,
) -> ShapleyResults:
    """Estimates the Shapley attribution for a least-squares problem.

    Args:
        X_train (np.ndarray | pd.DataFrame): The training data.
        X_test (np.ndarray | pd.DataFrame): The test data.
        y_train (np.ndarray | pd.Series): The training labels.
        y_test (np.ndarray | pd.Series): The test labels.
        reg (float): The regularization parameter.
        max_samples (int): The maximum number of samples.
        batch_size (int): The number of samples to use in each batch.
        tolerance (float): The tolerance for the stopping criterion.
        seed (int): The seed for the random number generator.
        perms (np.ndarray | None): The permutations to use. If None, the permutations are
            generated randomly.
        antithetical (bool): Whether to use antithetical sampling.
        return_attribution_history (bool): Whether to return the attribution history.

    Returns:
        A ShapleyResults object containing the Shapley attribution, the
        estimated error in the Shapley attribution, the fitted coefficients
        with all features, the out-of-sample R^2 with all features, and
        optionally the attribution history and the error history.
    """
    # Convert data into NumPy arrays.
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    validate_data(X_train, X_test, y_train, y_test)
    p = X_train.shape[1]

    # If perms were not passed or are invalid, then generate our own. XXX this
    # should throw an exception if invalid perms are passed, instead of
    # silently doing our own thing.
    rng = random.default_rng(seed)
    if perms is None:
        if p < MAX_FEAS_EXACT_FEATS:
            perms = it.permutations(range(p))
            batch_size = 2**8
            antithetical = False
        else:
            perms = (rng.permutation(p) for _ in range(max_samples))
    else:
        max_samples = 2**100

    # Compute the reduction
    y_test_norm_sq = np.linalg.norm(y_test) ** 2
    (X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde) = reduce_data(
        X_train,
        X_test,
        y_train,
        y_test,
        reg,
    )

    # Iterate over the permutations to compute lifts
    shapley_values = np.zeros(p)
    attribution_cov = np.zeros((p, p))
    attribution_errors = np.full(p, 0.0)
    overall_error = 0.0
    error_history = np.zeros(0)
    attribution_history = np.zeros((0, p)) if return_attribution_history else None

    i = 0
    for perm in perms:
        i += 1
        do_mini_batch = True

        # Compute the lift
        perm_np = np.array(perm)
        lift = square_shapley(
            X_train_tilde,
            X_test_tilde,
            y_train_tilde,
            y_test_tilde,
            y_test_norm_sq,
            perm_np,
        )
        if antithetical:
            lift = (
                lift
                + square_shapley(
                    X_train_tilde,
                    X_test_tilde,
                    y_train_tilde,
                    y_test_tilde,
                    y_test_norm_sq,
                    perm_np[::-1],
                )
            ) / 2

        # Update the mean and biased sample covariance
        attribution_cov = merge_sample_cov(
            shapley_values,
            lift,
            attribution_cov,
            np.zeros((p, p)),
            i - 1,
            1,
        )
        shapley_values = merge_sample_mean(shapley_values, lift, i - 1, 1)
        if return_attribution_history:
            attribution_history = np.vstack((attribution_history, shapley_values))

        # Update the errors
        if (i % batch_size == 0 or i == max_samples - 1) and p >= MAX_FEAS_EXACT_FEATS:
            unbiased_cov = attribution_cov * i / (i - 1)
            attribution_errors, overall_error = error_estimates(rng, unbiased_cov / i)
            error_history = np.append(error_history, overall_error)
            do_mini_batch = False

            # Check the stopping criterion
            if overall_error < tolerance:
                break

    # Last mini-batch
    if p >= MAX_FEAS_EXACT_FEATS and do_mini_batch:
        unbiased_cov = attribution_cov * i / (i - 1)
        attribution_errors, overall_error = error_estimates(rng, unbiased_cov / i)
        error_history = np.append(error_history, overall_error)

    # Compute auxiliary information
    theta = np.linalg.lstsq(X_train_tilde, y_train_tilde, rcond=None)[0]
    r_squared = (
        np.linalg.norm(y_test_tilde) ** 2 - np.linalg.norm(y_test_tilde - X_test_tilde @ theta) ** 2
    ) / y_test_norm_sq

    return ShapleyResults(
        attribution=shapley_values,
        theta=theta,
        overall_error=overall_error,
        attribution_errors=attribution_errors,
        r_squared=r_squared,
        error_history=error_history,
        attribution_history=attribution_history,
    )


def square_shapley(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_norm_sq: float,
    perm: np.ndarray,
) -> np.ndarray:
    """Computes the performance lift of each feature for a given permutation.

    Args:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The test data.
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The test labels.
        y_norm_sq (float): The squared norm of the test labels.
        perm (np.ndarray): The permutation to use.

    Returns:
        np.ndarray: The lift vector.
    """
    p, _ = X_train.shape
    Q, R = np.linalg.qr(X_train[:, perm])
    X = X_test[:, perm]

    Y = np.triu(Q.T @ np.tile(y_train, (p, 1)).T)
    T = sp.linalg.solve_triangular(R, Y)
    T = np.hstack((np.zeros((p, 1)), T))

    Y_test = np.tile(y_test, (p + 1, 1))
    costs = np.sum((X @ T - Y_test.T) ** 2, axis=0)
    R_sq = (np.linalg.norm(y_test) ** 2 - costs) / y_norm_sq
    L = np.ediff1d(R_sq)[np.argsort(perm)]

    return L


def reduce_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    reg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduces the data to a smaller problem.

    Args:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The test data.
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The test labels.
        reg (float): The regularization parameter.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The reduced data.
    """
    N, p = X_train.shape

    X_train = X_train / np.sqrt(N)
    X_train = np.vstack((X_train, np.sqrt(reg) * np.eye(p)))
    y_train = y_train / np.sqrt(N)
    y_train = np.concatenate((y_train, np.zeros(p)))

    Q, X_train_tilde = np.linalg.qr(X_train)
    Q_ts, X_test_tilde = np.linalg.qr(X_test)
    y_train_tilde = Q.T @ y_train
    y_test_tilde = Q_ts.T @ y_test
    return X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde


def error_estimates(rng: random.Generator, cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimates the error in the approximate Shapley attribution.

    Args:
        rng (random.Generator): The random number generator.
        cov (np.ndarray): The covariance matrix of the Shapley attribution.

    Returns:
        tuple[np.ndarray, float]: The estimated error.
    """
    p = cov.shape[0]
    try:
        sample_diffs = rng.multivariate_normal(np.zeros(p), cov, size=2**10, method="cholesky")
    except (np.linalg.LinAlgError, ValueError):
        sample_diffs = rng.multivariate_normal(np.zeros(p), cov, size=2**10, method="svd")
    abs_diffs = np.abs(sample_diffs)
    norms = np.linalg.norm(sample_diffs, axis=1)
    abs_quantile = np.quantile(abs_diffs, 0.95, axis=0)
    norms_quantile = np.quantile(norms, 0.95)
    return abs_quantile, norms_quantile
