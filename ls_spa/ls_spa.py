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

"""
This module contains a method to efficiently estimate a Shapley
attribution for least squares problems.

This method is described in the paper Efficient Shapley Performance
Attribution for Least-Squares Regression (arXiv:2310.19245) by Logan
Bell, Nikhil Devanathan, and Stephen Boyd.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy as sp
from numpy import random
import pandas as pd
import itertools as it

@dataclass
class ShapleyResults:
    attribution: np.ndarray
    theta: np.ndarray
    overall_error: float
    error_history: np.ndarray | None
    attribution_errors: np.ndarray
    r_squared: float

    def __repr__(self):
        """Makes printing the dataclass look nice."""
        attr_str = ""
        coefs_str = ""

        if len(self.attribution) <= 5:
            attr_str = "(" + "".join("{:.2f}, ".format(a) for a in self.attribution.flatten())[:-2] + ")"
            coefs_str = "(" + "".join("{:.2f}, ".format(c) for c in self.theta.flatten())[:-2] + ")"
        else:
            attr_str = "(" + "".join("{:.2f}, ".format(a) for a in self.attribution.flatten()[:5])[:-2] + ", ...)"
            coefs_str = "(" + "".join("{:.2f}, ".format(c) for c in self.theta.flatten()[:5])[:-2] + ", ...)"

        return """
        p = {}
        Out-of-sample R^2 with all features: {:.2f}

        Shapley attribution: {}
        Estimated error in Shapley attribution: {:.2E}

        Fitted coeficients with all features: {}
        """.format(
            len(self.attribution.flatten()),
            self.r_squared,
            attr_str,
            self.overall_error,
            coefs_str
        )


class SizeIncompatible(Exception):
    """Raised when the size of the data is incompatible with the function."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def validate_data(X_train: np.ndarray,
                  X_test: np.ndarray,
                  y_train: np.ndarray,
                  y_test: np.ndarray):
    if X_train.shape[1] != X_test.shape[1]:
        raise SizeIncompatible("X_train and X_test should have the "
                               "same number of columns (features).")

    if X_train.shape[0] != y_train.shape[0]:
        raise SizeIncompatible("X_train should have the same number of "
                               "rows as y_train has entries (observations).")

    if X_test.shape[0] != y_test.shape[0]:
        raise SizeIncompatible("X_test should have the same number of "
                               "rows as y_test has entries (observations).")

    if X_train.shape[1] > X_train.shape[0]:
        raise SizeIncompatible("The function works only if the number of "
                               "features is at most the number of "
                               "observations.")


def ls_spa(X_train: np.ndarray | pd.DataFrame,
           X_test: np.ndarray | pd.DataFrame,
           y_train: np.ndarray | pd.Series,
           y_test: np.ndarray | pd.Series,
           reg: float = 0.,
           num_batches: int = 2 ** 6,
           batch_size: int = 2 ** 8,
           tolerance: float = 1e-2,
           seed: int = 42,
           perms: np.ndarray | None = None) -> ShapleyResults:
    """
    Estimates the Shapley attribution for a least squares problem.

    Args:
        X_train: The training data.
        X_test: The test data.
        y_train: The training labels.
        y_test: The test labels.
        reg: The regularization parameter.
        batch_size: The number of samples to use in each batch.
        num_batches: The number of batches to use.
        tolerance: The tolerance for the stopping criterion.
        seed: The seed for the random number generator.
        perms: The permutations to use. If None, the permutations are
            generated randomly.

    Returns:
        A ShapleyResults object containing the Shapley attribution, the
        estimated error in the Shapley attribution, the fitted coefficients
        with all features, the out-of-sample R^2 with all features, and
        optionally the attribution history and the error history.
    """

    # Converting data into NumPy arrays.
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    validate_data(X_train, X_test, y_train, y_test)
    p = X_train.shape[1]

    rng = random.default_rng(seed)

    if perms is None or perms.ndim != 2 or perms.shape[1] != p or len(perms) % batch_size != 0:
        if p < 9:
            perms = it.permutations(range(p))
            num_batches = 1
            batch_size = 2 ** 8
        else:
            perms = (rng.permutation(p) for _ in range(num_batches * batch_size))
    else:
        num_batches = len(perms) // batch_size

    # Compute the reduction
    y_test_norm_sq = np.linalg.norm(y_test) ** 2
    X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde = reduce_data(X_train, X_test, y_train, y_test, reg)
    theta = np.linalg.lstsq(X_train_tilde, y_train_tilde, rcond=None)[0]
    r_squared = (np.linalg.norm(y_test_tilde) ** 2 - np.linalg.norm(y_test_tilde - X_test_tilde @ theta) ** 2) / y_test_norm_sq

    # Iterate over the permutations to compute lifts
    shapley_values = np.zeros(p)
    attribution_cov = np.zeros((p, p))
    attribution_errors = np.full(p, 0.)
    overall_error = 0.
    error_history = np.zeros(num_batches)

    for i, perm in enumerate(perms, 1):
        # Compute the lift
        perm = np.array(perm)
        lift = square_shapley(X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde, y_test_norm_sq, perm)

        # Update the mean and biased sample covariance
        shapley_values = (i - 1) / i * shapley_values + lift / i
        deviation = lift - shapley_values
        attribution_cov = (i - 1) / i * (attribution_cov + np.outer(deviation, deviation) / i)

        # Update the errors
        if ((i % batch_size == 0) or (i == num_batches * batch_size)) and p >= 9:
            unbiased_cov = attribution_cov * i / (i - 1)
            attribution_errors, overall_error = error_estimates(rng, unbiased_cov / i)
            error_history[i // batch_size - 1] = overall_error

            # Check the stopping criterion
            if overall_error < tolerance:
                break

    return ShapleyResults(
        attribution=shapley_values,
        theta=theta,
        overall_error=overall_error,
        error_history=error_history,
        attribution_errors=attribution_errors,
        r_squared=r_squared
    )


def square_shapley(X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray,
                   y_norm_sq: float, perm: np.ndarray) -> np.ndarray:
    """
    Estimates the Shapley attribution for a least squares problem.

    Args:
        X_train: The training data.
        X_test: The test data.
        y_train: The training labels.
        y_test: The test labels.
        y_norm_sq: The squared norm of the test labels.
        perms: The permutations to use.

    Returns:
        The lift vector.
    """

    p, _ = X_train.shape
    Q, R = np.linalg.qr(X_train[:, perm])
    X = X_test[:, perm]

    Y = np.triu(Q.T @ np.tile(y_train, (p, 1)).T)
    T = sp.linalg.solve_triangular(R, Y)
    T = np.hstack((np.zeros((p, 1)), T))

    Y_test = np.tile(y_test, (p+1, 1))
    costs = np.sum((X @ T - Y_test.T) ** 2, axis=0)
    R_sq = (np.linalg.norm(y_test) ** 2 - costs) / y_norm_sq
    L = np.ediff1d(R_sq)[np.argsort(perm)]

    return L


def reduce_data(X_train: np.ndarray, X_test: np.ndarray,
                y_train: np.ndarray, y_test: np.ndarray,
                reg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduces the data to a smaller problem.

    Args:
        X_train: The training data.
        X_test: The test data.
        y_train: The training labels.
        y_test: The test labels.
        reg: The regularization parameter.

    Returns:
        The reduced data.
    """
    N, p = X_train.shape
    M, _ = X_test.shape

    X_train = X_train / np.sqrt(N)
    X_train = np.vstack((X_train, np.sqrt(reg) * np.eye(p)))
    y_train = y_train / np.sqrt(N)
    y_train = np.concatenate((y_train, np.zeros(p)))

    Q, X_train_tilde = np.linalg.qr(X_train)
    Q_ts, X_test_tilde = np.linalg.qr(X_test)
    y_train_tilde = Q.T @ y_train
    y_test_tilde = Q_ts.T @ y_test
    return X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde


def error_estimates(rng: float, cov: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Estimates the error in the Shapley attribution.

    Args:
        rng: The random number generator.
        cov: The covariance matrix of the Shapley attribution.

    Returns:
        The estimated error in the Shapley attribution.
    """
    p = cov.shape[0]
    try:
        sample_diffs = rng.multivariate_normal(np.zeros(p), cov, size=2 ** 10, method="cholesky")
    except:
        sample_diffs = rng.multivariate_normal(np.zeros(p), cov, size=2 ** 10, method="svd")
    abs_diffs = np.abs(sample_diffs)
    norms = np.linalg.norm(sample_diffs, axis=1)
    abs_quantile = np.quantile(abs_diffs, 0.95, axis=0)
    norms_quantile = np.quantile(norms, 0.95)
    return abs_quantile, norms_quantile
