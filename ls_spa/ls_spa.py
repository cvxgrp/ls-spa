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
from typing import Literal

import numpy as np
import pandas as pd
import scipy as sp
from joblib import Parallel, delayed
from numpy import random

from ls_spa.qmc import argsort_samples, permutohedron_samples

# The maximum number of features for which we can display the full attribution.
MAX_ATTR_DISP = 5

# The maximum number of features for which we can feasibly compute the exact Shapley values.
MAX_FEAS_EXACT_FEATS = 9

SAMPLERS = Literal["exact", "random", "argsort", "permutohedron"]
PERM_TYPE = tuple[np.ndarray, ...] | np.ndarray | SAMPLERS


@dataclass
class ShapleyResults:
    """Results from the LS-SPA (Least Squares Shapley Performance Attribution) algorithm.

    This dataclass contains the Shapley attribution values for each feature in a
    least-squares regression problem, along with auxiliary information about the
    fitted model and estimation quality.

    Attributes:
        attribution : np.ndarray
            The Shapley attribution values for each feature, shape (p,) where p is the
            number of features. The i-th entry represents feature i's contribution to
            the out-of-sample R^2. The sum of all attribution values equals the overall
            R^2 of the model with all features.
        theta : np.ndarray
            The fitted regression coefficients with all features included, shape (p,).
            These are the least-squares coefficients obtained from regressing y_train
            on X_train (with regularization if specified).
        overall_error : float
            The estimated error (95th percentile of L2 norm) in the Shapley attribution
            vector. This provides a confidence bound on the distance between the
            estimated and true Shapley values. Only meaningful when using approximate
            (not exact) computation.
        attribution_errors : np.ndarray
            The estimated error (95th percentile) for each individual feature's
            attribution, shape (p,). The i-th entry gives a confidence bound on how
            far the i-th attribution value is from its true value. Only meaningful
            when using approximate computation.
        r_squared : float
            The out-of-sample R^2 with all features included. This is the proportion
            of variance in y_test explained by the model fitted on X_train and y_train.
            Note: this can be negative if the model performs worse than the null model.
        error_history : np.ndarray | None
            The overall error estimate after each batch of permutations, shape (n_batches,).
            Useful for diagnosing convergence. None if exact computation was used or if
            only one batch was processed.
        attribution_history : np.ndarray | None
            The running estimate of the attribution values after each permutation,
            shape (n_samples, p). Only populated if `return_attribution_history=True`
            was passed to `ls_spa`, otherwise None. Useful for analyzing convergence
            behavior and creating diagnostic plots.

    Methods:
        __repr__ : str
            Returns a formatted summary of the results including the number of features,
            out-of-sample R^2, Shapley attribution values, estimation error, and fitted
            coefficients.

    Examples:
        After running LS-SPA, you can access the results as follows:

        >>> results = ls_spa(X_train, X_test, y_train, y_test)
        >>> results.attribution  # Shapley values for each feature
        array([0.15, 0.45, 0.22, 0.08])
        >>> results.r_squared  # Overall out-of-sample R^2
        0.90
        >>> results.overall_error  # Estimation error
        0.005
        >>> print(results)  # Pretty-printed summary

        You can also examine convergence if you requested the history:

        >>> results = ls_spa(X_train, X_test, y_train, y_test,
        ...                  return_attribution_history=True)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(results.attribution_history)
        >>> plt.xlabel('Permutation')
        >>> plt.ylabel('Attribution Value')
        >>> plt.title('Convergence of Shapley Attributions')
    """

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


def validate_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Validate that input data dimensions are compatible for LS-SPA computation.

    Checks that X_train and X_test have the same number of features, that X and y
    dimensions match within train/test sets, and that the number of features does
    not exceed the number of training samples (required for least-squares).

    Args:
        X_train (np.ndarray): Training feature matrix, shape (n_train, p).
        X_test (np.ndarray): Test feature matrix, shape (n_test, p).
        y_train (np.ndarray): Training target vector, shape (n_train,).
        y_test (np.ndarray): Test target vector, shape (n_test,).

    Raises:
        SizeIncompatibleError: If dimensions are incompatible.
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
    """Merge means from two samples using weighted averaging.

    Implements the online algorithm for combining sample means from two batches
    of data. Used to maintain running mean estimates during Monte Carlo sampling.

    Args:
        old_mean (np.ndarray): Mean of the first sample, shape (p,).
        new_mean (np.ndarray): Mean of the second sample, shape (p,).
        old_N (int): Number of observations in the first sample.
        new_N (int): Number of observations in the second sample.

    Returns:
        np.ndarray: Combined mean of both samples, shape (p,).
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
    """Merge covariance matrices from two samples using parallel algorithm.

    Implements the parallel algorithm for combining sample covariances from two
    batches of data. Accounts for the shift in means between batches. Used to
    maintain running covariance estimates during Monte Carlo sampling.

    Args:
        old_mean (np.ndarray): Mean of the first sample, shape (p,).
        new_mean (np.ndarray): Mean of the second sample, shape (p,).
        old_cov (np.ndarray): Covariance of the first sample, shape (p, p).
        new_cov (np.ndarray): Covariance of the second sample, shape (p, p).
        old_N (int): Number of observations in the first sample.
        new_N (int): Number of observations in the second sample.

    Returns:
        np.ndarray: Combined covariance of both samples, shape (p, p).
    """
    N = old_N + new_N
    mean_diff = old_mean - new_mean
    adj_old_cov = (old_N / N) * old_cov
    adj_new_cov = (new_N / N) * new_cov
    delta = (old_N / N) * (new_N / N) * np.outer(mean_diff, mean_diff)
    return adj_old_cov + adj_new_cov + delta


def process_perms(
    p: int, rng: random.Generator, max_samples: int, perms: PERM_TYPE | None
) -> np.ndarray:
    """Process and generate feature permutations based on the specified method.

    Handles different permutation generation strategies including exact enumeration,
    random sampling, and quasi-Monte Carlo methods. Validates that exact computation
    is only requested for feasible problem sizes.

    Args:
        p (int): Number of features.
        rng (random.Generator): NumPy random number generator for sampling.
        max_samples (int): Maximum number of permutations to generate.
        perms (PERM_TYPE | None): Permutation specification - either a string
            ("exact", "random", "argsort", "permutohedron"), a custom array of
            permutations, or None for automatic selection.

    Raises:
        ValueError: If exact permutations are requested for p >= 9.

    Returns:
        np.ndarray: Array or iterator of permutations, where each permutation
            is a 1D array of indices [0, 1, ..., p-1] in some order.
    """
    match perms:
        case "exact":
            if p < MAX_FEAS_EXACT_FEATS:
                return it.permutations(range(p))
            raise ValueError(
                f"Exact permutations are not available"
                f" for more than {MAX_FEAS_EXACT_FEATS} features."
            )
        case "random":
            return np.array([rng.permutation(p) for _ in range(max_samples)])
        case "argsort":
            return argsort_samples(p, max_samples, seed=rng.choice(1000))
        case "permutohedron":
            return permutohedron_samples(p, max_samples, seed=rng.choice(1000))
        case None:
            if p < MAX_FEAS_EXACT_FEATS:
                return it.permutations(range(p))
            return np.array([rng.permutation(p) for _ in range(max_samples)])
        case _:
            if isinstance(perms, (tuple, list)):
                return np.array(perms)
            return perms


def _compute_lift(
    perm: np.ndarray,
    X_train_tilde: np.ndarray,
    X_test_tilde: np.ndarray,
    y_train_tilde: np.ndarray,
    y_test_tilde: np.ndarray,
    y_test_norm_sq: float,
    antithetical: bool,
) -> np.ndarray:
    """Compute the Shapley lift vector for a single permutation.

    This is the core computation for each Monte Carlo sample. Evaluates the
    performance lift along a feature ordering and optionally applies antithetical
    sampling (averaging with the reversed permutation) for variance reduction.

    Args:
        perm: Feature permutation to evaluate, shape (p,).
        X_train_tilde: Reduced training feature matrix, shape (p, p).
        X_test_tilde: Reduced test feature matrix, shape (n_test, p).
        y_train_tilde: Reduced training target vector, shape (p,).
        y_test_tilde: Reduced test target vector, shape (n_test,).
        y_test_norm_sq: Squared L2 norm of the original test target vector.
        antithetical: Whether to apply antithetical sampling with reversed permutation.

    Returns:
        np.ndarray: Lift vector representing each feature's marginal contribution
            to R^2 along this ordering, shape (p,).
    """
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
    return lift


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
    perms: PERM_TYPE | None = None,
    antithetical: bool = True,
    return_attribution_history: bool = False,
    n_jobs: int = 1,
) -> ShapleyResults:
    r"""Estimate Shapley performance attribution for least-squares regression.

    This function computes Shapley values that attribute the out-of-sample R^2
    to individual features in a least-squares regression problem. The Shapley
    value for each feature represents its average marginal contribution to the
    model performance across all possible orderings of features.

    The implementation uses an efficient algorithm that avoids explicitly
    computing models for all feature subsets, making it feasible for problems
    with many features. For small numbers of features (< 9), exact Shapley values
    are computed. For larger problems, Monte Carlo estimation is used with
    adaptive stopping based on the error tolerance.

    Parameters:
        X_train : np.ndarray or pd.DataFrame
            Training feature matrix of shape (n_train, p) where n_train is the number
            of training samples and p is the number of features. If a DataFrame is
            provided, it will be converted to a NumPy array.
        X_test : np.ndarray or pd.DataFrame
            Test feature matrix of shape (n_test, p). Must have the same number of
            features as X_train. Used to compute out-of-sample performance.
        y_train : np.ndarray or pd.Series
            Training target vector of shape (n_train,). Used to fit the regression
            models.
        y_test : np.ndarray or pd.Series
            Test target vector of shape (n_test,). Used to compute out-of-sample
            performance (R^2) for each feature subset.
        reg : float, optional, default: 0.0
            Ridge regularization parameter (lambda). The regularization term added
            to the least-squares objective is `reg * ||theta||^2`. Use reg > 0 to
            improve numerical stability or when features are nearly collinear.
        max_samples : int, optional, default: 8192
            Maximum number of feature permutations to sample. Only used when p >= 9
            (otherwise exact computation is performed). Larger values give more
            accurate estimates but take longer. The algorithm may use fewer samples
            if the tolerance criterion is met earlier.
        batch_size : int, optional, default: 256
            Number of permutations to process before checking the stopping criterion.
            Smaller batches allow earlier termination but increase overhead from
            error estimation. Typical values are 128-512.
        tolerance : float, optional, default: 0.01
            Stopping criterion for the overall error estimate. The algorithm stops
            when the estimated 95th percentile L2 error in the Shapley attribution
            vector falls below this threshold. Only used for approximate (p >= 9)
            computation.
        seed : int, optional, default: 42
            Random seed for reproducibility. Controls the random number generator
            used for sampling permutations and estimating errors.
        perms : str or np.ndarray or tuple of np.ndarray or None, optional, default: None
            Specifies how to generate feature permutations. Options are:

            - None (default): Automatically selects "exact" if p < 9, otherwise "random"
            - "exact": Enumerate all p! permutations (only feasible for p < 9)
            - "random": Uniformly random permutations sampled with replacement
            - "argsort": Low-discrepancy quasi-Monte Carlo permutations using argsort
              of random uniforms. Often converges faster than "random".
            - "permutohedron": Low-discrepancy permutations from the permutohedron
              lattice. Can provide better coverage of the permutation space.
            - np.ndarray: Custom array of permutations, shape (n_perms, p), where
              each row is a permutation of [0, 1, ..., p-1]
            - tuple or list: Sequence of permutations (each a 1D array or list),
              will be converted to an array

            The "argsort" and "permutohedron" options implement quasi-Monte Carlo
            methods that can provide better convergence than "random" sampling.
        antithetical : bool, optional, default: True
            Whether to use antithetical (paired) sampling. For each permutation π,
            also evaluate the reversed permutation π[::-1] and average the results.
            This reduces variance and typically improves convergence. Automatically
            set to False when using exact computation.
        return_attribution_history : bool, optional, default: False
            Whether to track and return the attribution estimate after each permutation.
            If True, the `attribution_history` field in the returned ShapleyResults
            will contain an array of shape (n_samples, p) showing convergence. Useful
            for creating diagnostic plots but increases memory usage.
        n_jobs : int, optional, default: 1
            Number of parallel jobs for computing permutation samples. Use -1 to
            use all available CPU cores. Parallelization is most beneficial for
            larger batch sizes and when p is large. For small p or batch_size,
            overhead may outweigh benefits.

    Returns:
        ShapleyResults
            A dataclass containing the following fields:

            - `attribution` : Shapley values for each feature (shape (p,))
            - `theta` : Fitted regression coefficients with all features (shape (p,))
            - `overall_error` : Estimated L2 error in attribution vector (float)
            - `attribution_errors` : Estimated error for each feature (shape (p,))
            - `r_squared` : Out-of-sample R^2 with all features (float)
            - `error_history` : Error estimates after each batch (array or None)
            - `attribution_history` : Attribution estimates over time (array or None)

    Raises:
        SizeIncompatibleError
            If the input data dimensions are incompatible (e.g., X_train and X_test
            have different numbers of features, or X_train has more features than
            samples).
        ValueError
            If "exact" permutations are requested for p >= 9.

    See Also:
        ShapleyResults : The return type containing detailed results and usage examples.

    Notes:
        The Shapley value is defined as the average marginal contribution of a feature
        across all possible orderings of features. For feature i:

        .. math::
            \phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(p-|S|-1)!}{p!}
                      [R^2(S \cup \{i\}) - R^2(S)]

        where N is the set of all features, S is a subset not containing i, R^2(S)
        is the out-of-sample R^2 using only features in S, and p is the total number
        of features.

        The algorithm efficiently computes Shapley values by evaluating performance
        along random feature orderings (permutations) rather than enumerating all
        2^p feature subsets. This reduces complexity from O(2^p) to O(p^2 * K) where
        K is the number of permutations.

    References:
        .. [1] Bell, L., Devanathan, N., & Boyd, S. (2023). Efficient Shapley Performance
           Attribution for Least-Squares Regression. arXiv:2310.19245.

    Examples:
        Basic usage with synthetic data:

        >>> import numpy as np
        >>> from ls_spa import ls_spa
        >>>
        >>> # Generate synthetic data
        >>> np.random.seed(0)
        >>> n_train, n_test, p = 100, 50, 5
        >>> X_train = np.random.randn(n_train, p)
        >>> X_test = np.random.randn(n_test, p)
        >>> true_coef = np.array([2.0, -1.0, 0.5, 0.0, 1.5])
        >>> y_train = X_train @ true_coef + 0.1 * np.random.randn(n_train)
        >>> y_test = X_test @ true_coef + 0.1 * np.random.randn(n_test)
        >>>
        >>> # Compute Shapley attributions
        >>> results = ls_spa(X_train, X_test, y_train, y_test)
        >>> print(f"R^2: {results.r_squared:.3f}")
        >>> print(f"Attributions: {results.attribution}")

        Using ridge regularization:

        >>> results = ls_spa(X_train, X_test, y_train, y_test, reg=0.1)

        With many features, control the estimation accuracy:

        >>> n_train, n_test, p = 200, 100, 50
        >>> X_train = np.random.randn(n_train, p)
        >>> X_test = np.random.randn(n_test, p)
        >>> y_train = X_train @ np.random.randn(p) + 0.5 * np.random.randn(n_train)
        >>> y_test = X_test @ np.random.randn(p) + 0.5 * np.random.randn(n_test)
        >>>
        >>> # Use more samples for higher accuracy
        >>> results = ls_spa(X_train, X_test, y_train, y_test,
        ...                  max_samples=10000, tolerance=0.005)
        >>> print(f"Estimation error: {results.overall_error:.4f}")

        Using quasi-Monte Carlo for faster convergence:

        >>> results = ls_spa(X_train, X_test, y_train, y_test,
        ...                  perms="argsort", max_samples=5000)

        Visualizing convergence:

        >>> results = ls_spa(X_train, X_test, y_train, y_test,
        ...                  return_attribution_history=True)
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(10, 6))
        >>> for i in range(min(5, p)):  # Plot first 5 features
        ...     plt.plot(results.attribution_history[:, i], label=f'Feature {i}')
        >>> plt.xlabel('Permutation')
        >>> plt.ylabel('Attribution Value')
        >>> plt.legend()
        >>> plt.title('Convergence of Shapley Attributions')
        >>> plt.show()

        Using custom permutations:

        >>> # Provide specific permutations to evaluate
        >>> custom_perms = np.array([
        ...     [0, 1, 2, 3, 4],
        ...     [4, 3, 2, 1, 0],
        ...     [2, 0, 4, 1, 3]
        ... ])
        >>> results = ls_spa(X_train[:100, :5], X_test[:50, :5],
        ...                  y_train[:100], y_test[:50], perms=custom_perms)

        Parallel computation for large problems:

        >>> results = ls_spa(X_train, X_test, y_train, y_test,
        ...                  n_jobs=-1)  # Use all CPU cores

        Working with pandas DataFrames:

        >>> import pandas as pd
        >>> df_train = pd.DataFrame(X_train, columns=[f'feat_{i}' for i in range(p)])
        >>> df_test = pd.DataFrame(X_test, columns=[f'feat_{i}' for i in range(p)])
        >>> y_train_series = pd.Series(y_train, name='target')
        >>> y_test_series = pd.Series(y_test, name='target')
        >>> results = ls_spa(df_train, df_test, y_train_series, y_test_series)
    """
    # Convert data into NumPy arrays.
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    validate_data(X_train, X_test, y_train, y_test)
    p = X_train.shape[1]

    rng = random.default_rng(seed)
    if perms is None and p < MAX_FEAS_EXACT_FEATS:
        batch_size = 2**8
        antithetical = False

    perms = process_perms(p, rng, max_samples, perms)

    # Convert to list for batching (handles iterators like it.permutations)
    perms_list = list(perms)
    max_samples = len(perms_list)

    # Compute the reduction
    y_test_norm_sq = np.linalg.norm(y_test) ** 2
    (X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde) = reduce_data(
        X_train,
        X_test,
        y_train,
        y_test,
        reg,
    )

    # Initialize accumulators for the Shapley attribution
    shapley_values = np.zeros(p)
    attribution_cov = np.zeros((p, p))
    attribution_errors = np.full(p, 0.0)
    overall_error = 0.0
    error_history = np.zeros(0)
    attribution_history = np.zeros((0, p)) if return_attribution_history else None

    # Iterate over permutations in batches
    i = 0
    for batch_start in range(0, max_samples, batch_size):
        batch_end = min(batch_start + batch_size, max_samples)
        batch_perms = perms_list[batch_start:batch_end]

        # Compute lifts for the batch (parallel or sequential)
        if n_jobs == 1:
            lifts = [
                _compute_lift(
                    perm,
                    X_train_tilde,
                    X_test_tilde,
                    y_train_tilde,
                    y_test_tilde,
                    y_test_norm_sq,
                    antithetical,
                )
                for perm in batch_perms
            ]
        else:
            lifts = Parallel(n_jobs=n_jobs)(
                delayed(_compute_lift)(
                    perm,
                    X_train_tilde,
                    X_test_tilde,
                    y_train_tilde,
                    y_test_tilde,
                    y_test_norm_sq,
                    antithetical,
                )
                for perm in batch_perms
            )

        # Aggregate lifts sequentially (updates running mean and covariance)
        for lift in lifts:
            i += 1
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

        # Update errors after each batch
        if p >= MAX_FEAS_EXACT_FEATS and i > 1:
            unbiased_cov = attribution_cov * i / (i - 1)
            attribution_errors, overall_error = error_estimates(rng, unbiased_cov / i)
            error_history = np.append(error_history, overall_error)

            # Check the stopping criterion
            if overall_error < tolerance:
                break

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
    """Compute marginal R^2 contributions for each feature along a given ordering.

    This is the core algorithmic routine that efficiently computes the performance
    lift (change in out-of-sample R^2) when adding each feature in the specified
    order. Uses QR decomposition to avoid repeatedly solving least-squares problems.

    Args:
        X_train (np.ndarray): Training feature matrix, shape (p, p).
        X_test (np.ndarray): Test feature matrix, shape (n_test, p).
        y_train (np.ndarray): Training target vector, shape (p,).
        y_test (np.ndarray): Test target vector, shape (n_test,).
        y_norm_sq (float): Squared L2 norm of y_test.
        perm (np.ndarray): Feature ordering/permutation to evaluate, shape (p,).

    Returns:
        np.ndarray: Lift vector where entry i is the R^2 gain from adding feature i
            (in its original index) to the feature subset that precedes it in the
            permutation, shape (p,).
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
    """Reduce the regression problem to canonical form via QR decomposition.

    Applies ridge regularization (if specified) and performs QR decompositions
    to reduce the problem to a smaller, equivalent form. This preprocessing step
    significantly speeds up the repeated least-squares computations required for
    Shapley value estimation.

    Args:
        X_train (np.ndarray): Training feature matrix, shape (n_train, p).
        X_test (np.ndarray): Test feature matrix, shape (n_test, p).
        y_train (np.ndarray): Training target vector, shape (n_train,).
        y_test (np.ndarray): Test target vector, shape (n_test,).
        reg (float): Ridge regularization parameter (lambda).

    Returns:
        tuple: (X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde) where
            X_train_tilde has shape (p, p), X_test_tilde has shape (n_test, p),
            y_train_tilde has shape (p,), and y_test_tilde has shape (n_test,).
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
    """Estimate confidence bounds on Shapley attribution errors via sampling.

    Uses the estimated covariance matrix of the Shapley values to generate samples
    from the approximate distribution of estimation errors. Computes 95th percentile
    bounds for both individual feature attributions and the overall L2 error.

    Args:
        rng (random.Generator): NumPy random number generator.
        cov (np.ndarray): Estimated covariance matrix of Shapley values, shape (p, p).

    Returns:
        tuple: (attribution_errors, overall_error) where attribution_errors is an
            array of per-feature error estimates (shape (p,)) and overall_error is
            the 95th percentile L2 norm of the error vector (float).
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
