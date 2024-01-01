import marimo

__generated_with = "0.1.64"
app = marimo.App(width="full")


@app.cell
def __(attrs):
    # Copyright 2023 Logan Bell, Nikhil Devanathan, and Stephen Boyd

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
    This module contains a JAX implementation of a method to efficiently 
    estimate a Shapley attribution for least squares problems.

    This method is described in the paper Efficient Shapley Performance 
    Attribution for Least-Squares Regression (arXiv:2310.19245) by Logan 
    Bell, Nikhil Devanathan, and Stephen Boyd.
    """

    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from functools import partial
    from typing import Literal, Tuple

    import jax
    import numpy as np
    import jax.numpy as jnp
    import jax.scipy as jsp
    import jax.lax as lax
    import jax.ops as ops
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
            attrs_str = ""
            coefs_str = ""

            if len(attrs) <= 5:
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


    def validate_data(X_train: jnp.ndarray,
                      X_test: jnp.ndarray,
                      y_train: jnp.ndarray,
                      y_test: jnp.ndarray):
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


    def ls_spa(X_train: np.ndarray | jnp.ndarray | pd.DataFrame,
               X_test: np.ndarray | jnp.ndarray | pd.DataFrame,
               y_train: np.ndarray | jnp.ndarray | pd.Series,
               y_test: np.ndarray | jnp.ndarray | pd.Series,
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
            num_batches: The number of batches of permutations to use.
            batch_size: The size of each batch of permutations.
            tolerance: The tolerance for the stopping criterion.
            seed: The seed for the random number generator.
            perms: The permutations to use.

        Returns:
            The Shapley attribution.
        """

        # Convert data into JAX arrays.
        X_train = jnp.array(X_train)
        X_test = jnp.array(X_test)
        y_train = jnp.array(y_train)
        y_test = jnp.array(y_test)
        validate_data(X_train, X_test, y_train, y_test)
        p = X_train.shape[1]

        rng = random.default_rng(seed)

        if perms is None or perms.ndim != 2 or perms.shape[1] != p or len(perms) % batch_size != 0:
            if p < 9:
                perms = np.array(list(it.permutations(range(p))))
                num_batches = 1
                batch_size = len(perms)
            else:
                perms = np.array([rng.permutation(p) for _ in range(num_batches * batch_size)])

        # Compute the reduction
        y_test_norm_sq = jnp.linalg.norm(y_test) ** 2
        X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde = reduce_data(X_train, X_test, y_train, y_test, reg)
        theta = jnp.linalg.lstsq(X_train_tilde, y_train_tilde, rcond=None)[0]
        r_squared = (jnp.linalg.norm(y_test_tilde) ** 2 - jnp.linalg.norm(y_test_tilde - X_test_tilde @ theta) ** 2) / y_test_norm_sq

        # Initialize the Shapley values and covariance
        shapley_values = jnp.zeros(p)
        attribution_cov = jnp.zeros((p, p))

        # Iterate over the batches of permutations
        for i in range(num_batches):
            perm_batch = perms[i * batch_size:(i + 1) * batch_size]

            # Compute the lifts for each permutation in the batch in parallel
            lifts = jax.pmap(partial(square_shapley, X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde, y_test_norm_sq))(perm_batch)

            # Update the Shapley values and covariance
            shapley_values = shapley_values * (i * batch_size) / ((i + 1) * batch_size) + jnp.sum(lifts, axis=0) / ((i + 1) * batch_size)
            deviation = lifts - shapley_values
            attribution_cov = attribution_cov * (i * batch_size) / ((i + 1) * batch_size) + jnp.sum(jnp.outer(d, d) for d in deviation) / ((i + 1) * batch_size)

            # Estimating errors
            if ((i + 1) % batch_size == 0) or (i + 1 == num_batches):
                unbiased_cov = attribution_cov * ((i + 1) * batch_size) / (i * batch_size)
                attribution_errors, overall_error = error_estimates(rng, unbiased_cov / ((i + 1) * batch_size))

                # Check the stopping criterion
                if overall_error < tolerance:
                    break

        return ShapleyResults(
            attribution=shapley_values,
            theta=theta,
            overall_error=overall_error,
            error_history=None,
            attribution_errors=attribution_errors,
            r_squared=r_squared
        )


    @partial(jax.jit, static_argnums=(2, 3, 4))
    def square_shapley(X_train: jnp.ndarray, X_test: jnp.ndarray,
                       y_train: jnp.ndarray, y_test: jnp.ndarray,
                       y_norm_sq: float, perm: jnp.ndarray) -> jnp.ndarray:
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
        Q, R = jnp.linalg.qr(X_train[:, perm])
        X = X_test[:, perm]

        Y = jnp.triu(Q.T @ jnp.tile(y_train, (p, 1)).T)
        T = jsp.linalg.solve_triangular(R, Y)
        T = jnp.hstack((np.zeros((p, 1)), T))

        Y_test = jnp.tile(y_test, (p+1, 1))
        costs = jnp.sum((X @ T - Y_test.T) ** 2, axis=0)
        R_sq = (y_norm_sq - costs) / y_norm_sq
        L = jnp.ediff1d(R_sq)[jnp.argsort(perm)]

        return L


    @jax.jit
    def reduce_data(X_train: jnp.ndarray, X_test: jnp.ndarray,
                    y_train: jnp.ndarray, y_test: jnp.ndarray, 
                    reg: float):
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

        X_train = X_train / jnp.sqrt(N)
        X_train = jnp.vstack((X_train, jnp.sqrt(reg) * jnp.eye(p)))
        y_train = y_train / jnp.sqrt(N)
        y_train = jnp.concatenate((y_train, jnp.zeros(p)))

        Q, X_train_tilde = jnp.linalg.qr(X_train)
        Q_ts, X_test_tilde = jnp.linalg.qr(X_test)
        y_train_tilde = Q.T @ y_train
        y_test_tilde = Q_ts.T @ y_test
        return X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde


    @partial(jax.jit, static_argnums=(0,))
    def error_estimates(rng, cov: jnp.ndarray):
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
            sample_diffs = rng.multivariate_normal(jnp.zeros(p), cov, size=2 ** 10, method="cholesky")
        except:
            sample_diffs = rng.multivariate_normal(jnp.zeros(p), cov, size=2 ** 10, method="svd")
        abs_diffs = jnp.abs(sample_diffs)
        norms = jnp.linalg.norm(sample_diffs, axis=1)
        abs_quantile = jnp.quantile(abs_diffs, 0.95, axis=0)
        norms_quantile = jnp.quantile(norms, 0.95)
        return abs_quantile, norms_quantile
    return (
        ABC,
        Literal,
        ShapleyResults,
        SizeIncompatible,
        Tuple,
        abstractmethod,
        dataclass,
        error_estimates,
        it,
        jax,
        jnp,
        jsp,
        lax,
        ls_spa,
        np,
        ops,
        partial,
        pd,
        random,
        reduce_data,
        sp,
        square_shapley,
        validate_data,
    )


@app.cell
def __(ls_spa, np):
    p, N, M, STN_RATIO, REG = 500, int(1e5), int(1e5), 5., 0.
    rng = np.random.default_rng(364)
    A = rng.normal(size=(p, p//20))
    cov = A @ A.T + np.eye(p)
    v = np.sqrt(np.diag(cov))
    cov = cov / np.outer(v, v)

    X_train = rng.multivariate_normal(np.zeros(p), cov, size=N, method="svd")
    X_test = rng.multivariate_normal(np.zeros(p), cov, size=M, method="svd")

    theta_vals = np.full((p+1)//2, 2)
    padded_theta_vals = np.pad(theta_vals, (0, p - (p+1)//2))
    theta_true = rng.permutation(padded_theta_vals)

    std = np.sqrt(np.sum(np.diag(cov) * theta_true**2) / STN_RATIO)
    y_train = X_train @ theta_true + std * rng.normal(size=N)
    X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    X_train = X_train - X_train_mean
    y_train_mean = np.mean(y_train)
    y_train = y_train - y_train_mean

    y_test = X_test @ theta_true + std * rng.normal(size=M)
    X_test = X_test - X_train_mean
    y_test = y_test - y_train_mean

    results = ls_spa(X_train, X_test, y_train, y_test, reg=REG, num_iters=2**14, tolerance=1e-2, seed=42)
    return (
        A,
        M,
        N,
        REG,
        STN_RATIO,
        X_test,
        X_train,
        X_train_mean,
        cov,
        p,
        padded_theta_vals,
        results,
        rng,
        std,
        theta_true,
        theta_vals,
        v,
        y_test,
        y_train,
        y_train_mean,
    )


@app.cell
def __(results):
    results
    return


if __name__ == "__main__":
    app.run()
