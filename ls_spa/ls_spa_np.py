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

"""This modeule contains a method to efficiently estimate a Shapley 
attribution for least squares problems.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Literal, Tuple

import numpy as np
import scipy as sp
from numpy import random
import pandas as pd
from scipy.stats.qmc import MultivariateNormalQMC, Sobol

SampleMethod = Literal['random', 'permutohedron', 'argsort', 'exact']

@dataclass
class ShapleyResults:
    attribution: np.ndarray
    attribution_history: np.ndarray | None
    theta: np.ndarray
    overall_error: float
    error_history: np.ndarray | None
    attribution_errors: np.ndarray
    r_squared: float

    def __repr__(self):
        """Makes printing the dataclass look nice."""
        attr_str = "(" + "".join("{:.2f}, ".format(a) for a in self.attribution.flatten())[:-2] + ")"
        coefs_str = "(" + "".join("{:.2f}, ".format(c) for c in self.theta.flatten())[:-2] + ")"

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
    """Custom exception for incompatible data sizes."""
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
           method: SampleMethod | None = None,
           batch_size: int = 2 ** 7,
           num_batches: int = 2 ** 7,
           tolerance: float = 1e-2,
           seed: int = 42) -> ShapleyResults:
    # Converting data into NumPy arrays.
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    validate_data(X_train, X_test, y_train, y_test)

    N, p = X_train.shape
    M, _ = X_test.shape
    if method is None:
        if p > 10:
            method = 'argsort'
        else:
            method = 'argsort' ### XXX exact needs to be implemented still

    rng = random.default_rng(seed)
    compute_spa = LSSPA(rng=rng,
                       p=p,
                       sample_method=method,
                       batch_size=batch_size)

    return compute_spa(X_train=X_train,
                       X_test=X_test,
                       y_train=y_train,
                       y_test=y_test,
                       reg=reg,
                       max_num_batches=num_batches,
                       eps=tolerance,
                       return_history=False)


class Permutations(ABC):

    def __init__(self, rng, p: int):
        self.rng = rng
        self.p = p

    @abstractmethod
    def __call__(self, num_perms: int) -> np.ndarray:
        pass

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        self._p = new_p


class RandomPermutations(Permutations):

    def __call__(self, num_perms: int) -> np.ndarray:
        to_permute = np.tile(np.arange(0, self.p), (num_perms, 1))
        # Generate random permutations
        return rng.permutation(to_permute, axis=1, independent=True)


class PermutohedronPermutations(Permutations):

    def __init__(self, rng, p: int):
        self.rng = rng
        self.p = p

    def __call__(self, num_perms: int) -> np.ndarray:
        # Generate permutohedron samples
        samples = np.array(self.qmc.random(num_perms))
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        samples = self.project(samples)
        samples = np.argsort(samples, axis=1)
        return samples

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        self._p = new_p
        seed = int(self.rng.randint(0, 100000))
        self.qmc = MultivariateNormalQMC(np.zeros(self.p-1), seed=seed,
                                         inv_transform=False)

    def project(self, x: np.ndarray):
        tril_part = np.tril(np.ones((self.p-1, self.p)))
        diag_part = np.diag(-np.arange(1, self.p), 1)[:-1]
        U = tril_part + diag_part
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        return x @ U


class ArgsortPermutations(Permutations):

    def __init__(self, rng, p: int):
        self.rng = rng
        self.p = p

    def __call__(self, num_perms: int) -> np.ndarray:
        # Generate argsort samples
        samples = np.array(self.qmc.random(num_perms))
        return np.argsort(samples, axis=1)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        self._p = new_p
        seed = int(self.rng.randint(0, 100000))
        self.qmc = Sobol(self.p, seed=seed)


class RiskEstimate:

    def __init__(self, rng, batch_size: int, p: int):
        self.rng = rng
        self.batch_size = batch_size
        self.p = p
        self.mean = np.zeros((self.p,))
        self.cov = np.zeros((self.p, self.p))
        self._i = 1

        def risk_sample(rng, cov):
            sample_diffs = rng.multivariate_normal(np.zeros(p),
                                                      cov, size=((1000,)))

            abs_diffs = np.abs(sample_diffs)
            norms = np.linalg.norm(sample_diffs, axis=1)
            abs_quantile = np.quantile(abs_diffs, 0.95, axis=0)
            norms_quantile = np.quantile(norms, 0.95)
            return abs_quantile, norms_quantile

        self.risk_sample = risk_sample

    def __call__(self, batch: np.ndarray) -> float:
        self.mean, self.cov = self._call_helper(self._i, self.mean,
                                                self.cov, batch)
        num_pts = self.batch_size * self._i
        unbiased_cov = num_pts / (num_pts - 1) * self.cov
        feature_err, global_err = self.risk_sample(self.rng,
                                                   unbiased_cov/num_pts)
        self._i += 1
        return feature_err, global_err

    @staticmethod
    def _call_helper(i: int, mean: np.ndarray, cov: np.ndarray,
                     batch: np.ndarray) -> np.ndarray:
        batch_mean = np.mean(batch, axis=0)
        batch_cov = np.cov(batch.T, bias=True)

        mean_diff = mean - batch_mean
        correction_term = (i-1) / i**2 * np.outer(mean_diff, mean_diff)
        new_mean = (i-1) / i * mean + batch_mean / i
        new_cov = (i-1) / i * cov + batch_cov / i + correction_term
        return new_mean, new_cov

    def reset(self):
        self.mean = np.zeros((self.p,))
        self.cov = np.zeros((self.p, self.p))
        self._i = 1


class SquareShapley:

    def __init__(self, p: int):
        self.p = p

    def __call__(self, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray,
                 y_norm_sq: np.ndarray, perms: np.ndarray) -> np.ndarray:
        return self.square_shapley(X_train, X_test, y_train, y_test,
                                   y_norm_sq, perms)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        self._p = new_p
        def square_shapley(X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray,
                           y_norm_sq: np.ndarray,
                           perms: np.ndarray) -> np.ndarray:
            Q, R = np.linalg.qr(X_train[:, perms])
            X = X_test[:, perms]

            Y = np.triu(Q.T @ np.tile(y_train, (self.p, 1)).T)
            T = sp.linalg.solve_triangular(R, Y)
            T = np.hstack((np.zeros((self.p, 1)), T))

            Y_test = np.tile(y_test, (self.p+1, 1))
            costs = np.sum((X @ T - Y_test.T) ** 2, axis=0)
            R_sq = (np.linalg.norm(y_test) ** 2 - costs) / y_norm_sq
            perm_scores = np.ediff1d(R_sq)[np.argsort(perms)]
            return perm_scores


class LSSPA:

    def __init__(self, rng, p: int = 10,
                 sample_method: SampleMethod = 'random',
                 batch_size: int = 2**13):
        self._p = p
        self.rng = rng
        self.sample_method = sample_method
        # Initialize appropriate permutation generator based on sampling method
        if self.sample_method == 'random':
            self.sampler = RandomPermutations(rng, self.p)
        elif self.sample_method == 'permutohedron':
            self.sampler = PermutohedronPermutations(rng, self.p)
        else:
            self.sampler = ArgsortPermutations(rng, self.p)

        self.batch_size = batch_size
        self._square_shapley = SquareShapley(p)
        self.risk_estimate = RiskEstimate(rng, self.batch_size, self.p)

    def __call__(self, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray,
                 reg: float, max_num_batches: int = 1,
                 eps: float = 1e-3, y_norm_sq: np.ndarray | None = None,
                 return_history: bool = True) -> ShapleyResults:

        if y_norm_sq is None:
            N = 1 if np.isclose(reg, 0) else len(X_train)
            M = len(X_test)
            X_train, X_test, y_train, y_test, y_norm_sq = (
                self.process_data(N, M, X_train, X_test, y_train, y_test, reg))
        theta = np.linalg.lstsq(X_train, y_train)[0]

        r_squared = (np.linalg.norm(y_test) ** 2 - 
                    np.linalg.norm(X_test @ theta - y_test) ** 2) / y_norm_sq

        attribution_history = np.zeros((0, self.p)) if return_history else None
        scores = np.zeros(self.p)
        error_history = np.zeros((0,)) if return_history else None
        self.risk_estimate.reset()

        for i in range(1, max_num_batches+1):
            batch = self.sampler(self.batch_size)
            perm_scores = self._square_shapley(X_train, X_test, y_train,
                                               y_test, y_norm_sq, batch)
            scores = (i-1)/i * scores + np.mean(perm_scores, axis=0) / i
            feature_risk, global_risk = self.risk_estimate(perm_scores)
            if return_history:
                attribution_history = np.vstack((attribution_history,
                                                  perm_scores))
                error_history = np.append(
                    error_history, global_risk
                )
            if global_risk < eps:
                break

        results = ShapleyResults(attribution=scores,
                                 attribution_history=attribution_history,
                                 theta=theta,
                                 overall_error=global_risk,
                                 error_history=error_history,
                                 attribution_errors=feature_risk,
                                 r_squared=r_squared)
        return results

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        if self.p == new_p:
            return

        # Initialize appropriate permutation generator based on sampling method
        if self.sample_method == 'random':
            self.sampler = RandomPermutations(self.rng, self.p)
        elif self.sample_method == 'permutohedron':
            self.sampler = PermutohedronPermutations(self.rng, self.p)
        else:
            self.sampler = ArgsortPermutations(self.rng, self.p)

        self._square_shapley = SquareShapley(self.p)
        self.risk_estimate = RiskEstimate(self.rng, self.batch_size, self.p)

    def process_data(self, N: int, M: int, X_train: np.ndarray,
                     X_test: np.ndarray, y_train: np.ndarray,
                     y_test: np.ndarray,
                     reg: float) -> Tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray,
                                          np.ndarray]:
        X_train = X_train / np.sqrt(N)
        X_train = np.vstack((X_train, np.sqrt(reg) * np.eye(self.p)))
        y_train = y_train / np.sqrt(N)
        y_train = np.concatenate((y_train, np.zeros(self.p)))

        y_norm_sq = np.linalg.norm(y_test) ** 2

        Q, X_train, = np.linalg.qr(X_train)
        Q_ts, X_test = np.linalg.qr(X_test)
        y_train = Q.T @ y_train
        y_test = Q_ts.T @ y_test
        return X_train, X_test, y_train, y_test, y_norm_sq