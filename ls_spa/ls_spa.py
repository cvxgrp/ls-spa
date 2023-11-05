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

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import jit, random, vmap
import pandas as pd
from scipy.stats.qmc import MultivariateNormalQMC, Sobol

SampleMethod = Literal['random', 'permutohedron', 'argsort', 'exact']

@dataclass
class ShapleyResults:
    """_summary_

    Attributes:
        attribution (jnp.ndarray): _description_
        attribution_history (jnp.ndarray | None): _description_
        theta (jnp.ndarray): _description_
        overall_error (float): _description_
        error_history (jnp.ndarray | None): _description_
        attribution_errors (jnp.ndarray): _description_
        r_squared (float): _description_
    """
    attribution: jnp.ndarray
    attribution_history: jnp.ndarray | None
    theta: jnp.ndarray
    overall_error: float
    error_history: jnp.ndarray | None
    attribution_errors: jnp.ndarray
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


def validate_data(X_train: jnp.ndarray,
                  X_test: jnp.ndarray,
                  y_train: jnp.ndarray,
                  y_test: jnp.ndarray):
    """_summary_

    Args:
        X_train (jnp.ndarray): _description_
        X_test (jnp.ndarray): _description_
        y_train (jnp.ndarray): _description_
        y_test (jnp.ndarray): _description_

    Raises:
        SizeIncompatible: _description_
    """
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
           method: SampleMethod | None = None,
           batch_size: int = 2 ** 7,
           num_batches: int = 2 ** 7,
           tolerance: float = 1e-2,
           seed: int = 42) -> ShapleyResults:
    """_summary_

    Args:
        X_train (np.ndarray | jnp.ndarray | pd.DataFrame): _description_
        X_test (np.ndarray | jnp.ndarray | pd.DataFrame): _description_
        y_train (np.ndarray | jnp.ndarray | pd.Series): _description_
        y_test (np.ndarray | jnp.ndarray | pd.Series): _description_
        reg (float, optional): _description_. Defaults to 0.
        method (SampleMethod | None, optional): _description_. Defaults to None.
        batch_size (int, optional): _description_. Defaults to 2**7.
        num_batches (int, optional): _description_. Defaults to 2**7.
        tolerance (float, optional): _description_. Defaults to 1e-2.
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        ShapleyResults: _description_
    """
    # Converting data into JAX arrays.
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    validate_data(X_train, X_test, y_train, y_test)

    N, p = X_train.shape
    M, _ = X_test.shape
    if method is None:
        if p > 10:
            method = 'argsort'
        else:
            method = 'argsort' ### XXX exact needs to be implemented still

    rng = random.PRNGKey(seed)
    compute_spa = LSSPA(key=rng,
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
    """_summary_

    Attributes:
        key (_type_): _description_
        p (_type_): _description_
    """

    def __init__(self, key, p: int):
        """_summary_

        Args:
            key (_type_): _description_
            p (int): _description_
        """
        self.key = key
        self.p = p

    @abstractmethod
    def __call__(self, num_perms: int) -> jnp.ndarray:
        """_summary_

        Args:
            num_perms (int): _description_

        Returns:
            jnp.ndarray: _description_
        """
        pass

    @property
    def p(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """_summary_

        Args:
            new_p (int): _description_
        """
        self._p = new_p


class RandomPermutations(Permutations):
    """_summary_

    Attributes:
        key (_type_): _description_
        p (_type_): _description_
    """

    def __call__(self, num_perms: int) -> jnp.ndarray:
        """_summary_

        Args:
            num_perms (int): _description_

        Returns:
            jnp.ndarray: _description_
        """
        # Split the key to ensure different permutations each call
        self.key, keygenkey = random.split(self.key)
        to_permute =jnp.tile(jnp.arange(0, self.p), (num_perms, 1))
        # Generate random permutations
        return random.permutation(keygenkey, to_permute,
                                  axis=1, independent=True)


class PermutohedronPermutations(Permutations):
    """_summary_

    Attributes:
        key (_type_): _description_
        p (_type_): _description_
        qmc (_type_): _description_
    """

    def __init__(self, key, p: int):
        """_summary_

        Args:
            key (_type_): _description_
            p (int): _description_
        """
        self.key = key
        self.p = p

    def __call__(self, num_perms: int) -> jnp.ndarray:
        """_summary_

        Args:
            num_perms (int): _description_

        Returns:
            jnp.ndarray: _description_
        """
        # Generate permutohedron samples
        samples = jnp.array(self.qmc.random(num_perms))
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        samples = self.project(samples)
        samples = jnp.argsort(samples, axis=1)
        return samples

    @property
    def p(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """_summary_

        Args:
            new_p (int): _description_
        """
        self._p = new_p
        self.key, keygenkey = random.split(self.key)
        seed = int(random.choice(keygenkey, 100000))
        self.qmc = MultivariateNormalQMC(np.zeros(self.p-1), seed=seed,
                                         inv_transform=False)

    @partial(jit, static_argnums=0)
    def project(self, x: jnp.ndarray):
        """_summary_

        Args:
            x (jnp.ndarray): _description_

        Returns:
            _type_: _description_
        """
        tril_part = jnp.tril(jnp.ones((self.p-1, self.p)))
        diag_part = jnp.diag(-jnp.arange(1, self.p), 1)[:-1]
        U = tril_part + diag_part
        U = U / jnp.linalg.norm(U, axis=1, keepdims=True)
        return x @ U


class ArgsortPermutations(Permutations):
    """_summary_

    Attributes:
        key (_type_): _description_
        p (_type_): _description_
        qmc (_type_): _description_
    """

    def __init__(self, key, p: int):
        """_summary_

        Args:
            key (_type_): _description_
            p (int): _description_
        """
        self.key = key
        self.p = p

    def __call__(self, num_perms: int) -> jnp.ndarray:
        """_summary_

        Args:
            num_perms (int): _description_

        Returns:
            jnp.ndarray: _description_
        """
        # Generate argsort samples
        samples = jnp.array(self.qmc.random(num_perms))
        return jnp.argsort(samples, axis=1)

    @property
    def p(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """_summary_

        Args:
            new_p (int): _description_
        """
        self._p = new_p
        self.key, keygenkey = random.split(self.key)
        seed = int(random.choice(keygenkey, 100000))
        self.qmc = Sobol(self.p, seed=seed)


class RiskEstimate:
    """_summary_

    Attributes:
        key (_type_): _description_
        batch_size (int): _description_
        p (int): _description_
        mean (jnp.ndarray): _description_
        cov (jnp.ndarray): _description_
        _i (int): _description_
    """

    def __init__(self, key, batch_size: int, p: int):
        """_summary_

        Args:
            key (_type_): _description_
            batch_size (int): _description_
            p (int): _description_

        Returns:
            _type_: _description_
        """
        self.key = key
        self.batch_size = batch_size
        self.p = p
        self.mean = jnp.zeros((self.p,))
        self.cov = jnp.zeros((self.p, self.p))
        self._i = 1

        def risk_sample(key, cov):
            """_summary_

            Args:
                key (_type_): _description_
                cov (_type_): _description_

            Returns:
                _type_: _description_
            """
            sample_diffs = random.multivariate_normal(key, jnp.zeros(p),
                                                      cov, shape=((1000,)),
                                                      method='svd')
            abs_diffs = jnp.abs(sample_diffs)
            norms = jnp.linalg.norm(sample_diffs, axis=1)
            abs_quantile = jnp.quantile(abs_diffs, 0.95, axis=0)
            norms_quantile = jnp.quantile(norms, 0.95)
            return abs_quantile, norms_quantile

        self.risk_sample = jit(risk_sample)

    def __call__(self, batch: jnp.ndarray) -> float:
        """_summary_

        Args:
            batch (jnp.ndarray): _description_

        Returns:
            float: _description_
        """
        self.key, samplekey = random.split(self.key)
        self.mean, self.cov = self._call_helper(self._i, self.mean,
                                                self.cov, batch)
        num_pts = self.batch_size * self._i
        unbiased_cov = num_pts / (num_pts - 1) * self.cov
        feature_err, global_err = self.risk_sample(samplekey,
                                                   unbiased_cov/num_pts)
        self._i += 1
        return feature_err, global_err

    @staticmethod
    @jit
    def _call_helper(i: int, mean: jnp.ndarray, cov: jnp.ndarray,
                     batch: jnp.ndarray) -> jnp.ndarray:
        """_summary_

        Args:
            i (int): _description_
            mean (jnp.ndarray): _description_
            cov (jnp.ndarray): _description_
            batch (jnp.ndarray): _description_

        Returns:
            jnp.ndarray: _description_
        """
        batch_mean = jnp.mean(batch, axis=0)
        batch_cov = jnp.cov(batch.T, bias=True)

        mean_diff = mean - batch_mean
        correction_term = (i-1) / i**2 * jnp.outer(mean_diff, mean_diff)
        new_mean = (i-1) / i * mean + batch_mean / i
        new_cov = (i-1) / i * cov + batch_cov / i + correction_term
        return new_mean, new_cov

    def reset(self):
        """_summary_"""
        self.mean = jnp.zeros((self.p,))
        self.cov = jnp.zeros((self.p, self.p))
        self._i = 1


class SquareShapley:
    """_summary_

    Attributes:
        p (int): _description_
    """

    def __init__(self, p: int):
        """_summary_

        Args:
            p (int): _description_
        """
        self.p = p

    def __call__(self, X_train: jnp.ndarray, X_test: jnp.ndarray,
                 y_train: jnp.ndarray, y_test: jnp.ndarray,
                 y_norm_sq: jnp.ndarray, perms: jnp.ndarray) -> jnp.ndarray:
        """_summary_

        Args:
            X_train (jnp.ndarray): _description_
            X_test (jnp.ndarray): _description_
            y_train (jnp.ndarray): _description_
            y_test (jnp.ndarray): _description_
            y_norm_sq (jnp.ndarray): _description_
            perms (jnp.ndarray): _description_

        Returns:
            jnp.ndarray: _description_
        """
        return self.square_shapley(X_train, X_test, y_train, y_test,
                                   y_norm_sq, perms)

    @property
    def p(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """_summary_

        Args:
            new_p (int): _description_

        Returns:
            _type_: _description_
        """
        self._p = new_p
        def square_shapley(X_train: jnp.ndarray, X_test: jnp.ndarray,
                           y_train: jnp.ndarray, y_test: jnp.ndarray,
                           y_norm_sq: jnp.ndarray,
                           perms: jnp.ndarray) -> jnp.ndarray:
            Q, R = jnp.linalg.qr(X_train[:, perms])
            X = X_test[:, perms]

            Y = jnp.triu(Q.T @ jnp.tile(y_train, (self.p, 1)).T)
            T = jsp.linalg.solve_triangular(R, Y)
            T = jnp.hstack((jnp.zeros((self.p, 1)), T))

            Y_test = jnp.tile(y_test, (self.p+1, 1))
            costs = jnp.sum((X @ T - Y_test.T) ** 2, axis=0)
            R_sq = (jnp.linalg.norm(y_test) ** 2 - costs) / y_norm_sq
            perm_scores = jnp.ediff1d(R_sq)[jnp.argsort(perms)]
            return perm_scores

        vmap_square_shapley = vmap(square_shapley,
                                   (None, None, None, None, None, 0), 0)
        self.square_shapley = jit(vmap_square_shapley)


class LSSPA:
    """_summary_

    Attributes:
        p (int): _description_
        sample_method (SampleMethod): _description_
        batch_size (int): _description_
        key (_type_): _description_
        sampler (_type_): _description_
        _square_shapley (_type_): _description_
        risk_estimate (_type_): _description_
    """

    def __init__(self, key, p: int = 10,
                 sample_method: SampleMethod = 'random',
                 batch_size: int = 2**13):
        """_summary_

        Args:
            key (_type_): _description_
            p (int, optional): _description_. Defaults to 10.
            sample_method (SampleMethod, optional): _description_. Defaults to 'random'.
            batch_size (int, optional): _description_. Defaults to 2**13.
        """
        self._p = p
        self.sample_method = sample_method
        self.key, permkey, riskkey = random.split(key, 3)
        # Initialize appropriate permutation generator based on sampling method
        if self.sample_method == 'random':
            self.sampler = RandomPermutations(permkey, self.p)
        elif self.sample_method == 'permutohedron':
            self.sampler = PermutohedronPermutations(permkey, self.p)
        else:
            self.sampler = ArgsortPermutations(permkey, self.p)

        self.batch_size = batch_size
        self._square_shapley = SquareShapley(p)
        self.risk_estimate = RiskEstimate(riskkey, self.batch_size, self.p)

    def __call__(self, X_train: jnp.ndarray, X_test: jnp.ndarray,
                 y_train: jnp.ndarray, y_test: jnp.ndarray,
                 reg: float, max_num_batches: int = 1,
                 eps: float = 1e-3, y_norm_sq: jnp.ndarray | None = None,
                 return_history: bool = True) -> ShapleyResults:
        """_summary_

        Args:
            X_train (jnp.ndarray): _description_
            X_test (jnp.ndarray): _description_
            y_train (jnp.ndarray): _description_
            y_test (jnp.ndarray): _description_
            reg (float): _description_
            max_num_batches (int, optional): _description_. Defaults to 1.
            eps (float, optional): _description_. Defaults to 1e-3.
            y_norm_sq (jnp.ndarray | None, optional): _description_. Defaults to None.
            return_history (bool, optional): _description_. Defaults to True.

        Returns:
            ShapleyResults: _description_
        """
        if y_norm_sq is None:
            N = 1 if np.isclose(reg, 0) else len(X_train)
            M = len(X_test)
            X_train, X_test, y_train, y_test, y_norm_sq, = (
                self.process_data(N, M, X_train, X_test, y_train, y_test, reg))
        theta = jnp.linalg.lstsq(X_train, y_train)[0]
        # XXX we need to correct the r-squared computation if cholesky method
        # is done. the attributions add up to the right r-squared so maybe we
        # just return that always
        r_squared = (np.linalg.norm(y_test) ** 2 - 
                    np.linalg.norm(X_test @ theta - y_test) ** 2) / y_norm_sq

        attribution_history = jnp.zeros((0, self.p)) if return_history else None
        scores = jnp.zeros(self.p)
        error_history = jnp.zeros((0,)) if return_history else None
        self.risk_estimate.reset()

        for i in range(1, max_num_batches+1):
            batch = self.sampler(self.batch_size)
            perm_scores = self._square_shapley(X_train, X_test, y_train,
                                               y_test, y_norm_sq, batch)
            scores = (i-1)/i * scores + jnp.mean(perm_scores, axis=0) / i
            feature_risk, global_risk = self.risk_estimate(perm_scores)
            if return_history:
                attribution_history = jnp.vstack((attribution_history,
                                                  perm_scores))
                error_history = jnp.append(
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
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """_summary_

        Args:
            new_p (int): _description_
        """
        if self.p == new_p:
            return

        self.key, permkey, riskkey = random.split(self.key, 3)
        # Initialize appropriate permutation generator based on sampling method
        if self.sample_method == 'random':
            self.sampler = RandomPermutations(permkey, self.p)
        elif self.sample_method == 'permutohedron':
            self.sampler = PermutohedronPermutations(permkey, self.p)
        else:
            self.sampler = ArgsortPermutations(permkey, self.p)

        self._square_shapley = SquareShapley(self.p)
        self.risk_estimate = RiskEstimate(riskkey, self.batch_size, self.p)

    @partial(jit, static_argnums=(0, 1, 2))
    def process_data(self, N: int, M: int, X_train: jnp.ndarray,
                     X_test: jnp.ndarray, y_train: jnp.ndarray,
                     y_test: jnp.ndarray,
                     reg: float) -> Tuple[jnp.ndarray, jnp.ndarray,
                                          jnp.ndarray, jnp.ndarray,
                                          jnp.ndarray]:
        """_summary_

        Args:
            N (int): _description_
            M (int): _description_
            X_train (jnp.ndarray): _description_
            X_test (jnp.ndarray): _description_
            y_train (jnp.ndarray): _description_
            y_test (jnp.ndarray): _description_
            reg (float): _description_

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: _description_
        """
        X_train = X_train / jnp.sqrt(N)
        X_train = jnp.vstack((X_train, jnp.sqrt(reg) * jnp.eye(self.p)))
        y_train = y_train / jnp.sqrt(N)
        y_train = jnp.concatenate((y_train, jnp.zeros(self.p)))

        y_norm_sq = jnp.linalg.norm(y_test) ** 2

        Q, X_train, = jnp.linalg.qr(X_train)
        Q_ts, X_test = jnp.linalg.qr(X_test)
        y_train = Q.T @ y_train
        y_test = Q_ts.T @ y_test
        return X_train, X_test, y_train, y_test, y_norm_sq