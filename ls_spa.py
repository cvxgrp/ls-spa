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
    """Data class to store the results of the Shapley procedure.

    Attributes:
        attribution: Array of Shapley values for each feature.
        attribution_history: Array of Shapley values for each iteration.
            None if return_history=False in LSSA call.
        theta: Array of regression coefficients.
        overall_error: Mean absolute error of the Shapley values.
        error_history: Array of mean absolute errors for each iteration.
            None if return_history=False in LSSA call.
        attribution_errors: Array of absolute errors for each feature.
        r_squared: R-squared statistic of the regression.
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
    """Validate data dimensions.

    Args:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.
        y_train: Training response vector.
        y_test: Testing response vector.

    Raises:
        SizeIncompatible: If data dimensions are incompatible.
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
           seed: int = 42,
           return_history: bool = False) -> ShapleyResults:
    """
    Compute Shapley values for the given data using
    Least-Squares Shapley Performance Attribution (LS-SPA).

    Args:
        X_train: Training feature matrix.
        X_test: Testing feature matrix.
        y_train: Training response vector.
        y_test: Testing response vector.
        reg: Regularization parameter (Default 0).
        method: Permutation sampling method. Options include 'random',
            'permutohedron', 'argsort', and 'exact'. If None, 'argsort' is used
            if the number of features is greater than 10; otherwise, 'exact' is used.
        batch_size: Number of permutations in each batch (Default 2**7).
        num_batches: Maximum number of batches (Default 2**7).
        tolerance: Convergence tolerance for the Shapley values (Default 1e-2).
        seed: Seed for random number generation (Default 42).
        return_history: Flag to determine whether to return the history of
            error estimates and attributions for each feature chain.

    Returns:
        ShapleyResults: Calculated Shapley values and other results.
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
    compute_spa = LSSA(key=rng,
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
                       return_history=return_history)


class Permutations(ABC):
    """
    Base class for permutation generators. Subclasses must implement the __call__ method.

    Attributes:
        key (jax.random.PRNGKey): The random number generator key.
        p (int): The dimension of the problem.

    Methods:
        __call__(num_perms: int) -> jnp.ndarray: Abstract method to generate permutations.
    """

    def __init__(self, key, p: int):
        """
        Initialize a new instance of permutation generator.

        Args:
            key (jax.random.PRNGKey): The random number generator key.
            p (int): The dimension of the problem.
        """
        self.key = key
        self.p = p

    @abstractmethod
    def __call__(self, num_perms: int) -> jnp.ndarray:
        """
        Abstract method to generate permutations. Must be implemented by subclasses.

        Args:
            num_perms (int): The number of permutations to generate.

        Returns:
            jnp.ndarray: The generated permutations.
        """
        pass

    @property
    def p(self):
        """
        Get the dimension of the problem.

        Returns:
            int: The dimension of the problem.
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """
        Set a new dimension for the problem.

        Args:
            new_p (int): The new dimension of the problem.
        """
        self._p = new_p


class RandomPermutations(Permutations):
    """
    Class for generating random permutations.

    Inherits from Permutations.

    Methods:
        __call__(num_perms: int) -> jnp.ndarray: Generate random permutations.
    """

    def __call__(self, num_perms: int) -> jnp.ndarray:
        """
        Generate a specified number of random permutations.

        Args:
            num_perms (int): The number of permutations to generate.

        Returns:
            jnp.ndarray: An array containing the generated permutations. Each row represents a permutation.
        """
        # Split the key to ensure different permutations each call
        self.key, keygenkey = random.split(self.key)
        to_permute =jnp.tile(jnp.arange(0, self.p), (num_perms, 1))
        # Generate random permutations
        return random.permutation(keygenkey, to_permute,
                                  axis=1, independent=True)


class PermutohedronPermutations(Permutations):
    """
    Class for generating permutations based on the permutohedron sampling method.

    Inherits from Permutations.

    Methods:
        __call__(num_perms: int) -> jnp.ndarray: Generate permutations based on the permutohedron sampling method.
    """
    def __init__(self, key, p: int):
        """
        Initialize the PermutohedronPermutations object.

        Args:
            key (type): Description of what key is.
            p (int): Dimension of the permutohedron.
        """
        self.key = key
        self.p = p

    def __call__(self, num_perms: int) -> jnp.ndarray:
        """
        Generate permutations based on the permutohedron sampling method.

        Args:
            num_perms (int): Number of permutations to generate.

        Returns:
            jnp.ndarray: Array of generated permutations.
        """
        # Generate permutohedron samples
        samples = jnp.array(self.qmc.random(num_perms))
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        samples = self.project(samples)
        samples = jnp.argsort(samples, axis=1)
        return samples

    @property
    def p(self):
        """
        Getter for p.

        Returns:
            int: Dimension of the permutohedron.
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """
        Setter for p.

        Args:
            new_p (int): New dimension of the permutohedron.
        """
        self._p = new_p
        self.key, keygenkey = random.split(self.key)
        seed = int(random.choice(keygenkey, 100000))
        self.qmc = MultivariateNormalQMC(np.zeros(self.p-1), seed=seed,
                                         inv_transform=False)

    @partial(jit, static_argnums=0)
    def project(self, x: jnp.ndarray):
        """
        Project samples onto the permutohedron.

        Args:
            x (jnp.ndarray): Array of samples.

        Returns:
            jnp.ndarray : Array of projected samples.
        """
        tril_part = jnp.tril(jnp.ones((self.p-1, self.p)))
        diag_part = jnp.diag(-jnp.arange(1, self.p), 1)[:-1]
        U = tril_part + diag_part
        U = U / jnp.linalg.norm(U, axis=1, keepdims=True)
        return x @ U


class ArgsortPermutations(Permutations):
    """
    Class for generating permutations based on the argsort sampling method.

    Inherits from Permutations.

    Methods:
        __call__(num_perms: int) -> jnp.ndarray: Generate permutations based on the argsort sampling method.
    """

    def __init__(self, key, p: int):
        """
        Initialize an instance of ArgsortPermutations.

        Args:
            key: A key used for generating random values.
            p (int): The number of items to permute.

        """
        self.key = key
        self.p = p

    def __call__(self, num_perms: int) -> jnp.ndarray:
        """
        Generate permutations based on the argsort sampling method.

        Args:
            num_perms (int): The number of permutations to generate.

        Returns:
            jnp.ndarray: A 2D array of argsort permutations.
        """
        # Generate argsort samples
        samples = jnp.array(self.qmc.random(num_perms))
        return jnp.argsort(samples, axis=1)

    @property
    def p(self):
        """
        Getter for number of items to permute.

        Returns:
            int: The number of items to permute.
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """
        Setter for number of items to permute.

        Args:
            new_p (int): The new number of items to permute.
        """
        self._p = new_p
        self.key, keygenkey = random.split(self.key)
        seed = int(random.choice(keygenkey, 100000))
        self.qmc = Sobol(self.p, seed=seed)


class RiskEstimate:
    """
    Class for estimating the risk (error) of the Shapley value calculations.

    Attributes:
        key (jax.random.PRNGKey): The random number generator key.
        batch_size (int): The size of each batch of permutations.
        p (int): The dimension of the problem.
        atts (jnp.ndarray): Array to store the mean of the attribute values for each batch of permutations.
        _i (int): Counter for number of batches of permutations processed.

    Methods:
        __call__(batch: jnp.ndarray, curr_atts: jnp.ndarray) -> float: Estimate the risk (error) of the current Shapley value calculations.
        reset(): Reset the risk estimator.
    """

    def __init__(self, key, batch_size: int, p: int):
        """
        Initialize the RiskEstimate object.

        Args:
            key (jax.random.PRNGKey): The random number generator key.
            batch_size (int): The size of each batch of permutations.
            p (int): The dimension of the problem.
        """
        self.key = key
        self.batch_size = batch_size
        self.p = p
        self.atts = jnp.zeros((500, self.p))
        self._i = 1

        def risk_sample(batch, subsamplekey):
            # Helper function for estimating risk
            group = random.choice(subsamplekey, batch,
                                  shape=(batch_size//2, 1), axis=0)
            return jnp.mean(group, axis=0)
        self.risk_sample = jit(vmap(risk_sample, (None, 0), 0))

    def __call__(self, batch: jnp.ndarray, curr_atts: jnp.ndarray) -> float:
        """
        Estimate the risk (error) of the current Shapley value calculations.

        Args:
            batch (jnp.ndarray): Array of batches to calculate the risk.
            curr_atts (jnp.ndarray): Current attribute values.

        Returns:
            float: The estimated risk (error).
        """
        keys = random.split(self.key, 501)
        self.key, subsamplekeys = keys[0], keys[1:]
        group = self.risk_sample(batch, subsamplekeys)
        self.atts = self._call_helper(
            self._i, self.atts, group
        )
        diffs = self.atts - jnp.expand_dims(curr_atts, 0)
        errs = jnp.mean(jnp.abs(diffs), axis=0)
        global_error = jnp.mean(jnp.linalg.norm(diffs, axis=1))
        self._i += 1
        return errs, global_error

    @staticmethod
    @jit
    def _call_helper(i: int, atts: jnp.ndarray,
                     group: jnp.ndarray) -> jnp.ndarray:
        """
        Helper function for updating attribute values.

        Args:
            i (int): The counter for number of batches of permutations processed.
            atts (jnp.ndarray): Array to store the mean of the attribute values for each batch of permutations.
            group (jnp.ndarray): The risk sample group.

        Returns:
            jnp.ndarray: The updated attribute values.
        """
        atts = (i - 1) / i * atts + group / i
        return atts

    def reset(self):
        """
        Reset the risk estimator by setting the attribute values to zeros and counter to 1.
        """
        self.atts = jnp.zeros((500, self.p))
        self._i = 1


class SquareShapley:
    """
    Class for calculating Shapley values for a least squares problem
    with a square data matrix.

    Attributes:
        p (int): The dimension of the problem.

    Methods:
        __call__(X_train: jnp.ndarray, X_test: jnp.ndarray, y_train: jnp.ndarray, y_test: jnp.ndarray, y_norm_sq: jnp.ndarray, perms: jnp.ndarray) -> jnp.ndarray: Calculate Shapley values.
    """

    def __init__(self, p: int):
        """
        Initialize the SquareShapley class.

        Args:
            p (int): The dimension of the problem.
        """
        self.p = p

    def __call__(self, X_train: jnp.ndarray, X_test: jnp.ndarray,
                 y_train: jnp.ndarray, y_test: jnp.ndarray,
                 y_norm_sq: jnp.ndarray, perms: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate Shapley values.

        Args:
            X_train (jnp.ndarray): The training data.
            X_test (jnp.ndarray): The test data.
            y_train (jnp.ndarray): The training labels.
            y_test (jnp.ndarray): The test labels.
            y_norm_sq (jnp.ndarray): The square norm of the labels.
            perms (jnp.ndarray): The permutations.

        Returns:
            jnp.ndarray: The calculated Shapley values.
        """
        return self.square_shapley(X_train, X_test, y_train, y_test,
                                   y_norm_sq, perms)

    @property
    def p(self):
        """
        Get the dimension of the problem.

        Returns:
            int: The dimension of the problem.
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """
        Set the dimension of the problem.

        Args:
            new_p (int): The new dimension of the problem.
        """
        self._p = new_p
        def square_shapley(X_train: jnp.ndarray, X_test: jnp.ndarray,
                           y_train: jnp.ndarray, y_test: jnp.ndarray,
                           y_norm_sq: jnp.ndarray,
                           perms: jnp.ndarray) -> jnp.ndarray:
            """
            Helper function for calculating Shapley values.

            Args:
                X_train (jnp.ndarray): The features of the training data.
                X_test (jnp.ndarray): The features of the test data.
                y_train (jnp.ndarray): The target variable of the training data.
                y_test (jnp.ndarray): The target variable of the test data.
                y_norm_sq (jnp.ndarray): The square of the norm of the target variable of the test data.
                perms (jnp.ndarray): The permutations of the indices of the feature space.

            Returns:
                perm_scores (jnp.ndarray): The Shapley values for each feature in the training dataset.
            """
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


class LSSA:
    """
    Class for calculating Shapley values for a least squares problem.

    Attributes:
        _p (int): The dimension of the problem.
        sample_method (SampleMethod): The method for generating permutations ('random', 'permutohedron', or 'argsort').
        key (jax.random.PRNGKey): The random number generator key.
        sampler (Permutations): The permutation generator.
        batch_size (int): The size of each batch of permutations.
        _square_shapley (SquareShapley): The Shapley value calculator.
        risk_estimate (RiskEstimate): The risk estimator.

    Methods:
        __call__(X_train: jnp.ndarray, X_test: jnp.ndarray, y_train: jnp.ndarray, y_test: jnp.ndarray, reg: float, max_num_batches: int, eps: float, y_norm_sq: jnp.ndarray, return_history: bool) -> ShapleyResults: Calculate Shapley values.
        process_data(N: int, M: int, X_train: jnp.ndarray, X_test: jnp.ndarray, y_train: jnp.ndarray, y_test: jnp.ndarray, reg: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: Preprocess the data for Shapley value calculations.
    """

    def __init__(self, key, p: int = 10,
                 sample_method: SampleMethod = 'random',
                 batch_size: int = 2**13):
        """
        Initializes the LSSA class with given parameters and sets up the appropriate permutation generator and Shapley value calculator.

        Args:
            key: Random number generator key.
            p (int, optional): The dimension of the problem. Default is 10.
            sample_method (str, optional): The method for generating permutations. Options are 'random', 'permutohedron', or 'argsort'. Default is 'random'.
            batch_size (int, optional): The size of each batch of permutations. Default is 2**13.
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
        """
        Calculates Shapley values.

        Args:
            X_train (jnp.ndarray): Training data features.
            X_test (jnp.ndarray): Testing data features.
            y_train (jnp.ndarray): Training data labels.
            y_test (jnp.ndarray): Testing data labels.
            reg (float): Regularization parameter.
            max_num_batches (int, optional): Maximum number of batches. Default is 1.
            eps (float, optional): Error tolerance. Default is 1e-3.
            y_norm_sq (jnp.ndarray, optional): Squared norm of y. If None, it will be calculated.
            return_history (bool, optional): Whether to return the history of Shapley values. Default is True.

        Returns:
            ShapleyResults: A data class containing Shapley values and other relevant information.
        """
        if y_norm_sq is None:
            N = 1 if np.isclose(reg, 0) else len(X_train)
            M = len(X_test)
            X_train, X_test, y_train, y_test, y_norm_sq, y_test_proj = (
                self.process_data(N, M, X_train, X_test, y_train, y_test, reg))
        theta = jnp.linalg.lstsq(X_train, y_train)[0]
        r_squared = 1 - (
            jnp.linalg.norm(X_test @ theta - y_test) ** 2
            + y_norm_sq - jnp.linalg.norm(y_test_proj)**2)/y_norm_sq

        attribution_history = jnp.zeros((0, self.p)) if return_history else None
        scores = jnp.zeros(self.p)
        error_history = jnp.zeros((0,)) if return_history else None
        self.risk_estimate.reset()

        for i in range(1, max_num_batches+1):
            batch = self.sampler(self.batch_size)
            perm_scores = self._square_shapley(X_train, X_test, y_train,
                                               y_test, y_norm_sq, batch)
            scores = (i-1)/i * scores + jnp.mean(perm_scores, axis=0) / i
            feature_risk, global_risk = self.risk_estimate(perm_scores, scores)
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
        """
        Get the dimension of the problem.

        Returns:
            int: The dimension of the problem.
        """
        return self._p

    @p.setter
    def p(self, new_p: int):
        """
        Set the dimension of the problem.

        Args:
            new_p (int): The new dimension of the problem.
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
        """
        Preprocesses the data for Shapley value calculations.

        Args:
            N (int): Number of training samples.
            M (int): Number of testing samples.
            X_train (jnp.ndarray): Training data features.
            X_test (jnp.ndarray): Testing data features.
            y_train (jnp.ndarray): Training data labels.
            y_test (jnp.ndarray): Testing data labels.
            reg (float): Regularization parameter.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: Preprocessed data.
        """
        X_train = X_train / jnp.sqrt(N)
        X_train = jnp.vstack((X_train, jnp.sqrt(reg) * jnp.eye(self.p)))
        y_train = y_train / jnp.sqrt(N)
        y_train = jnp.concatenate((y_train, jnp.zeros(self.p)))

        y_norm_sq = jnp.linalg.norm(y_test) ** 2

        Q, X_train, = jnp.linalg.qr(X_train)
        Q_ts, X_test = jnp.linalg.qr(X_test)
        y_train = Q.T @ y_train
        y_test, y_test_proj = Q_ts.T @ y_test, Q_ts @ (Q_ts.T @ y_test)
        return X_train, X_test, y_train, y_test, y_norm_sq, y_test_proj