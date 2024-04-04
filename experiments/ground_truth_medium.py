import os
import time

import ls_spa
import numpy as np
from scipy.stats.qmc import MultivariateNormalQMC, Sobol

if not os.path.isdir("./experiments/data"):
    os.makedirs("./experiments/data")
    
rng = np.random.default_rng(42)

# Set params
EXP_NAME = "Medium"
p = 100
N = 100000
M = 100000
STN_RATIO = 5.
REG = 0
conditioning = 20.
max_samples = 2 ** 13

class GeneratorLen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


class AlternatingGenerator(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length
        self.last_sample = None
        self.next_call_is_direct = True

    def __len__(self):
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            if self.next_call_is_direct:
                self.last_sample = next(self.gen)
                yield self.last_sample
                self.next_call_is_direct = False
            else:
                yield self.last_sample[::-1]
                self.next_call_is_direct = True


def permutohedron_samples(qmc, num_perms: int):
    # Sample on surface of sphere
    samples = qmc.random(num_perms)
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)

    # Project onto permutohedron
    tril_part = np.tril(np.ones((p-1, p)))
    diag_part = np.diag(-np.arange(1, p), 1)[:-1]
    U = tril_part + diag_part
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    samples = samples @ U
    return np.argsort(samples, axis=1)


def argsort_samples(qmc, num_perms: int):
    return np.argsort(qmc.random(num_perms), axis=1)


def gen_data(rng):
    # We want to generate a covariance matrix so some features have very large
    # covariances (in magnitude).
    A = rng.standard_normal((p, int(p / conditioning)))
    cov = A @ A.T + np.eye(p)
    v = np.sqrt(np.diag(cov))
    cov = cov / np.outer(v, v)

    # We sample observations to create X_train and X_test.
    X_train = rng.multivariate_normal(np.zeros(p), cov, (N,),
                                      method='svd')
    X_test = rng.multivariate_normal(np.zeros(p), cov, (M,),
                                     method='svd')

    # We want most of the features to be irrelevant.
    theta_vals = np.zeros(p)
    theta_vals[:max((p+1)//10, 1)] = np.full(max((p+1)//10, 1), 2.0)
    theta_true = rng.permutation(theta_vals)

    # We create the response variables and add a little noise.
    std = np.sqrt(np.sum(np.diag(cov) * theta_true**2) / STN_RATIO)
    y_train = X_train @ theta_true + std * rng.standard_normal(N)

    X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    X_train = X_train - X_train_mean
    y_train_mean = np.mean(y_train)
    y_train = y_train - y_train_mean

    y_test = X_test @ theta_true + std * rng.standard_normal(M)
    X_test = X_test - X_train_mean
    y_test = y_test - y_train_mean

    return X_train, X_test, y_train, y_test, theta_true, cov


# Generate data
X_train, X_test, y_train, y_test, true_theta, cov = gen_data(rng)

# Generate ground truth
gt_location = f"./experiments/data/gt_{EXP_NAME}.npy"
gt_permutations_gen = GeneratorLen((rng.permutation(p) for _ in range(2**19)), 2**19)
gt_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                           perms=gt_permutations_gen, tolerance=0.0)
gt_attributions = gt_results.attribution
gt_attributions = gt_attributions * gt_results.r_squared / np.sum(gt_attributions)
np.save(gt_location, gt_results.attribution)
