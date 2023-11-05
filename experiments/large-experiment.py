import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import pmap, random

import ls_spa

# CONFIG
p = 1000 # Feature counts to test
N = int(1e6) # Number of train observations
M = int(1e6) # Number of test observations
STN_RATIO = 5.0 # Signal-to-noise ratio

NUM_BATCHES = 2 ** 4
BATCH_SIZE = 2 ** 7

key = random.PRNGKey(364)
gpus = jax.devices("gpu")
gpu_N = N // len(gpus)
gpu_M = M // len(gpus)

@partial(pmap, in_axes=(0, None, None), out_axes=0)
def gen_data_time(key, cov, theta):
    key, train_key = random.split(key)
    X_train = random.multivariate_normal(
        train_key, jnp.zeros(p), cov, (gpu_N,), method='svd'
    )

    key, train_noise_key = random.split(key)
    std = jnp.sqrt(jnp.sum(jnp.diag(cov) * theta**2) / STN_RATIO)
    y_train = X_train @ theta + std * random.normal(train_noise_key, (gpu_N,))

    X_train_mean = jnp.mean(X_train, axis=0, keepdims=True)
    X_train = X_train - X_train_mean
    y_train_mean = jnp.mean(y_train)
    y_train = y_train - y_train_mean
    return X_train_mean, y_train_mean


@partial(pmap, in_axes=(0, None, None, None, None), out_axes=0)
def gen_train_data_aux(key, cov, theta, X_train_mean, y_train_mean):
    key, train_key = random.split(key)
    X_train = random.multivariate_normal(
        train_key, jnp.zeros(p), cov, (gpu_N,), method='svd'
    )

    key, train_noise_key = random.split(key)
    std = jnp.sqrt(jnp.sum(jnp.diag(cov) * theta**2) / STN_RATIO)
    y_train = X_train @ theta + std * random.normal(train_noise_key, (gpu_N,))

    X_train = X_train - X_train_mean
    y_train = y_train - y_train_mean

    cat_train = jnp.hstack((X_train, y_train[:, None])) / jnp.sqrt(N)
    gram_train = jnp.dot(cat_train.T, cat_train)

    return gram_train


@partial(pmap, in_axes=(0, None, None, None, None), out_axes=0)
def gen_test_data_aux(key, cov, theta, X_train_mean, y_train_mean):
    key, test_key = random.split(key)
    X_test = random.multivariate_normal(
        test_key, jnp.zeros(p), cov, (gpu_M,), method='svd'
    )

    key, test_noise_key = random.split(key)
    std = jnp.sqrt(jnp.sum(jnp.diag(cov) * theta**2) / STN_RATIO)
    y_test = X_test @ theta + std * random.normal(test_noise_key, (gpu_M,))
    X_test = X_test - X_train_mean
    y_test = y_test - y_train_mean

    cat_test = jnp.hstack((X_test, y_test[:, None]))
    gram_test = jnp.dot(cat_test.T, cat_test)

    return gram_test


def gen_data(key):
    # We want to generate a covariance matrix so some features have very large
    # covariances (in magnitude).
    key, subkey, subkey_shuffle = random.split(key, 3)
    A = random.normal(subkey, (p, p//20))
    cov = A @ A.T + jnp.eye(p)
    v = jnp.sqrt(jnp.diag(cov))
    cov = cov / jnp.outer(v, v)

    # We want about p/10 features to have value 2.
    theta_vals = jnp.full((p+1)//10, 2)
    padded_theta_vals = jnp.pad(theta_vals, (0, p - (p+1)//10))
    theta_true = random.permutation(subkey_shuffle, padded_theta_vals)

    # We create the data
    aux_keys = random.split(key, len(gpus)+1)
    key, train_data_keys = aux_keys[0], aux_keys[1:]
    _, _ = gen_data_time(train_data_keys, cov, theta_true)
    start_datagen = time.time()
    X_train_mean, y_train_mean = gen_data_time(train_data_keys, cov, theta_true)
    X_train_mean = jnp.mean(X_train_mean, axis=0)
    y_train_mean = jnp.mean(y_train_mean, axis=0)
    print(f"X_train_mean shape: {X_train_mean.shape}")
    print(f"y_train_mean shape: {y_train_mean.shape}")
    start_gramgen = time.time()
    gram_train = gen_train_data_aux(train_data_keys, cov, theta_true, X_train_mean, y_train_mean)
    gram_train = jnp.sum(gram_train, axis=0)

    aux_keys = random.split(key, len(gpus)+1)
    key, train_data_keys = aux_keys[0], aux_keys[1:]

    gram_test = gen_test_data_aux(train_data_keys, cov, theta_true, X_train_mean, y_train_mean)
    gram_test = jnp.sum(gram_test, axis=0)
    y_norm_sq = gram_test[-1, -1]
    R_train = jsp.linalg.cholesky(gram_train)
    R_test = jsp.linalg.cholesky(gram_test)
    X_train, y_train = R_train[:-1, :-1], R_train[:-1, -1]
    X_test, y_test = R_test[:-1, :-1], R_test[:-1, -1]
    end_gramgen = time.time()
    print(f"Condition Number of gram_train is {jnp.linalg.cond(gram_train)}")
    print(f"Condition Number of gram_test is {jnp.linalg.cond(gram_test)}")
    qr_time = end_gramgen - 2*start_gramgen + start_datagen
    return X_train, X_test, y_train, y_test, theta_true, y_norm_sq, qr_time


if __name__ == "__main__":
    print("Started the experiment.")
    key, gt_key, datakey = random.split(key, 3)
    gt_lssa = ls_spa.LSSPA(gt_key, p, 'argsort', BATCH_SIZE)

    X_train, X_test, y_train, y_test, true_theta, y_norm_sq, qr_time = (
        gen_data(datakey))

    if jnp.any(jnp.isnan(X_train)) or jnp.any(jnp.isnan(X_test)):
        print("Try again")
    else:
        start = time.time()
        gt_results = gt_lssa(X_train, X_test, y_train, y_test, 0.,
                            NUM_BATCHES, eps=1e-2, y_norm_sq=y_norm_sq,
                            return_history=False)
        end = time.time()
        print(f"Initial QR Time: {qr_time}")
        print(f"LSSA Time: {end - start}")
        print(f"Total Time: {end - start + qr_time}")
        print(f"Risk Estimate: {gt_results.overall_error}")
        theta_err = (jnp.linalg.norm(gt_results.theta - true_theta)
                     / jnp.linalg.norm(true_theta))
        print(f"Fit Error: {theta_err}")
        print(f"R^2: {gt_results.r_squared}")
