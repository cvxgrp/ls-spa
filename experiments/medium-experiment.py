import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from matplotlib.ticker import FixedLocator

import ls_spa

## reset defaults
plt.rcdefaults()

## Set up LaTeX fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    })

# CONFIG
EXP_NAME = "px5p"

p = 100 # Feature counts to test
N = int(1e5) # Number of train observations
M = int(1e5) # Number of test observations
STN_RATIO = 5.0 # Signal-to-noise ratio

GT_NUM_BATCHES = 2 ** 15
EXP_NUM_BATCHES = 2 ** 5
BATCH_SIZE_GT = 2 ** 13
BATCH_SIZE_EXP = 2 ** 8

REG = 0.

LOAD_GT = False

key = random.PRNGKey(1)

def gen_data(key):
    # We want to generate a covariance matrix so some features have very large
    # covariances (in magnitude).
    key, subkey, subkey_shuffle = random.split(key, 3)
    A = random.normal(subkey, (p, p//20))
    cov = A @ A.T + jnp.eye(p)
    v = jnp.sqrt(jnp.diag(cov))
    cov = cov / jnp.outer(v, v)
    print(f'Max cov: {jnp.max(jnp.abs(cov - jnp.diag(jnp.diag(cov))))}')
    print(f'Cov cond. number: {jnp.linalg.cond(cov)}')

    # We sample observations to create X_train and X_test.
    key, train_key, test_key = random.split(key, 3)
    X_train = random.multivariate_normal(
        train_key, jnp.zeros(p), cov, (N,), method='svd'
    )
    X_test = random.multivariate_normal(
        test_key, jnp.zeros(p), cov, (M,), method='svd'
    )

    # We want about most of the features to be irrelevant.
    theta_vals = jnp.full((p+1)//10, 2)
    padded_theta_vals = jnp.pad(theta_vals, (0, p - (p+1)//10))
    theta_true = random.permutation(subkey_shuffle, padded_theta_vals)

    # We create the response variables and add a little noise.
    key, train_noise_key, test_noise_key = random.split(key, 3)
    std = jnp.sqrt(jnp.sum(jnp.diag(cov) * theta_true**2) / STN_RATIO)
    y_train = X_train @ theta_true + std * random.normal(train_noise_key, (N,))

    X_train_mean = jnp.mean(X_train, axis=0, keepdims=True)
    X_train = X_train - X_train_mean
    y_train_mean = jnp.mean(y_train)
    y_train = y_train - y_train_mean

    y_test = X_test @ theta_true + std * random.normal(test_noise_key, (M,))
    X_test = X_test - X_train_mean
    y_test = y_test - y_train_mean

    return X_train, X_test, y_train, y_test, theta_true


if __name__ == "__main__":
    key, gt_key, mc_key, ph_key, as_key = random.split(key, 5)
    gt_lssa = ls_spa.LSSPA(gt_key, p, 'argsort', BATCH_SIZE_GT)
    mc_lssa = ls_spa.LSSPA(mc_key, p, batch_size=BATCH_SIZE_EXP)
    ph_lssa = ls_spa.LSSPA(ph_key, p, 'permutohedron', BATCH_SIZE_EXP)
    as_lssa = ls_spa.LSSPA(as_key, p, 'argsort', BATCH_SIZE_EXP)

    key, datakey = random.split(key)
    X_train, X_test, y_train, y_test, true_theta = gen_data(datakey)
    print(f"Condition Number of X_train is "
          f"{jnp.linalg.cond(X_train.T @ X_train)}")
    print(f"Condition Number of X_test is "
          f"{jnp.linalg.cond(X_test.T @ X_test)}")
    if not LOAD_GT:
        gt_results = gt_lssa(X_train, X_test, y_train, y_test, REG,
                            GT_NUM_BATCHES, eps=0.0, return_history=False)
        print(f"Estimated GT error: {gt_results.overall_error}")
        jnp.save(f'data/gt_atts_{EXP_NAME}.npy', gt_results.attribution)
    else:
        gt_results = gt_lssa(X_train, X_test, y_train, y_test, REG,
                            1, eps=0.0, return_history=False)
        gt_results.attribution = jnp.load(f'data/gt_atts_{EXP_NAME}.npy')

    mc_start = time.time()
    mc_results = mc_lssa(X_train, X_test, y_train, y_test, REG,
                         EXP_NUM_BATCHES, eps=0.0)
    ph_start = time.time()
    ph_results = ph_lssa(X_train, X_test, y_train, y_test, REG,
                         EXP_NUM_BATCHES, eps=0.0)
    as_start = time.time()
    as_results = as_lssa(X_train, X_test, y_train, y_test, REG,
                         EXP_NUM_BATCHES, eps=0.0)
    end_time = time.time()
    print(f"R-squared: {as_results.r_squared}")

    print(f"Monte Carlo time: {ph_start - mc_start} seconds")
    print(f"Permutohedron time: {as_start - ph_start} seconds")
    print(f"Argsort time: {end_time - as_start} seconds")
    print(f"Argsort Max Est Err: {jnp.max(as_results.attribution_errors)}")

    count = jnp.arange(1, EXP_NUM_BATCHES * BATCH_SIZE_EXP + 1)
    den = count[:, None]
    mc_running_est = jnp.cumsum(mc_results.attribution_history, axis=0) / den
    ph_running_est = jnp.cumsum(ph_results.attribution_history, axis=0) / den
    as_running_est = jnp.cumsum(as_results.attribution_history, axis=0) / den

    mc_diff = jnp.linalg.norm(mc_running_est - gt_results.attribution, axis=1)
    ph_diff = jnp.linalg.norm(ph_running_est - gt_results.attribution, axis=1)
    as_diff = jnp.linalg.norm(as_running_est - gt_results.attribution, axis=1)

    mc_running_err = mc_diff
    ph_running_err = ph_diff
    as_running_err = as_diff

    plt.plot(jnp.abs(true_theta), label="True")
    plt.plot(jnp.abs(gt_results.theta), label="Fitted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plots/theta_star_{EXP_NAME}.pdf")

    # Create the plot
    fig, ax = plt.subplots(figsize=[10, 6])

    # Plot the Monte Carlo data
    ax.loglog(count, mc_running_err, label='MC')

    # Plot the Permutohedron QMC data
    ax.loglog(count, ph_running_err, label='Permutohedron QMC')

    # Plot the Argsort QMC data
    ax.loglog(count, as_running_err, label='Argsort QMC')

    # Change the base of the logarithm
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)

    # Add details to the plot
    plt.legend(fontsize=12)
    plt.xlabel('Number of Samples', fontsize=14)
    plt.ylabel('Error, $\|S - \hat S\|_2$', fontsize=14)
    plt.grid(True, which='both', linestyle='--', color='gray',
             linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    plt.savefig(f'./plots/err_vs_numsamples_{EXP_NAME}.pdf', format='pdf')

    # Create the plot.
    fig, ax = plt.subplots(figsize=[8, 6])

    # Plot the STDs.
    idx = int(jnp.argwhere(count == BATCH_SIZE_EXP - 1))
    ax.loglog(count[idx:], as_running_err[idx:], label='True Error, $\|S - \hat S\|_2$')
    ax.loglog(jnp.arange(1, EXP_NUM_BATCHES+1) * BATCH_SIZE_EXP,
                as_results.error_history, label=r'Estimated Error, $\|\hat S - \tilde S\|_2$')

    # Change the base of the logarithm
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)

    # Add details to the plot
    ax.legend(fontsize=12)
    ax.set_xlabel('Number of Samples', fontsize=14)
    ax.set_ylabel('Error', fontsize=14)

    ax.grid(True, which='both', linestyle='--', color='gray',
            linewidth=0.5, alpha=0.6)
    # Get current minor ticks and append the new one
    current_minor_ticks = ax.yaxis.get_minor_locator().tick_values(ax.get_ylim()[0], ax.get_ylim()[1])
    updated_minor_ticks = list(current_minor_ticks) + [5e-3]

    # Set the updated minor tick locations
    ax.yaxis.set_minor_locator(FixedLocator(updated_minor_ticks))

    # Create custom formatter to add the label for 5e-3 and leave others empty
    def custom_formatter(x, pos):
        if x == 5e-3:
            return r'$5\times 10^{-3}$'
        else:
            return ''

    ax.yaxis.set_minor_formatter(custom_formatter)

    plt.tight_layout()
    plt.savefig(f'./plots/est_err_vs_numsamples_{EXP_NAME}.pdf', format='pdf')
