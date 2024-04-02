import marimo

__generated_with = "0.3.1"
app = marimo.App()


@app.cell
def __():
    import os
    import time

    import ls_spa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats.qmc import MultivariateNormalQMC, Sobol

    if not os.path.isdir("./notebooks/data"):
        os.makedirs("./notebooks/data")
    if not os.path.isdir("./notebooks/plots"):
        os.makedirs("./notebooks/plots")
    return MultivariateNormalQMC, Sobol, ls_spa, mo, np, os, plt, time


@app.cell
def __(mo):
    in_EXP_NAME = mo.ui.text(value="Medium", label="Experiment Name")
    in_p = mo.ui.number(start=1, stop=200, step=1, value=100,
                        label="Number of Features")
    in_N = mo.ui.number(start=1e5, stop=5e5, step=1e5,
                     label="Number of Train Observations")
    in_M = mo.ui.number(start=1e5, stop=5e5, step=1e5,
                     label="Number of Test Observations")
    in_STN_RATIO = mo.ui.slider(start=0.0, stop=25.0, value=5.0,
                             label="Signal-to-Noise Ratio")
    in_REG = mo.ui.slider(start=0.0, stop=25.0, value=0.0,
                       label="Ridge Parameter")
    in_conditioning = mo.ui.dropdown({"Low": 0.01, "Medium": 0.5, "High": 20},
                                     label="Covariance Conditioning",
                                     value="Low")

    form = mo.md('''
        **Config.**

        {exp_name}

        {p}

        {N}

        {M}

        {STN_RATIO}

        {REG}

        {conditioning}
    ''').batch(
        exp_name=in_EXP_NAME,
        p=in_p,
        N=in_N,
        M=in_M,
        STN_RATIO=in_STN_RATIO,
        REG=in_REG,
        conditioning=in_conditioning
    ).form(show_clear_button=True, bordered=False)
    return (
        form,
        in_EXP_NAME,
        in_M,
        in_N,
        in_REG,
        in_STN_RATIO,
        in_conditioning,
        in_p,
    )


@app.cell
def __(form):
    form
    return


@app.cell
def __(form):
    EXP_NAME = form.value["exp_name"]
    p = int(form.value["p"])
    N = int(form.value["N"])
    M = int(form.value["M"])
    STN_RATIO = form.value["STN_RATIO"]
    REG = form.value["REG"]
    conditioning = form.value["conditioning"]
    part1 = p / p
    return EXP_NAME, M, N, REG, STN_RATIO, conditioning, p, part1


@app.cell
def __(mo, np, part1):
    part2 = part1 + 1
    rng = np.random.default_rng(42)
    mo.md("Generating data...")
    return part2, rng


@app.cell
def __(gen_data, mo, part2, rng):
    part3 = part2 + 1
    X_train, X_test, y_train, y_test, true_theta, cov = gen_data(rng)
    mo.md("Data generation complete.")
    return X_test, X_train, cov, part3, true_theta, y_test, y_train


@app.cell
def __(cov, mo, np, part3):
    part4 = part3 + 1
    max_covariance = np.max(np.abs(cov - np.diag(np.diag(cov))))
    cond_number = np.linalg.cond(cov)
    mo.md(f"The maximum feature covariance is {max_covariance:.2e}, and the condition number of the feature covariance matrix is {cond_number:.2e}.")
    return cond_number, max_covariance, part4


@app.cell
def __(EXP_NAME, mo, np, os, part4):
    part5 = part4 + 1
    gt_location = f"./notebooks/data/gt_{EXP_NAME}.npy"
    if os.path.exists(gt_location):
        gt_attributions = np.load(gt_location)
        gt_compute, gt_msg = False, mo.md("Saved ground-truth attributions loaded.")
    else:
        gt_compute, gt_msg = True, mo.md("No saved ground-truth attributions. Computing ground-truth attributions...")

    gt_msg
    return gt_attributions, gt_compute, gt_location, gt_msg, part5


@app.cell
def __(
    Sobol,
    X_test,
    X_train,
    argsort_samples,
    gt_compute,
    ls_spa,
    p,
    part5,
    rng,
    y_test,
    y_train,
):
    part6 = part5 + 1
    if gt_compute:
        gt_qmc = Sobol(p, seed=rng.choice(1000))
        gt_permutations = argsort_samples(gt_qmc, 2**12)
        # gt_permutations = (argsort_samples(gt_qmc, 1) for _ in range(2**12))
        gt_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                  max_samples=2**100)
    return gt_permutations, gt_qmc, gt_results, part6


@app.cell
def __(M, N, STN_RATIO, conditioning, np, p):
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
    return gen_data,


@app.cell
def __(np, p):
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
    return argsort_samples, permutohedron_samples


if __name__ == "__main__":
    app.run()
