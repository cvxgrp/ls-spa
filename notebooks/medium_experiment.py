import marimo

__generated_with = "0.3.1"
app = marimo.App()


@app.cell
def __():
    import time

    import ls_spa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    return ls_spa, mo, np, plt, time


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
                                     value="High")
    return (
        in_EXP_NAME,
        in_M,
        in_N,
        in_REG,
        in_STN_RATIO,
        in_conditioning,
        in_p,
    )


@app.cell
def __(
    in_EXP_NAME,
    in_M,
    in_N,
    in_REG,
    in_STN_RATIO,
    in_conditioning,
    in_p,
    mo,
):
    mo.vstack([in_EXP_NAME, in_p, in_N, in_M, in_STN_RATIO, in_REG, in_conditioning])
    return


@app.cell
def __(
    in_EXP_NAME,
    in_M,
    in_N,
    in_REG,
    in_STN_RATIO,
    in_conditioning,
    in_p,
):
    EXP_NAME = in_EXP_NAME.value
    p = int(in_p.value)
    N = int(in_N.value)
    M = int(in_M.value)
    STN_RATIO = in_STN_RATIO.value
    REG = in_REG.value
    conditioning = in_conditioning.value
    return EXP_NAME, M, N, REG, STN_RATIO, conditioning, p


@app.cell
def __(gen_data, mo, np):
    rng = np.random.default_rng(364)
    X_train, X_test, y_train, y_test, true_theta, cov = gen_data(rng)
    max_covariance = np.max(np.abs(cov - np.diag(np.diag(cov))))
    cond_number = np.linalg.cond(cov)
    mo.md(f"The maximum feature covariance is {max_covariance:.2e}, and the condition number of the feature covariance matrix is {cond_number:.2e}.")
    return (
        X_test,
        X_train,
        cond_number,
        cov,
        max_covariance,
        rng,
        true_theta,
        y_test,
        y_train,
    )


app._unparsable_cell(
    r"""
    key, gt_key, mc_key, ph_key, as_key = random.split(key, 5)
        gt_lssa = ls_spa.LSSPA(gt_key, p, 'argsort', BATCH_SIZE_GT)
        mc_lssa = ls_spa.LSSPA(mc_key, p, batch_size=BATCH_SIZE_EXP)
        ph_lssa = ls_spa.LSSPA(ph_key, p, 'permutohedron', BATCH_SIZE_EXP)
        as_lssa = ls_spa.LSSPA(as_key, p, 'argsort', BATCH_SIZE_EXP)

        key, datakey = random.split(key)
        X_train, X_test, y_train, y_test, true_theta = gen_data(datakey)
        print(f\"Condition Number of X_train is \"
              f\"{jnp.linalg.cond(X_train.T @ X_train)}\")
        print(f\"Condition Number of X_test is \"
              f\"{jnp.linalg.cond(X_test.T @ X_test)}\")
        if not LOAD_GT:
            gt_results = gt_lssa(X_train, X_test, y_train, y_test, REG,
                                GT_NUM_BATCHES, eps=0.0, return_history=False)
            print(f\"Estimated GT error: {gt_results.overall_error}\")
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
    """,
    name="__"
)


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


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


if __name__ == "__main__":
    app.run()
