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
    from matplotlib.ticker import FixedLocator
    import numpy as np
    from scipy.stats.qmc import MultivariateNormalQMC, Sobol

    if not os.path.isdir("./notebooks/data"):
        os.makedirs("./notebooks/data")
    if not os.path.isdir("./notebooks/plots"):
        os.makedirs("./notebooks/plots")

    plt.rcdefaults()

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
        })
    return (
        FixedLocator,
        MultivariateNormalQMC,
        Sobol,
        ls_spa,
        mo,
        np,
        os,
        plt,
        time,
    )


@app.cell
def __(mo):
    in_EXP_NAME = mo.ui.text(value="Medium", label=r"$\textnormal{Experiment Name}:$")
    in_MAX_SAMPLES = mo.ui.number(start=8, stop=20, step=1, value=12,
                                  label=r"$\log_2(\textnormal{Max Number of Samples}):$")
    in_load_gt = mo.ui.switch(value=True)
    in_p = mo.ui.number(start=10, stop=200, step=1, value=100,
                        label=r"$\textnormal{Number of Features}:$")
    in_N = mo.ui.number(start=1e5, stop=5e5, step=1e5,
                     label=r"$\textnormal{Number of Train Observations}:$")
    in_M = mo.ui.number(start=1e5, stop=5e5, step=1e5,
                     label=r"$\textnormal{Number of Test Observations}:$")
    in_STN_RATIO = mo.ui.slider(start=0.0, stop=25.0, value=5.0,
                             label=r"$\textnormal{Signal-to-Noise Ratio}:$")
    in_REG = mo.ui.slider(start=0.0, stop=25.0, value=0.0,
                       label=r"$\textnormal{Ridge Parameter}:$")
    in_conditioning = mo.ui.dropdown({"Low": 0.01, "Medium": 0.5, "High": 20},
                                     label=r"$\textnormal{Covariance Conditioning}:$",
                                     value="Low")

    form = mo.md('''
        **Experiment Config.**

        {exp_name}

        {max_samples}

        Load True Attributions if Saved : {load_gt}

        **Data Generation Config.**

        {p}

        {N}

        {M}

        {STN_RATIO}

        {REG}

        {conditioning}
    ''').batch(
        exp_name=in_EXP_NAME,
        max_samples=in_MAX_SAMPLES,
        load_gt=in_load_gt,
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
        in_MAX_SAMPLES,
        in_N,
        in_REG,
        in_STN_RATIO,
        in_conditioning,
        in_load_gt,
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
    max_samples = 2 ** form.value["max_samples"]
    load_gt = form.value["load_gt"]
    part1 = p / p
    return (
        EXP_NAME,
        M,
        N,
        REG,
        STN_RATIO,
        conditioning,
        load_gt,
        max_samples,
        p,
        part1,
    )


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
def __(EXP_NAME, load_gt, mo, os, part4):
    part5 = part4 + 1
    gt_location = f"./notebooks/data/gt_{EXP_NAME}.npy"
    if os.path.exists(gt_location) and load_gt:
        gt_compute, gt_msg = False, mo.md("Saved ground-truth attributions loaded.")
    else:
        gt_compute, gt_msg = True, mo.md("Computing ground-truth attributions...")

    gt_msg
    return gt_compute, gt_location, gt_msg, part5


@app.cell
def __(
    GeneratorLen,
    X_test,
    X_train,
    gt_compute,
    gt_location,
    ls_spa,
    mo,
    np,
    p,
    part5,
    rng,
    y_test,
    y_train,
):
    part6 = part5 + 1
    if gt_compute:
        gt_permutations_gen = GeneratorLen((rng.permutation(p) for _ in range(2**13)), 2**13)
        gt_permutations = mo.status.progress_bar(gt_permutations_gen)
        gt_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                   perms=gt_permutations, tolerance=0.0)
        gt_attributions = gt_results.attribution
        gt_attributions = gt_attributions * gt_results.r_squared / np.sum(gt_attributions)
        np.save(gt_location, gt_results.attribution)
    else:
        gt_attributions = np.load(gt_location)
    return (
        gt_attributions,
        gt_permutations,
        gt_permutations_gen,
        gt_results,
        part6,
    )


@app.cell
def __(gt_results):
    gt_results.overall_error
    return


@app.cell
def __(mo, part6):
    part7 = part6 + 1
    mo.md("Benchmarking naïve method...")
    return part7,


@app.cell
def __(
    GeneratorLen,
    X_test,
    X_train,
    ls_spa,
    max_samples,
    mo,
    np,
    p,
    part7,
    rng,
    y_test,
    y_train,
):
    part8 = part7 + 1
    naive_permutations_gen = GeneratorLen((rng.permutation(p) for _ in range(2**3)), 2**3)
    naive_permutations = mo.status.progress_bar(naive_permutations_gen)

    def naive_method(batch_size=2**8):
        shapley_values = np.zeros(p)
        attribution_cov = np.zeros((p, p))
        attribution_errors = np.full(p, 0.)
        overall_error = 0.
        error_history = np.zeros(0)
        attribution_history = np.zeros((0, p))

        counter = 0
        for i, perm in enumerate(naive_permutations, 1):
            counter = i
            do_mini_batch = True

            # Compute the lift
            perm = np.array(perm)
            X_train_perm = X_train.copy()[:, perm]
            X_test_perm = X_test.copy()[:, perm]
            lift = np.zeros(p)
            baseline = 0.0
            for j in range(1, p+1):
                theta_fit = np.linalg.lstsq(X_train_perm[:, 0:j], y_train, rcond=None)[0]
                costs = np.sum((X_test_perm[:, 0:j] @ theta_fit - y_test) ** 2)
                R_sq = (np.sum(y_test ** 2) - costs) / np.sum(y_test ** 2)
                lift[perm[j-1]] = R_sq - baseline
                baseline = R_sq

            # Update the mean and biased sample covariance
            attribution_cov = ls_spa.merge_sample_cov(shapley_values, lift,
                                                      attribution_cov, np.zeros((p, p)),
                                                      i-1, 1)
            shapley_values = ls_spa.merge_sample_mean(shapley_values, lift,
                                                      i-1, 1)
            attribution_history = np.vstack((attribution_history, shapley_values))

            # Update the errors
            if (i % batch_size == 0 or i == max_samples - 1):
                unbiased_cov = attribution_cov * i / (i - 1)
                attribution_errors, overall_error = ls_spa.error_estimates(rng,unbiased_cov / i)
                error_history = np.append(error_history, overall_error)
                do_mini_batch = False

        # Last mini-batch
        if do_mini_batch:
            unbiased_cov = attribution_cov * counter / (counter - 1)
            attribution_errors, overall_error = ls_spa.error_estimates(rng, unbiased_cov / i)
            error_history = np.append(error_history, overall_error)

        # Compute auxiliary information
        theta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        r_squared = ((np.linalg.norm(y_test) ** 2
                     - np.linalg.norm(y_test - X_test @ theta) ** 2)
                     / (np.linalg.norm(y_test)**2))

        return ls_spa.ShapleyResults(
            attribution=shapley_values,
            theta=theta,
            overall_error=overall_error,
            error_history=error_history,
            attribution_errors=attribution_errors,
            r_squared=r_squared,
            attribution_history=attribution_history
        )

    naive_results = naive_method()
    naive_attributions = naive_results.attribution
    return (
        naive_attributions,
        naive_method,
        naive_permutations,
        naive_permutations_gen,
        naive_results,
        part8,
    )


@app.cell
def __(mo, part8):
    part9 = part8 + 1
    mo.md("Benchmarking LS-SPA with Monte Carlo sampling, without antithetical sampling...")
    return part9,


@app.cell
def __(
    GeneratorLen,
    X_test,
    X_train,
    ls_spa,
    max_samples,
    mo,
    p,
    part9,
    rng,
    y_test,
    y_train,
):
    part10 = part9 + 1
    mc_samples = (rng.permutation(p) for _ in range(max_samples))
    mc_permutations_gen = GeneratorLen(mc_samples, max_samples)
    mc_permutations = mo.status.progress_bar(mc_permutations_gen)
    mc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                               perms=mc_permutations, tolerance=1e-8,
                               return_attribution_history=True,
                               antithetical=False)
    mc_attributions = mc_results.attribution
    return (
        mc_attributions,
        mc_permutations,
        mc_permutations_gen,
        mc_results,
        mc_samples,
        part10,
    )


@app.cell
def __(mo, part10):
    part11 = part10 + 1
    mo.md("Benchmarking LS-SPA with argsort QMC sampling, without antithetical sampling...")
    return part11,


@app.cell
def __(
    GeneratorLen,
    Sobol,
    X_test,
    X_train,
    argsort_samples,
    ls_spa,
    max_samples,
    mo,
    p,
    part11,
    rng,
    y_test,
    y_train,
):
    part12 = part11 + 1
    argsort_qmc = Sobol(p, seed=rng.choice(1000))
    argsort_qmc_permutations_gen = GeneratorLen((argsort_samples(argsort_qmc, 1).flatten() for _ in range(max_samples)), max_samples)
    argsort_qmc_permutations = mo.status.progress_bar(argsort_qmc_permutations_gen)
    argsort_qmc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                        perms=argsort_qmc_permutations, tolerance=1e-8,
                                        return_attribution_history=True,
                                        antithetical=False)
    argsort_attributions = argsort_qmc_results.attribution
    return (
        argsort_attributions,
        argsort_qmc,
        argsort_qmc_permutations,
        argsort_qmc_permutations_gen,
        argsort_qmc_results,
        part12,
    )


@app.cell
def __(mo, part12):
    part13 = part12 + 1
    mo.md("Benchmarking LS-SPA with permutohedron QMC sampling, without antithetical sampling...")
    return part13,


@app.cell
def __(
    GeneratorLen,
    MultivariateNormalQMC,
    X_test,
    X_train,
    ls_spa,
    max_samples,
    mo,
    np,
    p,
    part13,
    permutohedron_samples,
    rng,
    y_test,
    y_train,
):
    part14 = part13 + 1
    permutohedron_qmc = MultivariateNormalQMC(np.zeros(p-1), seed=rng.choice(1000), 
                                              inv_transform=False)
    permutohedron_qmc_permutations_gen = GeneratorLen((permutohedron_samples(permutohedron_qmc, 1).flatten() for _ in range(max_samples)), max_samples)
    permutohedron_qmc_permutations = mo.status.progress_bar(permutohedron_qmc_permutations_gen)
    permutohedron_qmc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                              perms=permutohedron_qmc_permutations,
                                              tolerance=1e-8,
                                              return_attribution_history=True,
                                              antithetical=False)
    permutohedron_attributions = permutohedron_qmc_results.attribution
    return (
        part14,
        permutohedron_attributions,
        permutohedron_qmc,
        permutohedron_qmc_permutations,
        permutohedron_qmc_permutations_gen,
        permutohedron_qmc_results,
    )


@app.cell
def __(mo, part14):
    part15 = part14 + 1
    mo.md("Benchmarking LS-SPA with Monte Carlo sampling, with antithetical sampling...")
    return part15,


@app.cell
def __(
    GeneratorLen,
    X_test,
    X_train,
    ls_spa,
    max_samples,
    mo,
    p,
    part15,
    rng,
    y_test,
    y_train,
):
    part16 = part15 + 1
    amc_samples = (rng.permutation(p) for _ in range(max_samples // 2))
    amc_permutations_gen = GeneratorLen(amc_samples, max_samples // 2)
    amc_permutations = mo.status.progress_bar(amc_permutations_gen)
    amc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                perms=amc_permutations, tolerance=1e-8,
                                return_attribution_history=True,
                                antithetical=True, batch_size=2**7)
    amc_attributions = amc_results.attribution
    return (
        amc_attributions,
        amc_permutations,
        amc_permutations_gen,
        amc_results,
        amc_samples,
        part16,
    )


@app.cell
def __(mo, part16):
    part17 = part16 + 1
    mo.md("Benchmarking LS-SPA with Monte argsort QMC sampling, with antithetical sampling...")
    return part17,


@app.cell
def __(
    GeneratorLen,
    Sobol,
    X_test,
    X_train,
    argsort_samples,
    ls_spa,
    max_samples,
    mo,
    p,
    part17,
    rng,
    y_test,
    y_train,
):
    part18 = part17 + 1
    aargsort_qmc = Sobol(p, seed=rng.choice(1000))
    aargsort_qmc_permutations_gen = GeneratorLen((argsort_samples(aargsort_qmc, 1).flatten() for _ in range(max_samples//2)), max_samples//2)
    aargsort_qmc_permutations = mo.status.progress_bar(aargsort_qmc_permutations_gen)
    aargsort_qmc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                        perms=aargsort_qmc_permutations, tolerance=1e-8,
                                        return_attribution_history=True,
                                        antithetical=True)
    aargsort_attributions = aargsort_qmc_results.attribution
    return (
        aargsort_attributions,
        aargsort_qmc,
        aargsort_qmc_permutations,
        aargsort_qmc_permutations_gen,
        aargsort_qmc_results,
        part18,
    )


@app.cell
def __(mo, part18):
    part19 = part18 + 1
    mo.md("Benchmarking LS-SPA with permutohedron QMC sampling, with antithetical sampling...")
    return part19,


@app.cell
def __(
    GeneratorLen,
    MultivariateNormalQMC,
    X_test,
    X_train,
    ls_spa,
    max_samples,
    mo,
    np,
    p,
    part19,
    permutohedron_samples,
    rng,
    y_test,
    y_train,
):
    part20 = part19 + 1
    apermutohedron_qmc = MultivariateNormalQMC(np.zeros(p-1), seed=rng.choice(1000), 
                                              inv_transform=False)
    apermutohedron_qmc_permutations_gen = GeneratorLen((permutohedron_samples(apermutohedron_qmc, 1).flatten() for _ in range(max_samples//2)), max_samples//2)
    apermutohedron_qmc_permutations = mo.status.progress_bar(apermutohedron_qmc_permutations_gen)
    apermutohedron_qmc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                              perms=apermutohedron_qmc_permutations,
                                              tolerance=1e-8,
                                              return_attribution_history=True,
                                              antithetical=True)
    apermutohedron_attributions = apermutohedron_qmc_results.attribution
    return (
        apermutohedron_attributions,
        apermutohedron_qmc,
        apermutohedron_qmc_permutations,
        apermutohedron_qmc_permutations_gen,
        apermutohedron_qmc_results,
        part20,
    )


@app.cell
def __(
    aargsort_qmc_results,
    amc_results,
    apermutohedron_qmc_results,
    argsort_qmc_results,
    gt_attributions,
    max_samples,
    mc_results,
    np,
    part20,
    permutohedron_qmc_results,
):
    part21 = part20 + 1
    count = np.arange(1, max_samples + 1)
    acount = np.arange(1, max_samples + 1, 2)

    mc_err = np.linalg.norm(mc_results.attribution_history - gt_attributions, axis=1)
    argsort_err = np.linalg.norm(argsort_qmc_results.attribution_history - gt_attributions, axis=1)
    permutohedron_err = np.linalg.norm(permutohedron_qmc_results.attribution_history - gt_attributions, axis=1)

    amc_err = np.linalg.norm(amc_results.attribution_history - gt_attributions, axis=1)
    aargsort_err = np.linalg.norm(aargsort_qmc_results.attribution_history - gt_attributions, axis=1)
    apermutohedron_err = np.linalg.norm(apermutohedron_qmc_results.attribution_history - gt_attributions, axis=1)
    return (
        aargsort_err,
        acount,
        amc_err,
        apermutohedron_err,
        argsort_err,
        count,
        mc_err,
        part21,
        permutohedron_err,
    )


@app.cell
def __():
    from matplotlib.lines import Line2D
    return Line2D,


@app.cell
def __(mo, part21):
    part22 = part21+1
    mo.md("Plotting the true errors against the total number of permutations sampled...")
    return part22,


@app.cell
def __(
    EXP_NAME,
    Line2D,
    aargsort_err,
    acount,
    amc_err,
    apermutohedron_err,
    argsort_err,
    count,
    mc_err,
    part22,
    permutohedron_err,
    plt,
):
    part23 = part22 + 1
    width_in_inches = 6.3
    original_aspect_ratio = 10 / 6  # Original ratio of width to height you had
    height_in_inches = width_in_inches / original_aspect_ratio * 1.2

    # Specify figsize with the calculated width and height
    compare_fig, compare_ax = plt.subplots(figsize=[width_in_inches, height_in_inches])

    # Define colors for each method
    colors = {
        "MC": "C0",
        "Permutohedron QMC": "C1",
        "Argsort QMC": "C2"
    }

    # Plot original sampling methods
    compare_ax.loglog(count, mc_err, label="MC", color=colors["MC"])
    compare_ax.loglog(count, permutohedron_err, label="Permutohedron QMC", color=colors["Permutohedron QMC"])
    compare_ax.loglog(count, argsort_err, label="Argsort QMC", color=colors["Argsort QMC"])

    # Plot antithetical sampling methods with dashed lines and matching colors
    compare_ax.loglog(acount, amc_err, linestyle="dotted", color=colors["MC"])
    compare_ax.loglog(acount, apermutohedron_err, linestyle="dotted", color=colors["Permutohedron QMC"])
    compare_ax.loglog(acount, aargsort_err, linestyle="dotted", color=colors["Argsort QMC"])

    compare_ax.set_xscale("log", base=2)
    compare_ax.set_yscale("log", base=10)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color=colors["MC"], label='Monte Carlo (MC)'),
        Line2D([0], [0], color=colors["Permutohedron QMC"], label='Permutohedron QMC'),
        Line2D([0], [0], color=colors["Argsort QMC"], label='Argsort QMC'),
        Line2D([0], [0], color='k', linestyle='dotted', label='With Antithetical Samples')
    ]
    # Adjusting the legend's position to the right of the plot
    compare_ax.legend(handles=legend_elements, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Adjust layout and spacing to accommodate the legend outside the plot
    plt.tight_layout(pad=1.0, h_pad=1.0)
    plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin to make room for the legend

    plt.xlabel("Total Number of Samples", fontsize=14)
    plt.ylabel("Error, $\\|S - \hat S\\|_2$", fontsize=14)
    plt.grid(True, which="both", linestyle="--", color="gray",
             linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'./notebooks/plots/err_vs_numsamples_{EXP_NAME}.pdf', format='pdf')
    plt.gca()
    return (
        colors,
        compare_ax,
        compare_fig,
        height_in_inches,
        legend_elements,
        original_aspect_ratio,
        part23,
        width_in_inches,
    )


@app.cell
def __(mo, part23):
    part24 = part23 + 1
    mo.md("Plotting the true and estimated errors for argsort QMC, without antithetical sampling, against the total number of permutations sampled...")
    return part24,


@app.cell
def __(
    EXP_NAME,
    FixedLocator,
    argsort_err,
    argsort_qmc_results,
    count,
    height_in_inches,
    max_samples,
    np,
    part24,
    plt,
    width_in_inches,
):
    part25 = part24 + 1
    err_fig, err_ax = plt.subplots(figsize=[width_in_inches, height_in_inches])

    err_ax.loglog(count[2**8:], argsort_err[2**8:], label="True Error, $\|S -\hat S\|_2$")
    err_ax.loglog(np.arange(1, max_samples // (2**8) + 1) * 2**8, argsort_qmc_results.error_history, label=r"Estimated Error, $\hat\sigma$")

    err_ax.set_xscale("log", base=2)
    err_ax.set_yscale("log", base=10)

    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.xlabel("Number of Samples", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.grid(True, which="both", linestyle="--", color="gray",
             linewidth=0.5, alpha=0.6)
    current_minor_ticks = err_ax.yaxis.get_minor_locator().tick_values(err_ax.get_ylim()[0], err_ax.get_ylim()[1])
    updated_minor_ticks = list(current_minor_ticks) + [5e-3]

    err_ax.yaxis.set_minor_locator(FixedLocator(updated_minor_ticks))

    def custom_formatter(x, pos):
        if x == 5e-3:
            return r'$5\times 10^{-3}$'
        else:
            return ''

    err_ax.yaxis.set_minor_formatter(custom_formatter)

    plt.tight_layout()
    plt.savefig(f'./notebooks/plots/est_error_argsort_{EXP_NAME}.pdf', format='pdf')
    plt.gca()
    return (
        current_minor_ticks,
        custom_formatter,
        err_ax,
        err_fig,
        part25,
        updated_minor_ticks,
    )


@app.cell
def __(mo, part25):
    part26 = part25 + 1
    mo.md("Plotting the true and estimated errors for Monte Carlo, with antithetical sampling, against the total number of permutations sampled...")
    return part26,


@app.cell
def __(
    EXP_NAME,
    FixedLocator,
    amc_err,
    amc_results,
    count,
    custom_formatter,
    height_in_inches,
    max_samples,
    np,
    part26,
    plt,
    width_in_inches,
):
    part27 = part26 + 1
    aerr_fig, aerr_ax = plt.subplots(figsize=[width_in_inches, height_in_inches])

    aerr_ax.loglog(count[2**8::2], amc_err[2**7:], label="True Error, $\|S -\hat S\|_2$")
    aerr_ax.loglog(np.arange(1, max_samples // (2**8) + 1) * 2**8, amc_results.error_history, label=r"Estimated Error, $\hat\sigma$")

    aerr_ax.set_xscale("log", base=2)
    aerr_ax.set_yscale("log", base=10)

    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.xlabel("Number of Samples", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.grid(True, which="both", linestyle="--", color="gray",
             linewidth=0.5, alpha=0.6)
    acurrent_minor_ticks = aerr_ax.yaxis.get_minor_locator().tick_values(aerr_ax.get_ylim()[0], aerr_ax.get_ylim()[1])
    aupdated_minor_ticks = list(acurrent_minor_ticks) + [5e-3]

    aerr_ax.yaxis.set_minor_locator(FixedLocator(aupdated_minor_ticks))
    aerr_ax.yaxis.set_minor_formatter(custom_formatter)

    plt.tight_layout()
    plt.savefig(f'./notebooks/plots/est_error_antithetical_mc_{EXP_NAME}.pdf', format='pdf')
    plt.gca()
    return (
        acurrent_minor_ticks,
        aerr_ax,
        aerr_fig,
        aupdated_minor_ticks,
        part27,
    )


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
    return (
        AlternatingGenerator,
        GeneratorLen,
        argsort_samples,
        permutohedron_samples,
    )


if __name__ == "__main__":
    app.run()
