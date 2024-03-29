import marimo

__generated_with = "0.3.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # LS-SPA Demonstration Notebook
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"### In this notebook, we use the data from the toy example in Section 2.5 of the paper \"Efficient Shapley Performance Attribution for Least-Squares Regression\" to demonstrate how Shapley values can be computed directly for a linear, least-squares model. We then demonstrate how LS-SPA can be used to generate the same Shapley attribution. In this specific case, we have a very small number of features, so it is feaible to compute the exact Shapley attribution. When the number of features exceeds 15, this is no longer the case. LS-SPA is able to accurately approximate Shapley attributions for linear least-squares models even when the number of features exceeds 1000.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Imports
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import math
    import itertools
    import matplotlib.pyplot as plt

    from ls_spa import ls_spa
    return itertools, ls_spa, math, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Data loading
        """
    )
    return


@app.cell
def _():
    N = 50
    M = 50
    p = 3
    return M, N, p


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The rows of $X$ correspond to observations and the columns of $X$ correspond to features. We fit a least-squares model on the training data `X_train` and `y_train` and evaluate its performance on the test data `X_test` and `y_test`.
        """
    )
    return


@app.cell
def _(np):
    X_train, X_test, y_train, y_test = [np.load("./data/toy_data.npz")[key] for key in ["X_train","X_test","y_train","y_test"]]
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Direct computation of lifts and $R^2$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### We compute the out-of-sample $R^2$ for a least-squares model fitted on each subset of our features.
        """
    )
    return


@app.cell
def _(X_test, X_train, itertools, np, p, y_test, y_train):
    R2 = np.zeros((tuple(2 for _ in range(p))))

    for _n in range(2 ** p):
        _mask = [_n // (2 ** i) % 2 for i in range(p)]
        _indices = list(itertools.compress(range(p), _mask))

        X_train_sel = X_train[:,_indices]
        X_test_sel = X_test[:,_indices]
        theta = np.linalg.lstsq(X_train_sel, y_train, rcond=1e-5)[0]
        R2[*_mask] = (np.linalg.norm(y_test) ** 2 - np.linalg.norm((X_test_sel @ theta) - y_test) ** 2) / (np.linalg.norm(y_test) ** 2)

    R2 = np.around(R2, 2)
    return R2, X_test_sel, X_train_sel, theta


@app.cell
def _(mo):
    mo.md(
        r"""
        ### For every ordering of our features, we remove one from our model and re-fit sequentially. For each feature, we consider the change in the $R^2$ of the model due to its addition/removal. For a single ordering, the vector of these performance differences due to each feature is a lift vector. The Shapley attribution of our model is the average of the lift vectors for every possible ordering of the features.
        """
    )
    return


@app.cell
def _(R2, itertools, math, np, p):
    lifts = np.zeros((math.factorial(p), p))
    perms =  list(itertools.permutations(range(p)))

    for _i, _perm in enumerate(perms):
        inds = [0,0,0]
        perf = R2[*inds]
        for lift in _perm:
            inds[lift] = 1
            lifts[_i,lift] =  R2[*inds] - perf
            perf = R2[*inds]

    attrs = np.around(np.mean(lifts, axis=0), 2)
    return attrs, inds, lift, lifts, perf, perms


@app.cell
def _(mo):
    mo.md(
        r"""
        ### We display the $R^2$ for the model fitted with each subset of the features, and we also display the lift vectors corresponding to each permutation of the features.
        """
    )
    return


@app.cell
def _(R2, itertools, p):
    print("{: ^8}| {}".format("Subset", "R^2"))
    print("----------------")
    for _n in range(2 ** p):
        _mask = [_n // (2 ** i) % 2 for i in range(p)]
        indices = list(itertools.compress(range(p), _mask))
        S = "{" + "".join("{},".format(idx + 1) for idx in indices)[:-1] + "}"

        print("{: ^8}| {}".format(S, R2[*_mask]))
    return S, indices


@app.cell
def _(lifts, perms):
    print("{: <12}| {}".format("Permutation", "Lift vector"))
    print("-----------------------------")
    for _i, _perm in enumerate(perms):
        pi = "(" + "".join("{},".format(_p+1) for _p in _perm)[:-1] + ")"
        print("{: ^12}| {}".format(pi, lifts[_i]))
    return pi,


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Comparison of true Shapley attribution and LS-SPA
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### We use LS-SPA to estimate (in one line) the Shapley attribution we computed exactly. We show both for comparison.
        """
    )
    return


@app.cell
def _(X_test, X_train, attrs, ls_spa, np, y_test, y_train):
    results = ls_spa(X_train, X_test, y_train, y_test)
    ls_spa_attrs = np.around(np.array(results.attribution), 2)
    print("Explicit Shapley attribution: {}".format(attrs))
    print("LS-SPA Shapley attribution:   {}".format(ls_spa_attrs))
    return ls_spa_attrs, results


@app.cell
def _(mo):
    mo.md(
        """
        ### We can also print the ShapleyResults object returned by `ls_spa` to see a useful dashboard about the computed Shapley attribution.
        """
    )
    return


@app.cell
def _(results):
    print(results)
    return


if __name__ == "__main__":
    app.run()
