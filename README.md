# Least-Squares Shapley Performance Attribution (LS-SPA)

Library companion to the paper [Efficient Shapley Performance Attribution for Least-Squares
Regression](https://web.stanford.edu/~boyd/papers/ls_shapley.html) by Logan Bell, 
Nikhil Devanathan, and Stephen Boyd.

## Installation

This library is not on PyPI, so it cannot be installed with `pip`.
Instead execute
```
git clone https://github.com/cvxgrp/ls-shapley.git
```
to clone this repository and download `ls_spa.py`. if you would like to use
LS-SPA in a Python project, copy `ls_spa.py` to the directory of your python
file and add 
```
from ls_spa import ls_spa
```
to the top of your python file.

`lsspa` has the following dependencies:
- `numpy`
- `scipy`
- `pandas`
- `jax`
Optional dependencies are
- `matplotlib` for plotting
- `jupyter-notebook` for using the demo notebook

`ls_spa` also requires [JAX](https://github.com/google/jax). 
JAX installation varies by platform so please follow 
[these instructions](https://github.com/google/jax#installation)
to correctly install JAX.

## Usage

We assume that you have imported LS-SPA and you have a $p\times N$
matrix of training data `X_train`, a $p\times M$ matrix of testing data `X_tst`,
a $N$ vector of training labels `y_train`, and a $M$ vector of testing labels `y_test`.
In this case, you can find the Shapley attribution of the test $R^2$ on your data by
executing

```
S = np.array(ls_spa(X, X_tst, y, y_tst).attribution
```
`S` will be a JAX vector containing the Shapley values of your features.


## Example notebokk

An example usage of LS-SPA can be found in this [notebook](./shapley_toy.ipynb).

