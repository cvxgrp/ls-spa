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

`ls_spa` has the following dependencies:
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

## Hello world

We present a complete Python script that utilizes LS-SPA to compute
the Shapley attribution on the data from the toy example described 
in the companion paper.

```
# Imports
import numpy as np
from ls_spa import ls_spa

# Data loading
X_train = np.load("./data/X_train.npy")
X_test = np.load("./data/X_test.npy")
y_train = np.load("./data/y_train.npy")
y_test = np.load("./data/y_test.npy")

# Compute Shapley attribution with LS-SPA
S = np.array(ls_spa(X_train, X_test, y_train, y_test).attribution

# Print attribution
print("LS-SPA Shapley attribution: {}".format(S))
```

This simple example uses the data included in the data directory of this
repository.

## Advanced usage

The `ls_spa` function computes Shapley values for the given data using
the LS-SPA method described in the companion paper. It takes arguments:

- `X_train`: Training feature matrix.
- `X_test`: Testing feature matrix.
- `y_train`: Training response vector.
- `y_test`: Testing response vector.
- `reg`: Regularization parameter (Default 0).
- `method`: Permutation sampling method. Options include `'random'`, 
  `'permutohedron'`, `'argsort'`, and `'exact'`. If `None`, `'argsort'` is used 
  if the number of features is greater than 10; otherwise, `'exact'` is used.
- `batch_size`: Number of permutations in each batch (Default `2**7`).
- `num_batches`: Maximum number of batches (Default `2**7`).
- `tolerance`: Convergence tolerance for the Shapley values (Default `1e-2`).
- `seed`: Seed for random number generation (Default `42`).

`ls_spa` returns a `ShapleyResults` object. The `ShapleyResults` object
has the fields:
- `attribution`: Array of Shapley values for each feature.
- `attribution_history`: Array of Shapley values for each iteration. 
  `None` if `return_history=False` in `LSSA` call.
- `theta`: Array of regression coefficients.
- `overall_error`: Mean absolute error of the Shapley values.
- `error_history`: Array of mean absolute errors for each iteration. 
  `None` if `return_history=False` in `LSSA` call.
- `attribution_errors`: Array of absolute errors for each feature.
- `r_squared`: R-squared statistic of the regression.

## Example notebook

A more extensive example usage of LS-SPA can be found in 
this [notebook](./shapley_toy.ipynb).

