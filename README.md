# Least-Squares Shapley Performance Attribution (LS-SPA)

### [Installation](#Installation) - [Usage](#Usage) - [Hello world](#Hello-world) - [Example notebook](#Example-notebook) - [Optional arguments](#Optional-arguments)

Library companion to the paper [Efficient Shapley Performance Attribution for Least-Squares
Regression](https://web.stanford.edu/~boyd/papers/ls_shapley.html) by Logan Bell, 
Nikhil Devanathan, and Stephen Boyd.

This is a pre-release version of the code, and as such there may be significant tweaks and updates in the near future.

## Installation

This library is not yet packaged, so it cannot be installed with `pip`.
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

We assume that you have imported `ls_spa` and you have a $N\times p$
matrix of training data `X_train`, a $M\times p$ matrix of testing data `X_test`,
a $N$ vector of training labels `y_train`, and a $M$ vector of testing labels `y_test`
for positive integers $p, N, M$. In this case, you can find the Shapley attribution 
of the out-of-sample $R^2$ on your data by executing

```
attrs = ls_spa(X_train, X_test, y_train, y_test).attribution
```

`attrs` will be a JAX vector containing the Shapley values of your features.
The `ls_spa` function computes Shapley values for the given data using
the LS-SPA method described in the companion paper. It takes arguments:

- `X_train`: Training feature matrix.
- `X_test`: Testing feature matrix.
- `y_train`: Training response vector.
- `y_test`: Testing response vector.

## Hello world

We present a complete Python script that utilizes LS-SPA to compute
the Shapley attribution on the data from the toy example described 
in the companion paper.

```
# Imports
import numpy as np
from ls_spa import ls_spa

# Data loading
X_train, X_test, y_train, y_test = [np.load("./data/toy_data.npz")[key] for key in ["X_train","X_test","y_train","y_test"]]

# Compute Shapley attribution with LS-SPA
attrs = ls_spa(X_train, X_test, y_train, y_test).attribution

# Print attribution
print("LS-SPA Shapley attribution: {}".format(attrs))
```

This simple example uses the data included in the data directory of this
repository.

## Example notebook

In this [notebook](./shapley_toy.ipynb), we walk through the process of 
computing Shapley values on the data for the toy example in the 
companion paper. We then use `ls_spa` to compute the Shapley attribution
on the same data.

## Optional arguments
`ls_spa` takes the optional arguments:
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
