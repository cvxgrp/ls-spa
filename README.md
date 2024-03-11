# Least-Squares Shapley Performance Attribution (LS-SPA)

### [Installation](#Installation) - [Usage](#Usage) - [Hello world](#Hello-world) - [Example notebook](#Example-notebook) - [Optional arguments](#Optional-arguments) - [Citing](#Citing)

Library companion to the paper [Efficient Shapley Performance Attribution for Least-Squares
Regression](https://web.stanford.edu/~boyd/papers/ls_shapley.html) by Logan Bell,
Nikhil Devanathan, and Stephen Boyd.

The results provided in the reference paper were generated using a more performant, but
harder to use implementation of the same algorithm. This benchmark code and the numerical 
experiments from the reference paper can be found at 
[cvxgrp/ls-spa-benchmark](https://github.com/cvxgrp/ls-spa-benchmark). We recommend
caution in trying to use the benchmark code.

## Installation

To install this package, execute
```
pip install git+https://github.com/cvxgrp/ls-spa
```

Import `ls_spa` by adding
```
from ls_spa import ls_spa
```
to the top of your Python file.

`ls_spa` has the following dependencies:
- `numpy`
- `scipy`
- `pandas`

Optional dependencies are
- `marimo` for using the demo notebook

## Usage

We assume that you have imported `ls_spa` and you have a $N\times p$
matrix of training data `X_train`, a $M\times p$ matrix of testing data `X_test`,
a $N$ vector of training labels `y_train`, and a $M$ vector of testing labels `y_test`
for positive integers $p, N, M$ with $N,M\geq p$. In this case, you can find the
Shapley attribution of the out-of-sample $R^2$ on your data by executing

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
results = ls_spa(X_train, X_test, y_train, y_test)

# Print attribution
print(results)
```
This example uses data from the `data`
directory of this repository.

The line `print(results)` prints a dashboard of information generated while
computing the Shapley attribution such as the attribution, the $R^2$ of the
model fitted with all of the features, the feature cofficients of the fitted
model, and an error estimate on the attribution (since LS-SPA is a method
of estimation).

To extract just the vector of Shapley values, use `results.attribution`.
For more info, see [optional arguments](#Optional-arguments).

## Example notebook

In this [demo](./notebooks/shapley_toy.py), we walk through the process of
computing Shapley values on the data for the toy example in the
companion paper. We then use `ls_spa` to compute the Shapley attribution
on the same data.

## Optional arguments
`ls_spa` takes the optional arguments:
- `reg`: Regularization parameter (Default `0`).
- `method`: Permutation sampling method. Options include `'random'`,
  `'permutohedron'`, `'argsort'`, and `'exact'`. If `None`, `'argsort'` is used
  if the number of features is greater than 10; otherwise, `'exact'` is used.
- `batch_size`: Number of permutations in each batch (Default `2**7`).
- `num_batches`: Maximum number of batches (Default `2**7`).
- `tolerance`: Convergence tolerance for the Shapley values (Default `1e-2`).
- `seed`: Seed for random number generation (Default `42`).
- `return_history`: Flag to determine whether to return the history of error estimates and attributions for each feature chain (Default `False`).

`ls_spa` returns a `ShapleyResults` object. The `ShapleyResults` object
has the fields:
- `attribution`: Array of Shapley values for each feature.
- `attribution_history`: Array of Shapley values for each iteration.
  `None` if `return_history=False` in `ls_spa` call.
- `theta`: Array of regression coefficients.
- `overall_error`: Mean absolute error of the Shapley values.
- `error_history`: Array of mean absolute errors for each iteration.
  `None` if `return_history=False` in `ls_spa` call.
- `attribution_errors`: Array of absolute errors for each feature.
- `r_squared`: Out-of-sample R-squared statistic of the regression.

## Citing

If you use this code for research, please cite the associated paper.
```
@misc{https://doi.org/10.48550/arxiv.2310.19245,
  doi = {10.48550/ARXIV.2310.19245},
  url = {https://arxiv.org/abs/2310.19245},
  author = {Bell,  Logan and Devanathan,  Nikhil and Boyd,  Stephen},
  keywords = {Computation (stat.CO),  FOS: Computer and information sciences,  FOS: Computer and information sciences,  62-08 (Primary),  62-04,  62J99 (Secondary)},
  title = {Efficient Shapley Performance Attribution for Least-Squares Regression},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}
```
