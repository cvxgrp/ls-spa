# Least-Squares Shapley Performance Attribution (LS-SPA)

### [Installation](#installation) - [Usage](#usage) - [Hello world](#hello-world) - [Example notebook](#example-notebook) - [Optional arguments](#optional-arguments) - [Citing](#citing)

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

```bash
pip install ls_spa
```

Import `ls_spa` by adding

```python
from ls_spa import ls_spa
```

to the top of your Python file.

`ls_spa` has the following dependencies:

- `numpy`
- `scipy`
- `pandas`
- `joblib`

Optional dependencies are

- `marimo` for using the demo notebook
- `matplotlib` for plotting in the demo notebook

## Usage

We assume that you have imported `ls_spa` and you have a $N\times p$
matrix of training data `X_train`, a $M\times p$ matrix of testing data `X_test`,
a $N$ vector of training labels `y_train`, and a $M$ vector of testing labels `y_test`
for positive integers $p, N, M$ with $N,M\geq p$. In this case, you can find the
Shapley attribution of the out-of-sample $R^2$ on your data by executing

```python
attrs = ls_spa(X_train, X_test, y_train, y_test).attribution
```

`attrs` will be a NumPy array containing the Shapley values of your features.
The `ls_spa` function computes Shapley values for the given data using
the LS-SPA method described in the companion paper. It takes arguments:

- `X_train`: Training feature matrix (NumPy array or pandas DataFrame).
- `X_test`: Testing feature matrix (NumPy array or pandas DataFrame).
- `y_train`: Training response vector (NumPy array or pandas Series).
- `y_test`: Testing response vector (NumPy array or pandas Series).

## Hello world

We present a complete Python script that utilizes LS-SPA to compute
the Shapley attribution on the data from the toy example described
in the companion paper.

```python
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
For more info, see [optional arguments](#optional-arguments).

## Example notebook

In this [demo](./notebooks/shapley_toy.py), we walk through the process of
computing Shapley values on the data for the toy example in the
companion paper. We then use `ls_spa` to compute the Shapley attribution
on the same data.

## Optional arguments

`ls_spa` takes the optional arguments:

- `reg`: Ridge regularization parameter (Default `0.0`).
- `max_samples`: Maximum number of feature permutations to sample (Default `8192`).
- `batch_size`: Number of permutations to process per batch (Default `256`).
- `tolerance`: Stopping criterion for estimation error (Default `0.01`).
- `seed`: Seed for random number generation (Default `42`).
- `perms`: Permutation sampling method (Default `None`). Options include:
  - `None`: Auto-select `"exact"` for p < 9 features, otherwise `"random"`
  - `"exact"`: Enumerate all permutations (only feasible for p < 9)
  - `"random"`: Uniformly random permutations
  - `"argsort"`: Quasi-Monte Carlo permutations using argsort
  - `"permutohedron"`: Quasi-Monte Carlo permutations from permutohedron lattice
  - Custom array or tuple of permutations
- `antithetical`: Use antithetical (paired) sampling for variance reduction (Default `True`).
- `return_attribution_history`: Return convergence history of attributions (Default `False`).
- `n_jobs`: Number of parallel jobs; use `-1` for all CPU cores (Default `1`).

`ls_spa` returns a `ShapleyResults` object. The `ShapleyResults` object
has the fields:

- `attribution`: Array of Shapley values for each feature.
- `theta`: Array of regression coefficients with all features.
- `r_squared`: Out-of-sample R² with all features.
- `overall_error`: Estimated error (95th percentile L2 norm) in Shapley attribution vector.
- `attribution_errors`: Array of estimated errors for each feature's attribution.
- `error_history`: Array of error estimates after each batch. `None` if using exact computation.
- `attribution_history`: Array of attribution estimates over time. `None` if `return_attribution_history=False`.

## Citing

If you use this code for research, please cite the associated paper.

```bibtex
@article{Bell2024,
  title = {Efficient Shapley performance attribution for least-squares regression},
  volume = {34},
  ISSN = {1573-1375},
  url = {http://dx.doi.org/10.1007/s11222-024-10459-9},
  DOI = {10.1007/s11222-024-10459-9},
  number = {5},
  journal = {Statistics and Computing},
  publisher = {Springer Science and Business Media LLC},
  author = {Bell,  Logan and Devanathan,  Nikhil and Boyd,  Stephen},
  year = {2024},
  month = jul
}
```
