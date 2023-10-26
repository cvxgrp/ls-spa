# Least-Squares Shapley Performance Attribution (LS-SPA)

Library companion to the paper [Efficient Shapley Performance Attribution for Least-Squares
Regression](https://web.stanford.edu/~boyd/papers/pdf/ls_shapley_perf.pdf) by Logan Bell, 
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
import ls_spa
```
to the top of your python file.

`lsspa` has the following dependencies:
- `numpy`
- `scipy`
- `pandas`

`lsspa` also requires [JAX](https://github.com/google/jax). 
Because JAX installation varies by platform, we do not list JAX in our
`requirements.txt`. To install JAX, follow 
[these instructions](https://github.com/google/jax#installation).

## Usage

Here is an example usage of LS-SPA

