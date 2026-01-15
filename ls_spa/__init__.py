"""This module efficiently computes Shapley values for least-squares problems."""

from .ls_spa import (
    ShapleyResults,
    SizeIncompatibleError,
    error_estimates,
    ls_spa,
    merge_sample_cov,
    merge_sample_mean,
    reduce_data,
    square_shapley,
)

__all__ = [
    "ShapleyResults",
    "SizeIncompatibleError",
    "error_estimates",
    "ls_spa",
    "merge_sample_cov",
    "merge_sample_mean",
    "reduce_data",
    "square_shapley",
]
