"""
Cholesky Decomposition Utilities
==================================
Standalone Cholesky decomposition used by BrownianND and BlackScholesND
to generate correlated Gaussian increments.

Given a correlation matrix Sigma, computes the lower triangular factor L
such that Sigma = L * L^T  (Cholesky-Banachiewicz algorithm).

Usage:
  L = cholesky_decompose(corr_matrix)
  validate_correlation(corr_matrix, n)

These functions are separate from any class so they can be reused
by any process that needs correlated Brownian motion.
"""

import math
from typing import List


def cholesky_decompose(matrix: List[List[float]]) -> List[List[float]]:
    """
    Compute the lower Cholesky factor L such that matrix = L * L^T.

    Uses the Cholesky-Banachiewicz algorithm.
    Numerical noise (small negative diagonal) is clamped to 0.

    Parameters
    ----------
    matrix : list of list of float
        Symmetric positive semi-definite matrix of size n x n.

    Returns
    -------
    list of list of float
        Lower triangular Cholesky factor L.
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = matrix[i][i] - s
                # Clamp small numerical negatives to 0
                if val < 0:
                    val = 0.0
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0:
                    L[i][j] = 0.0
                else:
                    L[i][j] = (matrix[i][j] - s) / L[j][j]

    return L


def validate_correlation(matrix: List[List[float]], n: int) -> None:
    """
    Validate that a correlation matrix is n x n with diagonal entries = 1.

    Parameters
    ----------
    matrix : list of list of float
        Candidate correlation matrix.
    n : int
        Expected dimension.

    Raises
    ------
    ValueError
        If the matrix has wrong size or non-unit diagonal.
    """
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"Correlation matrix must be {n}x{n}.")
    for i in range(n):
        if abs(matrix[i][i] - 1.0) > 1e-9:
            raise ValueError(
                f"Diagonal entry [{i}][{i}] must be 1.0 (got {matrix[i][i]})."
            )


def recover_correlation(L: List[List[float]]) -> List[List[float]]:
    """
    Recover the correlation matrix Sigma = L * L^T from its Cholesky factor.

    Parameters
    ----------
    L : list of list of float
        Lower triangular Cholesky factor.

    Returns
    -------
    list of list of float
        Correlation matrix Sigma.
    """
    n = len(L)
    return [
        [sum(L[i][k] * L[j][k] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]
