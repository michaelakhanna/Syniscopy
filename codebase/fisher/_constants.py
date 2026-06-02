"""Numerical constants shared by Fisher diagnostics."""

from __future__ import annotations

import numpy as np


# Smallest denominator used when inverting the Fisher information matrix.
# Below this the matrix is considered singular and infinite CRLB is returned.
_FISHER_DET_EPS = 1e-30
_FISHER_VARIANCE_FLOOR = 1e-30

# Relative determinant and rank floors for scale-dependent Fisher singularity checks.
# _FISHER_RANK_RELATIVE_TOL is the strict default rank gate used by fusion.
_RELATIVE_DET_SINGULAR_TOL = 1e-18
_FISHER_RANK_RELATIVE_TOL = 1e-12
_FISHER_EIGENVALUE_UNDERFLOW_FLOOR = np.finfo(float).tiny

# Residual norm tolerance for deciding whether a state axis lies in the
# numerical range of a singular Fisher matrix.
_FISHER_RANGE_RESIDUAL_TOL = 1e-8

# SE(3) rank diagnostics mix translational and rotational units, so axis
# observability gets its own diagonal-scale gate before the shared rank check.
_FISHER_SE3_EPS = 1e-30
_FISHER_SE3_AXIS_RELATIVE_TOL = 1e-12

# Armijo backtracking constants for the time-allocation Frank-Wolfe step.
_LINE_SEARCH_DESCENT_TOL = 1e-18
_LINE_SEARCH_SHRINK = 0.5
_LINE_SEARCH_ARMIJO_C = 1e-4
_LINE_SEARCH_MAX_STEPS = 40
