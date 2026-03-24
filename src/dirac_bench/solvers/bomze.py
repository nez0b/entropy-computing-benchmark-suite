"""Bomze-regularized solver wrappers.

Each wrapper computes A_bar = A + 0.5*I, delegates to the underlying solver,
then fixes the omega value using the Bomze inversion formula.
"""

import numpy as np

from dirac_bench.problems import bomze_regularize, bomze_objective_to_omega
from dirac_bench.solvers.slsqp import solve_slsqp
from dirac_bench.solvers.lbfgs import solve_lbfgs


def _fix_omega(result: dict) -> dict:
    """Recompute omega using Bomze formula and tag formulation."""
    result["omega"] = bomze_objective_to_omega(result["best_objective"])
    result["formulation"] = "bomze"
    return result


def solve_bomze_slsqp(A: np.ndarray, **kwargs) -> dict:
    """Bomze-regularized SLSQP solver."""
    A_bar = bomze_regularize(A)
    result = solve_slsqp(A_bar, **kwargs)
    return _fix_omega(result)


def solve_bomze_lbfgs(A: np.ndarray, **kwargs) -> dict:
    """Bomze-regularized L-BFGS-B solver."""
    A_bar = bomze_regularize(A)
    result = solve_lbfgs(A_bar, **kwargs)
    return _fix_omega(result)


def solve_bomze_dirac(A: np.ndarray, **kwargs) -> dict:
    """Bomze-regularized Dirac-3 cloud solver."""
    from dirac_bench.solvers.dirac import solve_dirac

    A_bar = bomze_regularize(A)
    result = solve_dirac(A_bar, **kwargs)
    return _fix_omega(result)


def solve_bomze_dirac_direct(A: np.ndarray, **kwargs) -> dict:
    """Bomze-regularized Dirac-3 direct hardware solver."""
    from dirac_bench.solvers.dirac_direct import solve_dirac_direct

    A_bar = bomze_regularize(A)
    result = solve_dirac_direct(A_bar, **kwargs)
    return _fix_omega(result)
