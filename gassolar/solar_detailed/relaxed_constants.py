"""Utilities for solving SP models with relaxed constants.

relaxed_constants() wraps a model so constants can relax during solving,
which helps find a feasible starting point for hard SP problems.
post_process() reports any constants that were relaxed in the solution.
"""

from gpkit import Model
from gpkit.constraints.bounded import Bounded
from gpkit.constraints.relax import ConstantsRelaxed


def relaxed_constants(model, include_only=None, exclude=None):
    """Wrap a model so constants are allowed to relax during solving.

    Returns a new Model whose objective drives all relaxation variables
    toward 1 (no relaxation) while satisfying the original constraints.
    A solved relaxation value > 1 means that constant was made easier
    by that factor to achieve feasibility.
    """
    if model.substitutions:
        constsrelaxed = ConstantsRelaxed(Bounded(model))
        feas = Model(constsrelaxed.relaxvars.prod() ** 20 * model.cost, constsrelaxed)
    else:
        feas = Model(model.cost, model)
    return feas


def post_process(sol):
    """Print any constants relaxed beyond 0.1% and return their keys."""
    bdvars = []
    for k, v in sol.variables.items():
        if "Relax" not in str(k) or "before" in str(k):
            continue
        val = v.magnitude if hasattr(v, "magnitude") else float(v)
        if val >= 1.001:
            bdvars.append((k, val))
    if bdvars:
        print("GP iteration has relaxed constants")
        for k, val in bdvars:
            print(f"  {k}: {val:.4g}")
    return [k for k, _ in bdvars]
