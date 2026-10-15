from gpkit import Model
from gpkit.constraints.bounded import Bounded
from gpkit.constraints.relax import ConstantsRelaxed

"""
Methods to precondition an SP so that it solves with a relaxed constants algorithim
and postcondition an SP to ensure all relax values are 1
"""


def relaxed_constants(model, include_only=None, exclude=None):  # noqa: ARG001
    """
    Method to precondition an SP so it solves with a relaxed constants
        algorithim

    ARGUMENTS
    ---------
    model: the model to solve with relaxed constants

    RETURNS
    -------
    feas: the input model but with relaxed constants and a new objective
    """

    if model.substitutions:
        constsrelaxed = ConstantsRelaxed(Bounded(model))
        feas = Model(constsrelaxed.relaxvars.prod() ** 20 * model.cost, constsrelaxed)
        # NOTE: It hasn't yet been seen but might be possible that
        #       the model.cost component above could cause infeasibility
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
