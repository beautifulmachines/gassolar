"tests for gassolar.jho — Jungle Hawk Owl as-built gas UAV"

import pytest
from gpkit import Variable

from gassolar.jho.jho import Mission


def test_jho_builds():
    "Mission model instantiates without error and has expected variable count"
    m = Mission()
    assert len(m.vks) > 100


def test_jho_solves():
    """Mission model with fixed loiter time solves and MTOW is in physical range.

    Variable access: CONV-06 compliant — no string subscript access.
    - Loiter time: varkeys.by_name('t') filtered by lineage to Loiter submodel,
      wrapped in Variable() for use as a monomial expression
    - MTOW cost: varkeys.by_name('MTOW'), unique key, wrapped in Variable()
    - Vtail volume coefficient: model.JHO.emp.vtail.Vv (attribute access)

    Lineage filter for Loiter.t: lineage has exactly 2 entries, last is 'Loiter'.
    This is robust to gpkit's Mission counter (Mission, Mission1, Mission2, ...).
    """
    model = Mission()

    # CONV-06 compliant substitution: attribute access for vtail Vv
    model.substitutions[model.JHO.emp.vtail.Vv] = 0.04

    # CONV-06 compliant loiter time substitution via varkeys.by_name() + lineage filter.
    # Mission.Loiter.t is the loiter endurance variable (days), distinct from:
    #   - Mission.Loiter.FlightSegment.BreguetEndurance.t[:] (segment time array)
    #   - structural thickness variables (Aircraft.Wing.CapSpar.t etc.)
    loiter_t_vk = next(
        vk
        for vk in model.varkeys.by_name("t")
        if len(vk.lineage) == 2 and vk.lineage[-1][0] == "Loiter"
    )
    loiter_t = Variable(loiter_t_vk)
    model.substitutions[loiter_t] = 6  # 6 days loiter

    # CONV-06 compliant cost: MTOW via varkeys.by_name() (unique, no ambiguity)
    mtow_vk = next(iter(model.varkeys.by_name("MTOW")))
    mtow = Variable(mtow_vk)
    model.cost = mtow

    sol = model.localsolve(verbosity=0)

    # JHO as-built was ~150 lbf; baseline from localsolve: 138.458 lbf
    assert sol.cost == pytest.approx(138.458, rel=1e-2)
