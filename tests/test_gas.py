"tests for gassolar.gas"

import pytest

from gassolar.gas.gas import Mission


def test_gas_gp():
    model = Mission()
    model.substitutions[model.loiter["t"]] = 6
    model.cost = model["MTOW"]
    sol = model.solve(verbosity=0)
    assert sol.cost == pytest.approx(111.307, rel=1e-3)


def test_gas_sp():
    model = Mission(sp=True)
    model.substitutions[model.loiter["t"]] = 6
    model.cost = model["MTOW"]
    sol = model.localsolve(verbosity=0)
    assert sol.cost == pytest.approx(112.506, rel=1e-3)
