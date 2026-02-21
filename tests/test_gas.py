"tests for gassolar.gas"

from gassolar.gas.gas import Mission


def test_gas_gp():
    model = Mission()
    model.substitutions[model.loiter["t"]] = 6
    model.cost = model["MTOW"]
    sol = model.solve(verbosity=0)
    assert sol.cost > 0


def test_gas_sp():
    model = Mission(sp=True)
    model.substitutions[model.loiter["t"]] = 6
    model.cost = model["MTOW"]
    sol = model.localsolve(verbosity=0)
    assert sol.cost > 0
