"tests for gassolar.solar"

from gassolar.solar.solar import Mission


def test_solar_gp():
    model = Mission(latitude=11)
    model.cost = model["W_{total}"]
    sol = model.solve(verbosity=0)
    assert sol.cost > 0


def test_solar_sp():
    model = Mission(latitude=11, sp=True)
    model.cost = model["W_{total}"]
    sol = model.localsolve(verbosity=0)
    assert sol.cost > 0
