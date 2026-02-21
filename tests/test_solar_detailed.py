"tests for gassolar.solar_detailed"

from gassolar.solar_detailed.solar import Aircraft, Mission


def test_solar_detailed_gp():
    v = Aircraft(sp=False)
    m = Mission(v, latitude=[20])
    m.cost = m[m.aircraft.Wtotal]
    sol = m.solve(verbosity=0)
    assert sol.cost > 0


def test_solar_detailed_sp():
    v = Aircraft(sp=True)
    m = Mission(v, latitude=[20])
    m.cost = m[m.aircraft.Wtotal]
    sol = m.localsolve(verbosity=0)
    assert sol.cost > 0
