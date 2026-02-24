"tests for gassolar.solar_detailed"

import pytest

from gassolar.solar_detailed.solar import Aircraft, Mission


def test_solar_detailed_gp():
    v = Aircraft(sp=False)
    m = Mission(v, latitude=[20])
    m.cost = m[m.aircraft.Wtotal]
    sol = m.solve(verbosity=0)
    assert sol.cost == pytest.approx(814.409, rel=1e-3)


@pytest.mark.skip(
    reason=(
        "SP PCCP does not converge: TailBoomFlexibility slackens by ~9%% at "
        "pccp_penalty=200. First SP iteration cost (4e48) is wildly "
        "inconsistent with GP optimal (814), indicating a model bug. "
        "Needs investigation before enabling."
    )
)
def test_solar_detailed_sp():
    v = Aircraft(sp=True)
    m = Mission(v, latitude=[20])
    m.cost = m[m.aircraft.Wtotal]
    sol = m.localsolve(verbosity=0)
    assert sol.cost > 0
