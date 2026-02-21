"tests for gassolar.environment"
from gassolar.environment.wind_speeds import get_windspeed
from gassolar.environment.air_properties import get_airvars


def test_get_windspeed():
    v = get_windspeed(45, perc=90, altitude=15000, day=355)
    assert v > 0


def test_get_airvars():
    density, viscosity = get_airvars(15000)
    assert density > 0
    assert viscosity > 0


def test_windspeed_increases_with_altitude():
    v_low = get_windspeed(45, 90, 5000, 355)
    v_high = get_windspeed(45, 90, 30000, 355)
    assert v_high > v_low


def test_density_decreases_with_altitude():
    rho_low, _ = get_airvars(0)
    rho_high, _ = get_airvars(30000)
    assert rho_low > rho_high
