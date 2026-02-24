"flight state of gas powered aircraft"

from gpkit import Model, Var, Variable

from gassolar.environment.air_properties import get_airvars


class FlightState(Model):
    """
    environmental state of aircraft

    inputs
    ------
    latitude: earth latitude [deg]
    altitude: flight altitude [ft]
    percent: percentile wind speeds [%]
    day: day of the year [Jan 1st = 1]
    """

    mfac = Var("-", "wind speed margin factor", value=1.0)
    V = Var("m/s", "true airspeed")
    qne = Var("kg/s^2/m", "never exceed dynamic pressure")
    Vne = Var("m/s", "never exceed velocity", value=40)
    rhosl = Var("kg/m^3", "air density at sea level", value=1.225)
    href = Var("ft", "reference altitude", value=15000)

    def setup(self, Vwind, latitude=45, percent=90, altitude=15000, day=355):

        # wind = get_windspeed(latitude, percent, altitude, day)
        density, vis = get_airvars(altitude)

        # Vwind = Variable("V_{wind}", wind, "m/s", "wind velocity")
        self.rho = Variable("\\rho", density, "kg/m**3", "air density")
        self.mu = Variable("\\mu", vis, "N*s/m**2", "dynamic viscosity")
        Variable("h", altitude, "ft", "flight altitude")

        constraints = [
            self.V / self.mfac >= Vwind,
            self.qne == 0.5 * self.rhosl * self.Vne**2,
        ]

        return constraints
