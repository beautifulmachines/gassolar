"steady level flight model"

from gpkit import Model, Var


class SteadyLevelFlight(Model):
    "steady level flight model"

    T = Var("N", "thrust")

    def setup(self, state, aircraft, perf):
        return [
            (perf["W_{end}"] * perf["W_{start}"]) ** 0.5
            <= (
                0.5
                * state["\\rho"]
                * state["V"] ** 2
                * perf.wing.CL
                * aircraft.wing["S"]
            ),
            self.T
            >= (
                0.5
                * state["\\rho"]
                * state["V"] ** 2
                * perf["C_D"]
                * aircraft.wing["S"]
            ),
            perf["P_{shaft}"] >= self.T * state["V"] / aircraft["\\eta_{prop}"],
        ]
