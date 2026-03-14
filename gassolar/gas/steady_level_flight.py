"steady level flight model"

from gpkit import Model, Var


class SteadyLevelFlight(Model):
    "steady level flight model"

    T = Var("N", "thrust")

    def setup(self, state, aircraft, perf):
        etaprop = aircraft.etaprop
        wingS = aircraft.wing.S
        return [
            (perf.Wend * perf.Wstart) ** 0.5
            <= (0.5 * state.rho * state.V**2 * perf.wing.CL * wingS),
            self.T >= (0.5 * state.rho * state.V**2 * perf.CD * wingS),
            perf.engine.P_shaft >= self.T * state.V / etaprop,
        ]
