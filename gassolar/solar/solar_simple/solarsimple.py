"Simple Solar-Electric Powered Aircraft Model"

from gpkit import Model, Variable
from gpkitmodels.GP.aircraft.wing.wing import WingAero

from gassolar.solar.solar import FlightState


class Aircraft(Model):
    "vehicle"

    def setup(self):

        Wstructures = Variable("W_{structures}", "lbf", "structural weight")
        fstructures = Variable(
            "f_{structures}", "-", "fractional structural weight", value=0.35
        )
        Wpay = Variable("W_{pay}", "lbf", "payload", value=10)
        Wsolar = Variable("W_{solar}", "lbf", "solar cell weight")
        Wbatt = Variable("W_{batt}", "lbf", "battery weight")
        W = Variable("W", "lbf", "aircraft weight")
        S = Variable("S", "ft**2", "planform area")
        b = Variable("b", "ft", "wing span")
        cmac = Variable("c_{MAC}", "ft", "mean aerodynamic chord")
        AR = Variable("AR", "-", "aspect ratio", value=27)
        W = Variable("W", "lbf", "aircraft weight")
        Ssolar = Variable("S_{solar}", "ft**2", "solar cell area")
        eta_solar = Variable("\\eta_{solar}", "-", "Solar cell efficiency", value=0.2)
        eta_charge = Variable(
            "\\eta_{charge}", "-", "Battery charging efficiency", value=0.98
        )
        eta_discharge = Variable(
            "\\eta_{discharge}", "-", "Battery discharging efficiency", value=0.98
        )
        hbatt = Variable("h_{batt}", "W*hr/kg", "battery energy density", value=350)
        Ebatt = Variable("E_{batt}", "J", "total battery energy")
        rhosolar = Variable(
            "\\rho_{solar}", "kg/m^2", "solar cell area density", value=0.3
        )
        g = Variable("g", "m/s**2", "gravitational constant", value=9.81)

        constraints = [
            Wstructures >= W * fstructures,
            W >= Wstructures + Wsolar + Wbatt + Wpay,
            Wsolar >= rhosolar * Ssolar * g,
            Wbatt >= Ebatt / hbatt * g,
            Ssolar <= S,
            b**2 == S * AR,
            cmac == S / b,
        ]

        return constraints

    def flight_model(self, state):
        return AircraftPerf(self, state)


class AircraftPerf(Model):
    "aircraft performance"

    def setup(self, static, state):

        self.wing = WingAero(static, state)

        CD = Variable("C_D", "-", "aircraft drag coefficient")
        cda0 = Variable("CDA_0", "-", "non-wing drag coefficient", value=0.005)
        Pshaft = Variable("P_{shaft}", "hp", "shaft power")

        constraints = [CD >= cda0 + self.wing["C_d"]]

        return self.wing, constraints


class FlightSegment(Model):
    "flight segment"

    def setup(self, aircraft, etap=0.7, latitude=35, day=355):

        self.aircraft = aircraft
        self.fs = FlightState(latitude, day)
        self.aircraftPerf = self.aircraft.flight_model(self.fs)
        self.slf = SteadyLevelFlight(self.fs, self.aircraft, self.aircraftPerf, etap)
        self.power = Power(self.aircraft, self.fs)

        _submodels = [self.fs, self.aircraftPerf, self.slf, self.power]

        constraints = [
            self.power["P_{oper}"]
            >= self.power["P_{acc}"] + self.aircraftPerf["P_{shaft}"]
        ]

        return self.aircraft, _submodels, constraints


class SteadyLevelFlight(Model):
    "steady level flight model"

    def setup(self, state, aircraft, perf, etap, **kwargs):  # noqa: ARG002

        T = Variable("T", "N", "thrust")
        etaprop = Variable("\\eta_{prop}", "-", "propulsive efficiency", value=etap)

        constraints = [
            aircraft["W"]
            <= (0.5 * state["\\rho"] * state["V"] ** 2 * perf["C_L"] * aircraft["S"]),
            T >= (0.5 * state["\\rho"] * state["V"] ** 2 * perf["C_D"] * aircraft["S"]),
            perf["P_{shaft}"] >= T * state["V"] / etaprop,
        ]

        return constraints


class Power(Model):
    def setup(self, static, state):

        Poper = Variable("P_{oper}", "W", "Aircraft operating power")
        Pacc = Variable("P_{acc}", "W", "Accessory power draw", value=0.0)

        constraints = [
            state["(E/S)_{irr}"] * static["\\eta_{solar}"] * static["S_{solar}"]
            >= (
                Poper * state["t_{day}"]
                + static["E_{batt}"] / static["\\eta_{discharge}"]
            ),
            static["E_{batt}"]
            >= (Poper * state["t_{night}"] / static["\\eta_{discharge}"]),
        ]
        return constraints


class Mission(Model):
    "define mission for aircraft"

    def setup(self, etap=0.7, latitude=35, day=355):
        # http://sky-sailor.ethz.ch/docs/Conceptual_Design_of_Solar_Powered_Airplanes_for_continuous_flight2.pdf

        solarsimple = Aircraft()
        mission = []
        for l in range(20, latitude + 1, 1):
            mission.append(FlightSegment(solarsimple, etap, l, day))

        return solarsimple, mission


if __name__ == "__main__":
    M = Mission()
    M.cost = M["W"]
    sol = M.solve("mosek")
