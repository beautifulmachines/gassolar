"""Jungle Hawk Owl"""

import numpy as np
from gpkit import Model, Var, Variable, Vectorize
from gpkitmodels.GP.aircraft.engine.gas_engine import Engine
from gpkitmodels.GP.aircraft.fuselage.elliptical_fuselage import Fuselage
from gpkitmodels.GP.aircraft.tail.empennage import Empennage
from gpkitmodels.GP.aircraft.wing.wing import Wing as WingGP
from gpkitmodels.SP.aircraft.tail.tail_boom_flex import TailBoomFlexibility
from gpkitmodels.SP.aircraft.wing.wing import Wing as WingSP
from gpkitmodels.tools.summing_constraintset import summing_vars

from gassolar.gas.flight_segment import FlightSegment
from gassolar.gas.loiter import Loiter

# pylint: disable=invalid-name


class Aircraft(Model):
    "the JHO vehicle"

    Wzfw = Var("lbf", "zero fuel weight")
    Wpay = Var("lbf", "payload weight", value=10)
    Wavn = Var("lbf", "avionics weight", value=8)
    Wwing = Var("lbf", "wing weight for loading")
    etaprop = Var("-", "propulsive efficiency", value=0.8)

    def setup(self, sp=False):

        self.sp = sp

        self.fuselage = Fuselage()
        if sp:
            self.wing = WingSP()
        else:
            self.wing = WingGP()
        self.engine = Engine()
        self.emp = Empennage()

        components = [self.fuselage, self.wing, self.engine, self.emp]
        self.smeared_loads = [self.fuselage, self.engine]

        self.emp.substitutions[self.emp.vtail.Vv] = 0.04
        self.emp.substitutions[self.emp.vtail.planform.tau] = 0.08
        self.emp.substitutions[self.emp.htail.planform.tau] = 0.08
        self.wing.substitutions[self.wing.planform.tau] = 0.115

        if not sp:
            self.emp.substitutions[self.emp.htail.Vh] = 0.45
            self.emp.substitutions[self.emp.htail.planform.AR] = 5.0
            self.emp.substitutions[self.emp.htail.mh] = 0.1

        constraints = [
            self.Wzfw >= sum(summing_vars(components, "W")) + self.Wpay + self.Wavn,
            self.Wwing >= sum(summing_vars([self.wing], "W")),
            self.emp.htail.Vh
            <= (self.emp.htail.S * self.emp.htail.lh / self.wing.S**2 * self.wing.b),
            self.emp.vtail.Vv
            <= (self.emp.vtail.S * self.emp.vtail.lv / self.wing.S / self.wing.b),
            self.wing.planform.tau * self.wing.planform.croot >= self.emp.tailboom.d0,
        ]

        return components, constraints

    def flight_model(self, state):
        return AircraftPerf(self, state)


class AircraftPerf(Model):
    "performance model for aircraft"

    Wend = Var("lbf", "vector-end weight")
    Wstart = Var("lbf", "vector-begin weight")
    CD = Var("-", "drag coefficient")

    def setup(self, static, state):

        self.wing = static.wing.flight_model(static.wing, state)
        self.fuselage = static.fuselage.flight_model(static.fuselage, state)
        self.engine = static.engine.flight_model(state)
        self.htail = static.emp.htail.flight_model(static.emp.htail, state)
        self.vtail = static.emp.vtail.flight_model(static.emp.vtail, state)
        self.tailboom = static.emp.tailboom.flight_model(static.emp.tailboom, state)

        self.dynamicmodels = [
            self.wing,
            self.fuselage,
            self.engine,
            self.htail,
            self.vtail,
            self.tailboom,
        ]
        areadragmodel = [self.fuselage, self.htail, self.vtail, self.tailboom]
        areadragcomps = [
            static.fuselage,
            static.emp.htail,
            static.emp.vtail,
            static.emp.tailboom,
        ]

        CDA = Variable("CDA", "-", "area drag coefficient")
        mfac = Variable("m_{fac}", 1.0, "-", "drag margin factor")

        dvars = []
        for dc, dm in zip(areadragcomps, areadragmodel):
            if "Cf" in dm.varkeys:
                dvars.append(dm.get_var("Cf") * dc.S / static.wing.S)
            if "Cd" in dm.varkeys:
                dvars.append(dm.get_var("Cd") * dc.S / static.wing.S)
            if "C_d" in dm.varkeys:
                dvars.append(dm.get_var("C_d") * dc.S / static.wing.S)

        constraints = [CDA / mfac >= sum(dvars), self.CD >= CDA + self.wing.Cd]

        return self.dynamicmodels, constraints


class Cruise(Model):
    "make a cruise flight segment"

    def setup(
        self, aircraft, N, altitude=15000, latitude=45, percent=90, day=355, R=200
    ):
        self.fs = fs = FlightSegment(aircraft, N, altitude, latitude, percent, day)

        R = Variable("R", R, "nautical_miles", "Range to station")
        constraints = [R / N <= fs.fs.V * fs.be.t]

        return fs, constraints


class Climb(Model):
    "make a climb flight segment"

    def setup(
        self, aircraft, N, altitude=15000, latitude=45, percent=90, day=355, dh=15000
    ):
        self.fs = fs = FlightSegment(aircraft, N, altitude, latitude, percent, day)

        with Vectorize(N):
            hdot = Variable("\\dot{h}", "ft/min", "Climb rate")

        deltah = Variable("\\Delta_h", dh, "ft", "altitude difference")
        hdotmin = Variable("\\dot{h}_{min}", 100, "ft/min", "minimum climb rate")

        constraints = [
            hdot * fs.be.t >= deltah / N,
            hdot >= hdotmin,
            fs.slf.T
            >= (
                0.5 * fs.fs.rho * fs.fs.V**2 * fs.aircraftPerf.CD * fs.aircraft.wing.S
                + fs.aircraftPerf.Wstart * hdot / fs.fs.V
            ),
        ]

        return fs, constraints


class Mission(Model):
    "creates flight profile"

    def setup(self, latitude=38, percent=90, sp=False):

        self.mtow = mtow = Variable("MTOW", "lbf", "max-take off weight")
        Wcent = Variable("W_{cent}", "lbf", "center aircraft weight")
        Wfueltot = Variable("W_{fuel-tot}", "lbf", "total aircraft fuel weight")
        LS = Variable("(W/S)", "lbf/ft**2", "wing loading")

        JHO = Aircraft(sp=sp)

        self.JHO = JHO
        climb1 = Climb(
            JHO,
            10,
            latitude=latitude,
            percent=percent,
            altitude=np.linspace(0, 15000, 11)[1:],
        )
        # cruise1 = Cruise(JHO, 1, R=200, latitude=latitude, percent=percent)
        self.loiter = loiter1 = Loiter(JHO, 5, latitude=latitude, percent=percent)
        # cruise2 = Cruise(JHO, 1, latitude=latitude, percent=percent)
        # mission = [climb1, cruise1, loiter1, cruise2]
        mission = [climb1, loiter1]

        hbend = JHO.emp.tailboom.tailLoad(
            JHO.emp.tailboom, JHO.emp.htail, loiter1.fs.fs
        )
        vbend = JHO.emp.tailboom.tailLoad(
            JHO.emp.tailboom, JHO.emp.vtail, loiter1.fs.fs
        )
        loading = [
            JHO.wing.spar.loading(JHO.wing, loiter1.fs.fs),
            JHO.wing.spar.gustloading(JHO.wing, loiter1.fs.fs),
            hbend,
            vbend,
        ]

        if sp:
            loading.append(TailBoomFlexibility(JHO.emp.htail, hbend, JHO.wing))

        constraints = [
            mtow >= climb1.fs.aircraftPerf.Wstart[0],
            Wfueltot >= sum(fs.fs.W_fuel_fs for fs in mission),
            mission[-1].fs.aircraftPerf.Wend[-1] >= JHO.Wzfw,
            Wcent
            >= Wfueltot
            + JHO.Wpay
            + JHO.Wavn
            + sum(summing_vars(JHO.smeared_loads, "W")),
            LS == mtow / JHO.wing.S,
            JHO.fuselage.Vol >= Wfueltot / JHO.fuselage.rhofuel,
            Wcent == loading[0].W,
            Wcent == loading[1].W,
            loiter1.fs.fs.V[0] == loading[1].v,
            JHO.Wwing == loading[1].Ww,
            loiter1.fs.aircraftPerf.wing.CL[0] == loading[1].cl,
        ]

        for i, fs in enumerate(mission[1:]):
            constraints.extend(
                [mission[i].fs.aircraftPerf.Wend[-1] == fs.fs.aircraftPerf.Wstart[0]]
            )

        loading[0].substitutions[loading[0].Nmax] = 5
        loading[1].substitutions[loading[0].Nmax] = 2

        return JHO, mission, loading, constraints

    @classmethod
    def default(cls):
        "Return a ready-to-solve gas Mission (GP). Latitude=38, loiter time=6 days."
        m = cls()
        m.cost = m.mtow
        m.substitutions[m.loiter.t] = 6
        return m


def test():
    "test for integrated testing"
    model = Mission()
    model.substitutions[model.loiter.t] = 6
    model.cost = model.mtow
    model.solve()
    model = Mission(sp=True)
    model.substitutions[model.loiter.t] = 6
    model.cost = model.mtow
    model.localsolve()


if __name__ == "__main__":
    M = Mission()
    M.substitutions[M.loiter.t] = 6
    M.cost = M.mtow
    sol = M.solve()
