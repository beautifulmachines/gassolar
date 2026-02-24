"Simple Solar-Electric Powered Aircraft Model"

# pylint: disable=invalid-name, too-many-instance-attributes, too-many-locals
# pylint: disable=redefined-variable-type, too-many-statements, not-callable
import contextlib
import io
from os import sep
from os.path import abspath, dirname

import pandas as pd
from gpkit import (
    Model,
    SignomialsEnabled,
    Var,
    Variable,
    Vectorize,
    VectorVariable,
    ureg,
)
from gpkitmodels import g
from gpkitmodels.GP.aircraft.fuselage.elliptical_fuselage import Fuselage
from gpkitmodels.GP.aircraft.motor.motor import Motor
from gpkitmodels.GP.aircraft.prop.propeller import Propeller
from gpkitmodels.GP.aircraft.tail.empennage import Empennage
from gpkitmodels.GP.aircraft.tail.horizontal_tail import HorizontalTail
from gpkitmodels.GP.aircraft.tail.tail_boom import TailBoom
from gpkitmodels.GP.aircraft.tail.vertical_tail import VerticalTail
from gpkitmodels.GP.aircraft.wing.boxspar import BoxSpar as BoxSparGP
from gpkitmodels.GP.aircraft.wing.wing import Wing as WingGP
from gpkitmodels.GP.aircraft.wing.wing_skin import WingSecondStruct
from gpkitmodels.GP.materials import cfrpfabric, cfrpud, foamhd
from gpkitmodels.SP.aircraft.prop.propeller import BladeElementProp
from gpkitmodels.SP.aircraft.tail.tail_boom_flex import TailBoomFlexibility
from gpkitmodels.SP.aircraft.wing.boxspar import BoxSpar as BoxSparSP
from gpkitmodels.SP.aircraft.wing.wing import Wing as WingSP
from gpkitmodels.tools.fit_constraintset import FitCS as FCS
from numpy import array, exp, hstack

import gassolar.environment
from gassolar.environment.solar_irradiance import get_Eirr, twi_fits
from gassolar.environment.wind_speeds import get_month
from gassolar.solar_detailed.relaxed_constants import post_process, relaxed_constants

path = dirname(gassolar.environment.__file__)


# Explicit subclasses with solar-specific structural configurations,
# replacing the runtime class mutations that were used previously.
class _SolarHTail(HorizontalTail):
    spar_model = BoxSparGP
    fill_model = None
    skin_model = WingSecondStruct


class _SolarVTail(VerticalTail):
    spar_model = BoxSparGP
    fill_model = None
    skin_model = WingSecondStruct


class _SolarTailBoom(TailBoom):
    spar_model = BoxSparGP
    secondaryWeight = True


class _SolarWingGP(WingGP):
    spar_model = BoxSparGP
    fill_model = None
    skin_model = WingSecondStruct


class _SolarWingSP(WingSP):
    spar_model = BoxSparSP
    fill_model = None
    skin_model = WingSecondStruct


class AircraftPerf(Model):
    "Aircaft Performance"

    def setup(self, static, state, onDesign=False):
        self.drag = AircraftDrag(static, state, onDesign)
        self.CD = self.drag.CD
        self.CL = self.drag.CL
        self.Pshaft = self.drag.Pshaft
        Poper = self.drag.Poper
        E = self.E = static.battery.E
        etacharge = self.etacharge = static.battery.etacharge
        etadischarge = self.etadischarge = static.battery.etadischarge
        etasolar = self.etasolar = static.solarcells.etasolar
        Ssolar = self.Ssolar = static.Ssolar
        ESirr = self.ESirr = state.ESirr
        ESday = self.ESday = state.ESday
        EStwi = self.EStwi = state.EStwi
        tnight = self.tnight = state.tnight
        PSmin = self.PSmin = state.PSmin

        constraints = [
            ESirr >= (ESday + E / etacharge / etasolar / Ssolar),
            E * etadischarge >= (Poper * tnight + EStwi * etasolar * Ssolar),
            Poper == PSmin * Ssolar * etasolar,
        ]

        return self.drag, constraints


class AircraftDrag(Model):
    "Aircaft Performance"

    CD = Var("-", "aircraft drag coefficient")
    cda = Var("-", "non-wing drag coefficient")
    mfac = Var("-", "drag margin factor", value=1.05)
    Pshaft = Var("hp", "shaft power")
    Pavn = Var("W", "avionics power draw", value=200)
    Ppay = Var("W", "payload power draw", value=100)
    Poper = Var("W", "operating power")
    mpower = Var("-", "power margin", value=1.05)
    T = Var("lbf", "thrust")

    def setup(self, static, state, onDesign=False):
        CD, cda, mfac, Pavn, Ppay, Poper, mpower, T = (
            self.CD,
            self.cda,
            self.mfac,
            self.Pavn,
            self.Ppay,
            self.Poper,
            self.mpower,
            self.T,
        )

        fd = dirname(abspath(__file__)) + sep + "dai1336a.csv"

        self.wing = static.wing.flight_model(static.wing, state, fitdata=fd)
        self.htail = static.emp.htail.flight_model(static.emp.htail, state)
        self.vtail = static.emp.vtail.flight_model(static.emp.vtail, state)
        self.tailboom = static.emp.tailboom.flight_model(static.emp.tailboom, state)
        self.motor = static.motor.flight_model(static.motor, state)
        if static.sp:
            if onDesign:
                static.propeller.flight_model = BladeElementProp

        self.propeller = static.propeller.flight_model(static.propeller, state)

        self.flight_models = [
            self.wing,
            self.htail,
            self.vtail,
            self.tailboom,
            self.motor,
            self.propeller,
        ]

        e = self.e = self.wing.e
        cdht = self.cdht = self.htail.Cd
        cdvt = self.cdvt = self.vtail.Cd
        Sh = self.Sh = static.Sh
        Sv = self.Sv = static.Sv
        Sw = self.Sw = static.Sw
        cftb = self.cftb = self.tailboom.Cf
        Stb = self.Stb = static.emp.tailboom.S
        cdw = self.cdw = self.wing.Cd
        self.CL = self.wing.CL
        Nprop = static.Nprop
        Tprop = self.propeller.T
        Qprop = self.propeller.Q
        RPMprop = self.propeller.omega
        Qmotor = self.motor.Q
        RPMmotor = self.motor.omega
        Pelec = self.motor.Pelec

        self.wing.substitutions[e] = 0.95
        self.wing.substitutions[self.wing.CLstall] = 4

        self.wing.substitutions[e] = 0.95
        dvars = [cdht * Sh / Sw, cdvt * Sv / Sw, cftb * Stb / Sw]

        if static.Npod != 0:
            with Vectorize(static.Npod):
                self.fuse = static.fuselage.flight_model(static.fuselage, state)
            self.flight_models.extend([self.fuse])
            cdfuse = self.fuse.Cd
            Sfuse = static.fuselage.S
            dvars.extend(cdfuse * Sfuse / Sw)
            self.fuse.substitutions[self.fuse.mfac] = 1.1

        constraints = [
            cda >= sum(dvars),
            Tprop == T / Nprop,
            Qmotor == Qprop,
            RPMmotor == RPMprop,
            CD / mfac >= cda + cdw,
            Poper / mpower >= Pavn + Ppay + (Pelec * Nprop),
        ]

        return self.flight_models, constraints


class Aircraft(Model):
    "Aircraft Model"

    Wpay = Var("lbf", "payload weight", value=11)
    Wavn = Var("lbf", "avionics weight", value=22)
    Wtotal = Var("lbf", "aircraft weight")
    Wwing = Var("lbf", "wing weight")
    Wcent = Var("lbf", "center weight")
    mfac = Var("-", "total weight margin", value=1.05)
    fland = Var("-", "fractional landing gear weight", value=0.02)
    Wland = Var("lbf", "landing gear weight")
    Nprop = Var("-", "Number of propulsors", value=4)
    minvttau = Var("-", "minimum vertical tail tau ratio", value=0.09)
    minhttau = Var("-", "minimum horizontal tail tau ratio", value=0.06)
    maxtau = Var("-", "maximum wing tau ratio", value=0.144)

    fuseModel = None
    flight_model = AircraftPerf

    def setup(self, Npod=0, sp=False):
        self.Npod = Npod
        self.sp = sp

        Wpay, Wavn, Wtotal, Wwing, Wcent, mfac, fland, Wland = (
            self.Wpay,
            self.Wavn,
            self.Wtotal,
            self.Wwing,
            self.Wcent,
            self.mfac,
            self.fland,
            self.Wland,
        )
        Nprop, minvttau, minhttau, maxtau = (
            self.Nprop,
            self.minvttau,
            self.minhttau,
            self.maxtau,
        )

        cfrpud.substitutions.update(
            {cfrpud.rho: 1.5, cfrpud.E: 200, cfrpud.tmin: 0.1, cfrpud.sigma: 1500}
        )
        cfrpfabric.substitutions.update(
            {
                cfrpfabric.rho: 1.3,
                cfrpfabric.E: 40,
                cfrpfabric.tmin: 0.1,
                cfrpfabric.sigma: 300,
                cfrpfabric.tau: 80,
            }
        )
        foamhd.substitutions.update({foamhd.rho: 0.03})
        materials = [cfrpud, cfrpfabric, foamhd]

        self.emp = Empennage(
            N=5,
            htail_cls=_SolarHTail,
            vtail_cls=_SolarVTail,
            tailboom_cls=_SolarTailBoom,
        )
        self.solarcells = SolarCells()
        self.battery = Battery()
        if sp:
            self.wing = _SolarWingSP(N=20)
        else:
            self.wing = _SolarWingGP(N=20)
        self.motor = Motor()
        self.propeller = Propeller()  # ActuatorProp is already the default flight_model
        self.components = [self.solarcells, self.wing, self.battery, self.emp]
        self.propulsor = [self.motor, self.propeller]

        Sw = self.Sw = self.wing.planform.S
        cmac = self.cmac = self.wing.planform.cmac
        tau = self.tau = self.wing.planform.tau
        croot = self.croot = self.wing.planform.croot
        b = self.b = self.wing.planform.b
        Vh = self.Vh = self.emp.htail.Vh
        lh = self.lh = self.emp.htail.lh
        Sh = self.Sh = self.emp.htail.planform.S
        Vv = self.Vv = self.emp.vtail.Vv
        Sv = self.Sv = self.emp.vtail.planform.S
        lv = self.lv = self.emp.vtail.lv
        d0 = self.d0 = self.emp.tailboom.d0
        Ssolar = self.Ssolar = self.solarcells.S
        mfsolar = self.mfsolar = self.solarcells.mfac
        Volbatt = self.battery.Volbatt
        vttau = self.emp.vtail.planform.tau
        httau = self.emp.htail.planform.tau

        self.emp.substitutions[Vv] = 0.02
        self.emp.substitutions[self.emp.htail.skin.rhoA] = 0.4
        self.emp.substitutions[self.emp.vtail.skin.rhoA] = 0.4
        self.emp.substitutions[self.emp.tailboom.wlim] = 1.0
        self.wing.substitutions[self.wing.mfac] = 1.0
        if not sp:
            self.emp.substitutions[Vh] = 0.45
            self.emp.substitutions[self.emp.htail.mh] = 0.1

        constraints = [
            Ssolar * mfsolar <= Sw,
            Vh <= Sh * lh / Sw / cmac,
            Vv <= Sv * lv / Sw / b,
            d0 <= tau * croot,
            Wland >= fland * Wtotal,
            vttau >= minvttau,
            httau >= minhttau,
            tau <= maxtau,
        ]

        if self.Npod != 0:
            with Vectorize(1):
                with Vectorize(self.Npod):
                    self.fuselage = Fuselage()

            self.k = self.fuselage.k
            Volfuse = self.Volfuse = self.fuselage.Vol[:, 0]
            Wbatt = self.battery.W
            Wfuse = sum(self.fuselage.W)
            self.fuselage.substitutions[self.fuselage.nply] = 5

            constraints.extend(
                [
                    Volbatt <= Volfuse,
                    Wwing >= self.wing.W + self.solarcells.W,
                    Wcent
                    >= (
                        Wpay
                        + Wavn
                        + self.emp.W
                        + self.motor.W * Nprop
                        + self.fuselage.W[0]
                        + Wbatt / self.Npod
                    ),
                    Wtotal / mfac
                    >= (
                        Wpay
                        + Wavn
                        + Wland
                        + Wfuse
                        + sum([c.W for c in self.components])
                        + (Nprop) * sum([c.W for c in self.propulsor])
                    ),
                ]
            )

            self.components.append(self.fuselage)
        else:
            constraints.extend(
                [
                    Wwing
                    >= sum([c.W for c in [self.wing, self.battery, self.solarcells]]),
                    Wcent >= Wpay + Wavn + self.emp.W + self.motor.W * Nprop,
                    Volbatt <= cmac**2 * 0.5 * tau * b,
                    Wtotal / mfac
                    >= (
                        Wpay
                        + Wavn
                        + Wland
                        + sum([c.W for c in self.components])
                        + Nprop * sum([c.W for c in self.propulsor])
                    ),
                ]
            )

        return constraints, self.components, materials, self.propulsor


class Battery(Model):
    "Battery Model"

    W = Var("lbf", "battery weight")
    etacharge = Var("-", "charging efficiency", value=0.98)
    etadischarge = Var("-", "discharging efficiency", value=0.98)
    E = Var("kJ", "total battery energy")
    hbatt = Var("W*hr/kg", "battery specific energy", value=350)
    vbatt = Var("W*hr/l", "battery energy density", value=800)
    Volbatt = Var("m**3", "battery volume")
    etapack = Var("-", "packing efficiency", value=0.85)
    etaRTE = Var("-", "battery RTE", value=0.95)
    minSOC = Var("-", "minimum state of charge", value=1.03)
    rhomppt = Var("kg/kW", "power system mass density", value=0.4223)
    etamppt = Var("-", "power system efficiency", value=0.975)

    def setup(self):
        W, E, minSOC, hbatt, etaRTE, etapack, Volbatt, vbatt = (
            self.W,
            self.E,
            self.minSOC,
            self.hbatt,
            self.etaRTE,
            self.etapack,
            self.Volbatt,
            self.vbatt,
        )
        return [W >= E * minSOC / hbatt / etaRTE / etapack * g, Volbatt >= E / vbatt]


class SolarCells(Model):
    "solar cell model"

    rhosolar = Var("kg/m^2", "solar cell area density", value=0.3)
    S = Var("ft**2", "solar cell area")
    W = Var("lbf", "solar cell weight")
    etasolar = Var("-", "solar cell efficiency", value=0.2)
    mfac = Var("-", "solar cell area margin", value=1.0)

    def setup(self):
        return [self.W >= self.rhosolar * self.S * g]


class FlightState(Model):
    "Flight State (wind speed, solar irradiance, atmosphere)"

    Vwind = Var("m/s", "wind velocity")
    V = Var("m/s", "true airspeed")
    rho = Var("kg/m^3", "air density")
    mu = Var("N*s/m^2", "viscosity", value=1.42e-5)
    PSmin = Var("W/m^2", "minimum necessary solar power")
    ESday = Var("W*hr/m^2", "solar cells energy during daytime")
    EStwi = Var("W*hr/m^2", "twilight required battery energy")
    ESvar = Var("W*hr/m^2", "energy units variable", value=1)
    PSvar = Var("W/m^2", "power units variable", value=1)
    pct = Var("-", "percentile wind speeds", value=0.9)
    Vwindref = Var("m/s", "reference wind speed", value=100.0)
    rhoref = Var("kg/m^3", "reference air density", value=1.0)
    mfac = Var("-", "wind speed margin factor", value=1.0)
    rhosl = Var("kg/m^3", "sea level air density", value=1.225)
    Vne = Var("m/s", "never exceed speed at altitude")
    qne = Var("kg/s^2/m", "never exceed dynamic pressure")
    Nfac = Var("-", "factor on Vne", value=1.4)

    def setup(self, latitude, day, esirr, tn):
        # ESirr and tnight depend on args passed at construction time
        self.ESirr = Variable("ESirr", esirr, "W*hr/m^2", "solar energy")
        self.tnight = Variable("tnight", tn, "hr", "night duration")

        Vwind, V, rho, mfac, PSmin = self.Vwind, self.V, self.rho, self.mfac, self.PSmin
        ESday, EStwi, ESvar, PSvar = self.ESday, self.EStwi, self.ESvar, self.PSvar
        Vwindref, rhoref, pct = self.Vwindref, self.rhoref, self.pct
        Vne, qne, Nfac = self.Vne, self.qne, self.Nfac

        month = get_month(day)
        df = pd.read_csv(
            path + sep + "windfits" + month + "/windaltfit_lat%d.csv" % latitude
        ).to_dict(orient="records")[0]
        with contextlib.redirect_stdout(io.StringIO()):
            dft, dfd = twi_fits(latitude, day, gen=True)

        return [
            V / mfac >= Vwind,
            FCS(df, Vwind / Vwindref, [rho / rhoref, pct], name="wind"),
            FCS(dfd, ESday / ESvar, [PSmin / PSvar]),
            FCS(dft, EStwi / ESvar, [PSmin / PSvar]),
            Vne == Nfac * V,
            qne == 0.5 * rho * Vne**2,
        ]


class FlightSegment(Model):
    """Flight Segment"""

    def setup(self, aircraft, latitude=35, day=355):
        self.latitude = latitude
        self.day = day
        esirr, _, tn, _ = get_Eirr(latitude, day)

        self.aircraft = aircraft
        self.fs = FlightState(latitude, day, esirr, tn)
        self.aircraftPerf = self.aircraft.flight_model(aircraft, self.fs, False)
        self.slf = SteadyLevelFlight(self.fs, self.aircraft, self.aircraftPerf)

        if aircraft.Npod != 0 and aircraft.Npod != 1:
            assert self.aircraft.sp
            loadsp = self.aircraft.sp
        else:
            loadsp = False

        self.wingg = self.aircraft.wing.spar.loading(
            self.aircraft.wing, self.fs, out=loadsp
        )
        self.winggust = self.aircraft.wing.spar.gustloading(
            self.aircraft.wing, self.fs, out=loadsp
        )
        self.htailg = self.aircraft.emp.htail.spar.loading(
            self.aircraft.emp.htail, self.fs
        )
        self.vtailg = self.aircraft.emp.vtail.spar.loading(
            self.aircraft.emp.vtail, self.fs
        )

        self.tbhbend = self.aircraft.emp.tailboom.tailLoad(
            self.aircraft.emp.tailboom, self.aircraft.emp.htail, self.fs
        )
        self.tbvbend = self.aircraft.emp.tailboom.tailLoad(
            self.aircraft.emp.tailboom, self.aircraft.emp.vtail, self.fs
        )

        self.loading = [
            self.wingg,
            self.winggust,
            self.htailg,
            self.vtailg,
            self.tbhbend,
            self.tbvbend,
        ]

        if self.aircraft.sp:
            self.tbflex = TailBoomFlexibility(
                self.aircraft.emp.htail, self.tbhbend, self.aircraft.wing
            )
            self.tbflex.substitutions[self.tbflex.SMcorr] = 0.05
            self.loading.append(self.tbflex)

        self.wingg.substitutions[self.wingg.Nmax] = 2
        self.wingg.substitutions[self.wingg.Nsafety] = 1.5
        self.winggust.substitutions[self.winggust.vgust] = 5
        self.winggust.substitutions[self.winggust.Nmax] = 2
        self.winggust.substitutions[self.winggust.Nsafety] = 1.5
        self.tbhbend.substitutions[self.tbhbend.Nsafety] = 1.5
        self.tbvbend.substitutions[self.tbvbend.Nsafety] = 1.5

        Sh = self.aircraft.emp.htail.planform.S
        CLhmax = self.aircraft.emp.htail.planform.CLmax
        Sv = self.aircraft.emp.vtail.planform.S
        CLvmax = self.aircraft.emp.vtail.planform.CLmax
        qne = self.fs.qne

        constraints = [
            self.aircraft.Wcent == self.wingg.W,
            self.aircraft.Wcent == self.winggust.W,
            self.aircraft.Wwing == self.winggust.Ww,
            self.fs.V == self.winggust.v,
            self.aircraftPerf.CL == self.winggust.cl,
            self.htailg.W == qne * Sh * CLhmax,
            self.vtailg.W == qne * Sv * CLvmax,
        ]

        if self.aircraft.Npod != 0 and self.aircraft.Npod != 1:
            Nwing, Npod = self.aircraft.wing.N, self.aircraft.Npod
            ypod = Nwing / ((Npod - 1) / 2 + 1)
            ypods = [ypod * n for n in range(1, (Npod - 1) // 2 + 1)]
            Sgust, Mgust = self.winggust.S, self.winggust.M
            qgust, Sg, Mg = self.winggust.q, self.wingg.S, self.wingg.M
            qg = self.wingg.q
            deta = self.aircraft.wing.planform.deta
            b = self.aircraft.wing.planform.b
            weight = self.aircraft.battery.W / Npod * self.wingg.N
            for i in range(Nwing - 1):
                if i in ypods:
                    with SignomialsEnabled():
                        constraints.extend(
                            [
                                Sgust[i]
                                >= (
                                    Sgust[i + 1]
                                    + 0.5
                                    * deta[i]
                                    * (b / 2)
                                    * (qgust[i] + qgust[i + 1])
                                    - weight
                                ),
                                Sg[i]
                                >= (
                                    Sg[i + 1]
                                    + 0.5 * deta[i] * (b / 2) * (qg[i] + qg[i + 1])
                                    - weight
                                ),
                                Mgust[i]
                                >= (
                                    Mgust[i + 1]
                                    + 0.5
                                    * deta[i]
                                    * (b / 2)
                                    * (Sgust[i] + Sgust[i + 1])
                                ),
                                Mg[i]
                                >= (
                                    Mg[i + 1]
                                    + 0.5 * deta[i] * (b / 2) * (Sg[i] + Sg[i + 1])
                                ),
                            ]
                        )
                else:
                    constraints.extend(
                        [
                            Sgust[i]
                            >= (
                                Sgust[i + 1]
                                + 0.5 * deta[i] * (b / 2) * (qgust[i] + qgust[i + 1])
                            ),
                            Sg[i]
                            >= Sg[i + 1]
                            + 0.5 * deta[i] * (b / 2) * (qg[i] + qg[i + 1]),
                            Mgust[i]
                            >= (
                                Mgust[i + 1]
                                + 0.5 * deta[i] * (b / 2) * (Sgust[i] + Sgust[i + 1])
                            ),
                            Mg[i]
                            >= Mg[i + 1]
                            + 0.5 * deta[i] * (b / 2) * (Sg[i] + Sg[i + 1]),
                        ]
                    )

        self.submodels = [self.fs, self.aircraftPerf, self.slf, self.loading]

        return constraints, self.submodels


class Climb(Model):
    "Climb model"

    h = Var("ft", "climb altitude", value=60000)
    t = Var("min", "time to climb", value=500)
    hdotmin = Var("ft/min", "minimum climb rate")
    mu = Var("N*s/m^2", "viscosity", value=1.42e-5)

    def density(self, c):
        "find air density"
        alpha = 0.0065  # K/m
        h11k, T11k, p11k, rhosl = 11019, 216.483, 22532, 1.225  # m, K, Pa, kg/m^3
        T0, R, gms, n = 288.16, 287.04, 9.81, 5.2561  # K, m^2/K/s^2, m/s^2, -
        h_m = float(c[self.h]) * float(self.h.key.units.to("m").magnitude)
        hrange = [h_m * i / (self.N + 1) for i in range(1, self.N + 1)]
        rho = []
        for al in hrange:
            if al < h11k:
                T = T0 - alpha * al
                rho.append(rhosl * (T / T0) ** (n - 1))
            else:
                p = p11k * exp((h11k - al) * gms / R / T11k)
                rho.append(p / R / T11k)
        return array([rho]) * ureg("kg/m^3")

    def hstep(self, c):
        "find delta altitude"
        return float(c[self.h]) / self.N * ureg("ft")

    def setup(self, N, aircraft):
        self.N = N

        # scalar with linked function
        dh = self.dh = Variable("dh", self.hstep, "ft", "change in altitude")

        # vectors of shape (1, N) — used by AircraftDrag called with self as state
        dt = self.dt = VectorVariable((1, N), "dt", "min", "time step")
        V = self.V = VectorVariable((1, N), "V", "m/s", "vehicle speed")
        hdot = self.hdot = VectorVariable((1, N), "hdot", "ft/min", "climb rate")
        rho = self.rho = VectorVariable(
            (1, N), "rho", self.density, "kg/m^3", "air density"
        )

        with Vectorize(self.N):
            self.drag = AircraftDrag(aircraft, self)

        Wtotal = self.Wtotal = aircraft.Wtotal
        CD = self.CD = self.drag.CD
        CL = self.CL = self.drag.CL
        S = self.S = aircraft.wing.planform.S
        E = aircraft.battery.E
        Poper = self.drag.Poper
        T = self.drag.T

        constraints = [
            Wtotal <= 0.5 * rho * V**2 * CL * S,
            T >= 0.5 * rho * V**2 * CD * S + Wtotal * hdot / V,
            hdot >= dh / dt,
            self.t >= sum(hstack(dt)),
            E >= sum(hstack(Poper * dt)),
        ]

        return self.drag, constraints


class SteadyLevelFlight(Model):
    """steady level flight model"""

    def setup(self, state, aircraft, perf):
        Wtotal = self.Wtotal = aircraft.Wtotal
        CL = self.CL = perf.CL
        CD = self.CD = perf.CD
        S = self.S = aircraft.wing.planform.S
        rho = self.rho = state.rho
        V = self.V = state.V
        T = perf.drag.T

        return [Wtotal <= (0.5 * rho * V**2 * CL * S), T >= 0.5 * rho * V**2 * CD * S]


class Mission(Model):
    "define mission for aircraft"

    def setup(self, aircraft, latitude=range(1, 21, 1), day=355):

        self.aircraft = aircraft
        self.mission = []
        self.mission.append(Climb(5, self.aircraft))
        if day == 355 or day == 172:
            for l in latitude:
                self.mission.append(FlightSegment(self.aircraft, l, day))
        else:
            assert day < 172
            for l in latitude:
                self.mission.append(FlightSegment(self.aircraft, l, day))
                self.mission.append(FlightSegment(self.aircraft, l, 355 - 10 - day))

        return self.mission, self.aircraft


def test():
    "test model for continuous integration"
    v = Aircraft(sp=False)
    m = Mission(v, latitude=[20])
    m.cost = m[m.aircraft.Wtotal]
    m.solve()
    v = Aircraft(sp=True)
    m = Mission(v, latitude=[20])
    m.cost = m[m.aircraft.Wtotal]
    m.localsolve()
    v = Aircraft(Npod=3, sp=True)
    m = Mission(v, latitude=[20])
    m.cost = m[m.aircraft.Wtotal]
    f = relaxed_constants(M)
    s = f.localsolve()
    post_process(s)


if __name__ == "__main__":
    SP = True
    Vehicle = Aircraft(Npod=3, sp=SP)
    M = Mission(Vehicle, latitude=[20])
    M.cost = M[M.aircraft.Wtotal]
    try:
        sol = M.localsolve("mosek") if SP else M.solve("mosek")
    except RuntimeWarning:
        V2 = Aircraft(Npod=3, sp=SP)
        M2 = Mission(V2, latitude=[20])
        M2.cost = M2[M2.aircraft.Wtotal]
        feas = relaxed_constants(M2)
        sol = feas.localsolve("mosek")
        vks = post_process(sol)
