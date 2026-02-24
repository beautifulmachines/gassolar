"loiter segment"

from gpkit import Model, Var

from gassolar.gas.flight_segment import FlightSegment


class Loiter(Model):
    "loiter segment"

    t = Var("days", "endurance requirement")

    def setup(self, aircraft, N=5, altitude=15000, latitude=45, percent=90, day=355):
        self.fs = FlightSegment(aircraft, N, altitude, latitude, percent, day)

        return self.fs, [self.fs.be["t"] >= self.t / N]
