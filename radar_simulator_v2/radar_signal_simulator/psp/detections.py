from dataclasses import dataclass


@dataclass
class Detection:
    range: float
    velocity: float
    snr: float
    azimuth: float
    elevation: float

    def __repr__(self):
        return f"(range={self.range}, velocity={self.velocity}, snr={self.snr}, azimuth={self.azimuth}, elevation={self.elevation})"