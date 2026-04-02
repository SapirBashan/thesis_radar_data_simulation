from dataclasses import dataclass, asdict
import json

@dataclass
class ArrayConfig:
    rows: int
    cols: int
    dx: float
    dy: float

@dataclass
class RadarConfig:
    fc: float
    prf: float
    pulse_width: float
    fs: float
    num_pulses: int
    tx_power: float
    tx_gain: float
    rx_gain: float
    noise_figure: float
    noise_level: float
    num_range_gates: int
    num_doppler_bins: int
    array: ArrayConfig

    @property
    def pri(self) -> float:
        return 1.0 / self.prf

    @property
    def wavelength(self) -> float:
        c = 299792458.0
        return c / self.fc

    @staticmethod
    def save(cfg, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    @staticmethod
    def load(path: str) -> "RadarConfig":
        with open(path, "r") as f:
            d = json.load(f)
        d["array"] = ArrayConfig(**d["array"])
        return RadarConfig(**d)