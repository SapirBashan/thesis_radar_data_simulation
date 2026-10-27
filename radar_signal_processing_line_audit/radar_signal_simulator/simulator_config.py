from dataclasses import dataclass, asdict
import json

@dataclass
class ArrayConfig:
    rows: int
    cols: int
    dx: float
    dy: float

@dataclass
class SubmodeConfig:
    num_range_gates: int
    num_doppler_bins: int

    @property
    def npri(self) -> int:
        return self.num_doppler_bins

    @property
    def nrg(self) -> int:
        return self.num_range_gates

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
    noise_level: float

    submodes: dict
    active_submode: str

    array: ArrayConfig

    @property
    def pri(self) -> float:
        return 1.0 / self.prf

    @property
    def wavelength(self) -> float:
        c = 299792458.0
        return c / self.fc

    @property
    def submode(self) -> SubmodeConfig:
        sm = self.submodes[self.active_submode]
        return SubmodeConfig(**sm)

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