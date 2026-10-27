from dataclasses import dataclass
import json
from typing import List, Tuple

@dataclass
class ArrayConfig:
    rows: int
    cols: int
    dx: float
    dy: float

@dataclass
class RadarConfig:
    fc: float
    tx_power: float
    tx_gain: float
    rx_gain: float
    array: ArrayConfig
    azimuth_range: Tuple[float, float]
    elevation_range: Tuple[float, float]

    @property
    def wavelength(self) -> float:
        c = 299792458.0
        return c / self.fc

    @staticmethod
    def load(path: str) -> "RadarConfig":
        with open(path, "r") as f:
            d = json.load(f)
        d["array"] = ArrayConfig(**d["array"])
        return RadarConfig(**d)