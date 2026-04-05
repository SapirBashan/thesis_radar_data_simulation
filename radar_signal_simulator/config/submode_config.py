from dataclasses import dataclass
import json

@dataclass
class CFARParams:
    guard_cells: int
    training_cells: int
    threshold: float

@dataclass
class SubmodeTemplate:
    fs: float
    prf: float
    pulse_width: float
    num_pulses: int
    num_range_gates: int
    azimuth_width: float
    elevation_width: float
    time: float
    cfar: CFARParams

    @property
    def pri(self) -> float:
        return 1.0 / self.prf

    @property
    def num_doppler_bins(self) -> int:
        # Doppler FFT length = number of pulses
        return self.num_pulses

    @property
    def nrg(self) -> int:
        return self.num_range_gates

    @property
    def npri(self) -> int:
        return self.num_pulses

@dataclass
class SubmodeLibrary:
    submodes: dict

    @staticmethod
    def load(path: str) -> "SubmodeLibrary":
        with open(path, "r") as f:
            d = json.load(f)
        submodes = {k: SubmodeTemplate(**v) for k, v in d.items()}
        return SubmodeLibrary(submodes=submodes)