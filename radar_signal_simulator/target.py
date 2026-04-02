from dataclasses import dataclass, asdict
import json

@dataclass
class Target:
    rcs: float
    range: float
    doppler: float
    azimuth: float
    elevation: float

    @staticmethod
    def save_list(targets: list["Target"], path: str) -> None:
        with open(path, "w") as f:
            json.dump([asdict(t) for t in targets], f, indent=2)

    @staticmethod
    def load_list(path: str) -> list["Target"]:
        with open(path, "r") as f:
            data = json.load(f)
        return [Target(**d) for d in data]