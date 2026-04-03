from dataclasses import dataclass
import json

@dataclass
class OperatingSubmode:
    submode_name: str
    look_az: float
    look_el: float

    @staticmethod
    def load(path: str) -> "OperatingSubmode":
        with open(path, "r") as f:
            d = json.load(f)
        return OperatingSubmode(**d)

    @staticmethod
    def from_values(submode_name: str, look_az: float, look_el: float) -> "OperatingSubmode":
        # real-time construction (no JSON)
        return OperatingSubmode(submode_name=submode_name, look_az=look_az, look_el=look_el)

    def update_look(self, look_az: float, look_el: float) -> None:
        # real-time update
        self.look_az = look_az
        self.look_el = look_el