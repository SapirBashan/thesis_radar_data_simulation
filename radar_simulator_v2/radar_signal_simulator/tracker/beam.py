from dataclasses import dataclass

@dataclass
class Beam:
    submode_name: str
    az: float
    el: float
