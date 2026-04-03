from dataclasses import dataclass
import json
import uuid
import numpy as np

C = 299792458.0

@dataclass
class Target:
    id: str
    position_xyz: list[float]  # [x, y, z] meters
    velocity_xyz: list[float]  # [vx, vy, vz] m/s
    rcs: float

    @staticmethod
    def new(position_xyz, velocity_xyz, rcs) -> "Target":
        return Target(
            id=str(uuid.uuid4()),
            position_xyz=position_xyz,
            velocity_xyz=velocity_xyz,
            rcs=rcs
        )

    @property
    def polar_position(self):
        x, y, z = self.position_xyz
        r = np.sqrt(x**2 + y**2 + z**2)
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        return r, az, el

    @property
    def radial_velocity(self):
        x, y, z = self.position_xyz
        vx, vy, vz = self.velocity_xyz
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-12
        # unit LOS vector
        ux, uy, uz = x/r, y/r, z/r
        return ux*vx + uy*vy + uz*vz

    @staticmethod
    def load_list(path: str) -> list["Target"]:
        with open(path, "r") as f:
            data = json.load(f)
        return [Target(**d) for d in data]

    @staticmethod
    def save_list(targets: list["Target"], path: str) -> None:
        with open(path, "w") as f:
            json.dump([t.__dict__ for t in targets], f, indent=2)