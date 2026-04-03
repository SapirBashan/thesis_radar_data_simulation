from typing import List
from config.submode_config import SubmodeTemplate
from config.operating_submode import OperatingSubmode
from config.target import Target

C = 299_792_458.0  # Speed of Light (m/s)

def _wrap_angle_deg(a: float) -> float:
    """Wrap angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def _angle_diff_deg(a: float, b: float) -> float:
    """Shortest signed difference a - b in degrees."""
    return _wrap_angle_deg(a - b)


def filter_target_submode(
    targets: List[Target],
    submode: SubmodeTemplate,
    op: OperatingSubmode
) -> List[Target]:
    half_az = submode.azimuth_width / 2.0
    half_el = submode.elevation_width / 2.0

    filtered_targets = []
    for tgt in targets:
        r, az, el = tgt.polar_position
        daz = _angle_diff_deg(az, op.look_az)
        delv = _angle_diff_deg(el, op.look_el)

        # Target is within FoV if azimuth and elevation differences are within bounds
        if abs(daz) <= half_az and abs(delv) <= half_el:
            filtered_targets.append(tgt)

    return filtered_targets