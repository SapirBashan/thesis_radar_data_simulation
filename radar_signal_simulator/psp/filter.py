from typing import List
import numpy as np

from config.submode_config import SubmodeTemplate
from config.operating_submode import OperatingSubmode
from config.target import Target

C = 299_792_458.0


def _wrap_angle_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _angle_diff_deg(a: float, b: float) -> float:
    return _wrap_angle_deg(a - b)


def filter_target_submode(
    targets: List[Target],
    submode: SubmodeTemplate,
    op: OperatingSubmode
) -> List[Target]:
    half_az = submode.azimuth_width / 2.0
    half_el = submode.elevation_width / 2.0
    max_range_m = submode.num_range_gates * C / (2.0 * submode.fs)

    filtered_targets = []
    for tgt in targets:
        r, az, el = tgt.polar_position
        daz = _angle_diff_deg(az, op.look_az)
        delv = _angle_diff_deg(el, op.look_el)

        if r > max_range_m:
            continue

        if abs(daz) <= half_az and abs(delv) <= half_el:
            filtered_targets.append(tgt)

    return filtered_targets