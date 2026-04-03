import numpy as np

from .detections import Detection
from config.target import Target

def count_correct_detections(
    targets: list[Target],
    detections: list[Detection],
    *,
    max_range_m: float = 100.0,
) -> int:
    correct_count = 0

    for d in detections:
        for t in targets:
            tr, taz, tel = t.polar_position
            tv = t.radial_velocity

            # Compute differences
            dr = abs(d.range - tr)
            dv = abs(d.velocity - tv)
            daz = abs((d.azimuth - taz + 180) % 360 - 180)  # wrap azimuth to [-180, 180]
            delv = abs(d.elevation - tel)

            # Check if within tolerances
            if dr <= max_range_m:
                correct_count += 1
                break  # Stop after finding the first match for this detection

    return correct_count