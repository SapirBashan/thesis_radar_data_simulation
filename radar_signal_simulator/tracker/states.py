from __future__ import annotations

from enum import Enum, auto


class TrackState(Enum):
    """
    Lifecycle of a tracked object from first detection to removal.
    """
    CANDIDATE = auto()   # seen in coarse scan, not yet trusted
    CONFIRMING = auto()   # fine/confirm beams are validating the target
    TRACKED = auto()      # confirmed target, adaptive revisit scheduling
    LOST = auto()         # missed recently, still recoverable
    DROPPED = auto()      # no longer tracked


class RadarState(Enum):
    """
    High-level mode of the radar scheduler.
    """
    COARSE_SCAN = auto()
    FINE_SCAN = auto()
    CONFIRM_SCAN = auto()
    TRACKING = auto()
    RECOVERY = auto()
    IDLE = auto()


class BeamPurpose(Enum):
    """
    What the beam is intended to do.
    """
    COARSE = auto()
    FINE = auto()
    CONFIRM = auto()
    TRACK = auto()
    RECOVERY = auto()