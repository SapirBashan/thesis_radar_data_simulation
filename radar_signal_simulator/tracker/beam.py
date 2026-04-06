from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .states import BeamPurpose


@dataclass
class Beam:
    """
    One scheduled radar beam job.
    """
    submode_name: str
    az: float
    el: float
    purpose: BeamPurpose
    track_id: Optional[str] = None

    # Optional metadata for refinement / debugging / scheduling
    priority: int = 0
    grid_level: str = ""   # e.g. "coarse", "fine", "confirm"
    beam_id: Optional[str] = None