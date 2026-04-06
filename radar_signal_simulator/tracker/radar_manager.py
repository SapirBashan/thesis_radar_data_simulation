from typing import Optional, List, Tuple

from config import RadarConfig, SubmodeLibrary, SubmodeTemplate, Target
from psp import CFARParams, psp_process
from radar_signal.generator import RadarSimulator
from tracker.beam import Beam
from .beam import Beam
import numpy as np
import heapq

def _bounded_centers(min_deg, max_deg, width_deg, step_deg) -> np.ndarray:
    # keep centers so the beam (width) stays inside radar limits
    half = width_deg / 2
    start = min_deg + half
    stop = max_deg - half
    if stop < start:
        return np.array([])  # no valid centers
    return np.arange(start, stop + 1e-9, step_deg)


def _build_coarse_grid(radar_cfg: RadarConfig, coarse_sm: SubmodeTemplate) -> List[Tuple[float, float]]:
    az_min, az_max = radar_cfg.azimuth_range
    el_min, el_max = radar_cfg.elevation_range
    az_step = coarse_sm.azimuth_width
    el_step = coarse_sm.elevation_width

    az_centers = _bounded_centers(az_min, az_max, coarse_sm.azimuth_width, az_step)
    el_centers = _bounded_centers(el_min, el_max, coarse_sm.elevation_width, el_step)

    return [(az, el) for el in el_centers for az in az_centers]


def _build_fine_grid(center_az: float, center_el: float, coarse_sm: SubmodeTemplate) -> List[Tuple[float, float]]:
    az_span = coarse_sm.azimuth_width
    el_span = coarse_sm.elevation_width
    az_step = az_span / 4.0
    el_step = el_span / 4.0

    az_start = center_az - az_span / 2 + az_step / 2
    el_start = center_el - el_span / 2 + el_step / 2

    az_centers = [az_start + i * az_step for i in range(4)]
    el_centers = [el_start + i * el_step for i in range(4)]

    return [(az, el) for el in el_centers for az in az_centers]

def _build_confirm_grid(center_az: float, center_el: float, fine_sm: SubmodeTemplate) -> List[Tuple[float, float]]:
    az_span = fine_sm.azimuth_width
    el_span = fine_sm.elevation_width

    az_step = az_span / 4.0
    el_step = el_span / 4.0

    az_start = center_az - az_span / 2 + az_step / 2
    el_start = center_el - el_span / 2 + el_step / 2

    az_centers = [az_start + i * az_step for i in range(4)]
    el_centers = [el_start + i * el_step for i in range(4)]

    return [(az, el) for el in el_centers for az in az_centers]


class RadarManager:
    """
    This class will manage the Radar tracker.
    """
    SBM_COARSE: str = "coarse"
    SBM_FINE: str = "fine"
    SBM_CONFIRM: str = "confirm"
    SBM_TRACK: str = "track"

    PIR_COARSE: int = 1
    PIR_FINE: int = 2
    PIR_CONFIRM: int = 3
    PIR_TRACK: int = 4

    def __init__(self, radar_config: RadarConfig, submode_library: SubmodeLibrary):
        self.radar_config: RadarConfig = radar_config
        self.radar_library : SubmodeLibrary = submode_library

        # make sure it has the list of submodes
        if getattr(self.radar_config, self.SBM_FINE) is None:
            raise ValueError("Submode Library needs to have a 'fine' submode.")
        if getattr(self.radar_config, self.SBM_COARSE) is None:
            raise ValueError("Submode Library needs to have a 'coarse' submode.")
        if getattr(self.radar_config, self.SBM_CONFIRM) is None:
            raise ValueError("Submode Library needs to have a 'confirm' submode.")
        if getattr(self.radar_config, self.SBM_TRACK) is None:
            raise ValueError("Submode Library needs to have a 'track' submode.")

        # create a default coarse grid
        self.coarse_grid = _build_coarse_grid(self.radar_config, self.radar_library.submodes[self.SBM_COARSE])
        self.coarse_grid_len = len(self.coarse_grid)
        self.current_index_coarse_grid = 0

        self.beam_queue = []
        # push to the queue all the whole coarse grid
        for (az, el) in self.coarse_grid:
            beam = Beam(az=az, el=el, submode_name=self.SBM_COARSE)
            self._push_beam_queue(self.PIR_COARSE, beam)

        self.signal_generator = RadarSimulator(self.radar_config, self.radar_library)


    def _push_beam_queue(self, priority: int, beam: Beam) -> None:
        heapq.heappush(self.beam_queue, (priority, beam))

    def _pop_beam_queue(self) -> Optional[Beam]:
        if not self.beam_queue:
            return None
        return heapq.heappop(self.beam_queue)[1]

    def _next_coarse_grid(self):
        self.current_index_coarse_grid += 1
        return self.current_index_coarse_grid % self.coarse_grid_len

    def _get_fine_grid(self, coarse_grid) -> List[Tuple[float, float]]:
        return _build_fine_grid(coarse_grid[0], coarse_grid[1], self.radar_library.submodes[self.SBM_COARSE])

    def _get_confirm_grid(self, fine_grid) -> List[Tuple[float, float]]:
        return _build_confirm_grid(fine_grid[0], fine_grid[1], self.radar_library.submodes[self.SBM_FINE])

    def _get_next_beam(self) -> Optional[Beam]:
        beam = self._pop_beam_queue()
        if beam is not None:
            return beam
            # push all the coarse grid
        for (az, el) in self.coarse_grid:
            beam = Beam(az=az, el=el, submode_name=self.SBM_COARSE)
            self._push_beam_queue(self.PIR_COARSE, beam)

        beam = self._pop_beam_queue()
        return beam

    def next_step(self, targets: List[Target]):
        # state machine
        # if beam=COARSE and detection(s) -> build a fine grid
        # if beam=FINE and detection(s) -> build a confirm grid
        # if beam=CONFIRM and detection(s) -> build a track grid
        # after confirm we add the detection to track
        # and, we schedule a track beam for later
        current_beam = self._get_next_beam()
        if current_beam is None:
            raise ValueError("Beam not found.")

        sbm: SubmodeTemplate = self.radar_library.submodes[current_beam.submode_name]

        # generate the signal for the current beam
        signal = self.signal_generator.generate(targets, current_beam)

        # process the signal
        # TODO add a visualizer for the step of processing
        cfar_params: CFARParams = sbm.cfar

        detections = psp_process(signal, self.radar_config, self.radar_library, current_beam, cfar_params)

        if len(detections) == 0:
            return current_beam, []

        if current_beam.submode_name == self.SBM_COARSE:
            # create a fine grid
            fine_grid = self._get_fine_grid((current_beam.az, current_beam.el))
            for (az, el) in fine_grid:
                beam = Beam(az=az, el=el, submode_name=self.SBM_FINE)
                self._push_beam_queue(self.PIR_FINE, beam)
        if current_beam.submode_name == self.SBM_FINE:
            confirm_grid = self._get_confirm_grid((current_beam.az, current_beam.el))
            for (az, el) in confirm_grid:
                beam = Beam(az=az, el=el, submode_name=self.SBM_CONFIRM)
                self._push_beam_queue(self.PIR_CONFIRM, beam)
        if current_beam.submode_name == self.SBM_CONFIRM:
            # all detections from this beam should be added to track
            # TODO: add to track
            # TODO: schedule a track beam for later
            pass
        if current_beam.submode_name == self.SBM_TRACK:
            # update track position according to the tracker
            pass

        return current_beam, detections