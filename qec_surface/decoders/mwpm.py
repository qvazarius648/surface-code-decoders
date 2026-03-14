"""
MWPM decoder using pymatching.

Minimum Weight Perfect Matching is the standard decoder for surface codes.
It finds the minimum-weight correction consistent with the observed syndrome
by solving a matching problem on the detector error model graph.
"""

import numpy as np
import stim
import pymatching

from .base import BaseDecoder


class MWPMDecoder(BaseDecoder):
    """
    Minimum Weight Perfect Matching decoder via pymatching.

    Complexity: O(n^3) in general, but pymatching uses efficient
    sparse implementations that scale well for surface codes in practice.

    Threshold: ~1% for depolarizing circuit-level noise on rotated surface code.
    """

    def _build(self, dem: stim.DetectorErrorModel) -> None:
        self._matcher = pymatching.Matching.from_detector_error_model(dem)

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        return self._matcher.decode_batch(detectors)

    @property
    def name(self) -> str:
        return "MWPM"
