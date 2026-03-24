"""
Minimum Weight Perfect Matching decoder via pymatching.
 
MWPM is the standard decoder for surface codes. Given a syndrome (the set of
stabilizer measurements that fired), it constructs a graph where nodes are
detector events and edge weights are log-likelihood ratios of the corresponding
error mechanisms. The decoder then finds the minimum-weight perfect matching on
this graph — the set of error hypotheses that is most consistent with the
observed syndrome and has the lowest total weight.
 
The matching is solved using the blossom algorithm (Edmonds 1965), which finds
the globally optimal solution. pymatching implements a sparse variant that
runs in near-linear time for the sparse graphs that arise from surface codes.
 
Expected threshold: ~1.0% for uniform depolarizing circuit-level noise on the
rotated surface code (Fowler et al., arXiv:1208.0928).
 
Reference:
    Higgott & Gidney, arXiv:2303.15933  (pymatching v2, sparse blossom)
"""

import numpy as np
import stim
import pymatching

from .base import BaseDecoder


class MWPMDecoder(BaseDecoder):
    """
    Minimum Weight Perfect Matching decoder via pymatching.
 
    Finds the globally optimal correction for each syndrome by solving a
    minimum-weight perfect matching problem on the detector error model graph.
    This makes MWPM the performance baseline against which other decoders
    are compared.
 
    Threshold: ~1.0% for uniform depolarizing circuit-level noise.
    Complexity: near-linear in the number of detectors via sparse blossom.
    """

    def _build(self, dem: stim.DetectorErrorModel) -> None:
        self._matcher = pymatching.Matching.from_detector_error_model(dem)

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        return self._matcher.decode_batch(detectors)

    @property
    def name(self) -> str:
        return "MWPM"
