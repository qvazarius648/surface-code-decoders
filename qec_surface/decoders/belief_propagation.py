"""
Belief Propagation and BP+OSD decoders via the ldpc library.
 
Belief Propagation (BP) is the standard decoder for classical LDPC codes,
widely used in e.g. 5G communications. For quantum surface codes it has a
fundamental limitation: the Tanner graph contains many short 4-cycles (each
data qubit participates in 2 X-type and 2 Z-type stabilizers), which violates
the locally tree-like assumption that BP requires to converge correctly.
 
This is a known theoretical problem, not an implementation bug. The consequence
is that BP logical error rate *increases* with code distance on surface codes —
the opposite of correct QEC behavior.
 
Ordered Statistics Decoding (OSD) is a post-processing step that rescues BP
when it fails on cycles. After BP, OSD performs Gaussian elimination on the
most reliable bits and searches for the most likely correction. BP+OSD achieves
near-MWPM threshold performance at the cost of higher computational overhead.
 
Expected thresholds on rotated surface code (uniform depolarizing noise):
    BP alone:   ~0.3–0.5%  (well below MWPM)
    BP+OSD:     ~0.9–1.0%  (close to MWPM)
 
References:
    Panteleev & Kalachev, arXiv:2103.06309  (BP+OSD for quantum LDPC codes)
    Roffe et al., arXiv:2005.07016          (ldpc library)
"""

import numpy as np
import stim

from .base import BaseDecoder


def _dem_to_parity_check_matrices(dem: stim.DetectorErrorModel):
    """
    Convert a stim DetectorErrorModel to parity check matrices.
 
    Each error mechanism in the DEM defines a column in the matrices below.
 
    Returns:
        H:          (n_detectors, n_errors) uint8 array.
                    H[i, j] = 1 if error mechanism j flips detector i.
        obs_matrix: (n_observables, n_errors) uint8 array.
                    obs_matrix[k, j] = 1 if error mechanism j flips observable k.
        probs:      (n_errors,) float array of error probabilities.
    """
    n_detectors = dem.num_detectors
    n_observables = dem.num_observables
    error_mechanisms = []

    def _process_instruction(instruction):
        if isinstance(instruction, stim.DemRepeatBlock):
            for _ in range(instruction.repeat_count):
                for inner in instruction.body_copy():
                    _process_instruction(inner)
        elif instruction.type == "error":
            prob = instruction.args_copy()[0]
            detector_ids = []
            observable_ids = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detector_ids.append(target.val)
                elif target.is_logical_observable_id():
                    observable_ids.append(target.val)
            error_mechanisms.append((prob, detector_ids, observable_ids))

    for instruction in dem:
        _process_instruction(instruction)

    n_errors = len(error_mechanisms)
    H = np.zeros((n_detectors, n_errors), dtype=np.uint8)
    obs_matrix = np.zeros((n_observables, n_errors), dtype=np.uint8)
    probs = np.zeros(n_errors, dtype=float)

    for j, (prob, det_ids, obs_ids) in enumerate(error_mechanisms):
        probs[j] = prob
        for d in det_ids:
            if d < n_detectors:
                H[d, j] = 1
        for o in obs_ids:
            if o < n_observables:
                obs_matrix[o, j] = 1

    return H, obs_matrix, probs


class BeliefPropagationDecoder(BaseDecoder):
    """
    Pure Belief Propagation decoder without OSD post-processing.
 
    Intentionally omits OSD so that BP's degradation on surface codes is
    visible in threshold plots. At large code distances, this decoder performs
    worse than random guessing — a direct demonstration of the short-cycle
    problem in the surface code Tanner graph.
 
    For near-MWPM performance, use BPOSDDecoder instead.
 
    Requires: pip install ldpc
    """

    def _build(self, dem: stim.DetectorErrorModel) -> None:
        try:
            from ldpc import BpDecoder
        except ImportError:
            raise ImportError(
                "ldpc library not found. Install with: pip install ldpc"
            )

        self._H, self._obs_matrix, self._probs = _dem_to_parity_check_matrices(dem)
        self._n_observables = dem.num_observables

        self._decoder = BpDecoder(
            self._H,
            error_rate=float(np.mean(self._probs)),
            channel_probs=self._probs,
            max_iter=self._H.shape[1],
            bp_method="product_sum",
        )

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        n_samples = detectors.shape[0]
        predicted = np.zeros((n_samples, self._n_observables), dtype=bool)

        for i in range(n_samples):
            syndrome = detectors[i].astype(np.uint8)
            correction = self._decoder.decode(syndrome)
            for o in range(self._n_observables):
                predicted[i, o] = bool(np.dot(self._obs_matrix[o], correction) % 2)

        return predicted

    @property
    def name(self) -> str:
        return "BP"


class BPOSDDecoder(BaseDecoder):
    """
    Belief Propagation + Ordered Statistics Decoding decoder.
 
    After BP converges (or fails to converge), OSD post-processing applies
    Gaussian elimination on the most reliable bit positions and performs a
    column-sweep search (osd_cs) to find the most likely correction. This
    rescues the cases where BP is misled by short cycles.
 
    Args:
        dem:       Detector error model from stim.
        osd_order: Search depth for OSD column sweep.
                   order=0 is fast with a small threshold penalty.
                   order=2 approaches MWPM performance (~10x slower than BP alone).
 
    Requires: pip install ldpc
 
    Reference: Panteleev & Kalachev, arXiv:2103.06309
    """

    def __init__(self, dem: stim.DetectorErrorModel, osd_order: int = 2):
        self._osd_order = osd_order
        super().__init__(dem)

    def _build(self, dem: stim.DetectorErrorModel) -> None:
        try:
            from ldpc import BpOsdDecoder
        except ImportError:
            raise ImportError(
                "ldpc library not found. Install with: pip install ldpc"
            )

        self._H, self._obs_matrix, self._probs = _dem_to_parity_check_matrices(dem)
        self._n_observables = dem.num_observables

        self._decoder = BpOsdDecoder(
            self._H,
            error_rate=float(np.mean(self._probs)),
            channel_probs=self._probs,
            max_iter=self._H.shape[1],
            bp_method="product_sum",
            osd_method="osd_cs",
            osd_order=self._osd_order,
        )

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        n_samples = detectors.shape[0]
        predicted = np.zeros((n_samples, self._n_observables), dtype=bool)

        for i in range(n_samples):
            syndrome = detectors[i].astype(np.uint8)
            correction = self._decoder.decode(syndrome)
            for o in range(self._n_observables):
                predicted[i, o] = bool(np.dot(self._obs_matrix[o], correction) % 2)

        return predicted

    @property
    def name(self) -> str:
        return f"BP+OSD(order={self._osd_order})"