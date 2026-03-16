"""
Belief Propagation + OSD decoder via the ldpc library.

BP is the standard decoder for LDPC codes (e.g. used in classical 5G).
For surface codes it has a fundamental limitation: short cycles in the
Tanner graph cause BP to fail to converge or converge to wrong solutions.

This is not a bug — it's a known theoretical problem.
Surface codes have many 4-cycles (each data qubit touches 2 X and 2 Z
stabilizers), which violates the tree-like assumption BP relies on.

The standard fix is OSD (Ordered Statistics Decoding) as a post-processing
step after BP: BP+OSD achieves near-MWPM performance but at higher
computational cost.

For surface codes specifically:
- Pure BP:     threshold ~0.3-0.5% (well below MWPM's ~1%)
- BP+OSD:      threshold ~0.9-1.0% (close to MWPM)

Reference:
    Panteleev & Kalachev, arXiv:2103.06309 (BP+OSD for quantum LDPC)
    Roffe et al., arXiv:2005.07016 (ldpc library)
"""

import numpy as np
import stim

from .base import BaseDecoder


def _dem_to_parity_check_matrices(dem: stim.DetectorErrorModel):
    """
    Convert a stim DetectorErrorModel to parity check matrices H and observables.
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
    Belief Propagation decoder (pure BP, no OSD post-processing).

    Навмисно без OSD щоб деградація BP на surface code була видна
    на threshold plot. Це робить порівняння з MWPM фізично змістовним.

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
    Belief Propagation + Ordered Statistics Decoding.

    OSD рятує BP коли він не збігається через цикли.
    Near-MWPM threshold при вищій обчислювальній вартості.

    osd_order:
    - order=0: швидко, невелика втрата threshold
    - order=2: близько до MWPM, ~10x повільніше ніж чистий BP

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