"""
Union-Find decoder via pymatching v2.

Union-Find (also called "almost-linear" decoder) was introduced by
Delfosse & Nickerson (2021) as a near-linear time alternative to MWPM.

Trade-off vs MWPM:
- Faster: O(n * alpha(n)) vs O(n^3) in theory, meaningfully faster in practice
- Slightly worse threshold: ~0.9% vs ~1.0% for depolarizing circuit-level noise
- Less accurate per shot at low noise, comparable near threshold

This makes it relevant for real-time decoding where latency matters
(e.g. avoiding backlog in a continuously running quantum computer).

Reference: Delfosse & Nickerson, arXiv:2101.09310
"""

import numpy as np
import stim
import pymatching

from .base import BaseDecoder


class UnionFindDecoder(BaseDecoder):
    """
    Union-Find decoder using pymatching v2.

    pymatching v2 exposes Union-Find via the 'num_neighbours' parameter
    in Matching.decode_batch(). Setting num_neighbours=1 activates the
    Union-Find algorithm instead of MWPM.

    Note: requires pymatching >= 2.0.0
    """

    def _build(self, dem: stim.DetectorErrorModel) -> None:
        self._matcher = pymatching.Matching.from_detector_error_model(dem)
        # Verify pymatching version supports num_neighbours
        import pymatching as pm
        version = tuple(int(x) for x in pm.__version__.split(".")[:2])
        if version < (2, 0):
            raise RuntimeError(
                f"UnionFindDecoder requires pymatching >= 2.0.0, "
                f"found {pm.__version__}"
            )

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        # num_neighbours=1 activates Union-Find path in pymatching v2
        return self._matcher.decode_batch(detectors, num_neighbours=1)

    @property
    def name(self) -> str:
        return "UnionFind"
