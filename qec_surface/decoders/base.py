"""
Base decoder interface for surface code decoding.

Any decoder must inherit BaseDecoder and implement decode_batch().
This ensures all decoders are interchangeable in the benchmark pipeline.
"""

from abc import ABC, abstractmethod
import numpy as np
import stim


class BaseDecoder(ABC):
    """
    Abstract base class for all QEC decoders.

    The contract is simple:
    - Input:  detector measurements (bool array) + detector error model
    - Output: predicted logical observable flips (bool array)

    This mirrors the stim/pymatching convention so decoders can wrap
    existing libraries with minimal boilerplate.
    """

    def __init__(self, dem: stim.DetectorErrorModel):
        """
        Initialize decoder from a DetectorErrorModel.

        Args:
            dem: stim DetectorErrorModel describing the circuit's error mechanisms.
                 Decoders use this to build their internal decoding graph.
        """
        self.dem = dem
        self._build(dem)

    @abstractmethod
    def _build(self, dem: stim.DetectorErrorModel) -> None:
        """
        Build internal decoder structure from the DEM.
        Called once at initialization.
        """
        ...

    @abstractmethod
    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        """
        Decode a batch of syndrome measurements.

        Args:
            detectors: bool array of shape (n_samples, n_detectors)
                       Each row is one syndrome measurement outcome.

        Returns:
            predicted_observables: bool array of shape (n_samples, n_observables)
                                   Predicted logical flip for each sample.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable decoder name for plots and logs."""
        return self.__class__.__name__
