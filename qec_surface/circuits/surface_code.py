"""
Surface code circuit generation with configurable noise models.

Supports rotated memory experiments (standard benchmark) with
several noise configurations commonly studied in the literature.
"""

from dataclasses import dataclass, field
from typing import Literal
import stim


# --- Noise model configuration -------------------------------------------

CodeType = Literal[
    "surface_code:rotated_memory_x",
    "surface_code:rotated_memory_z",
    "surface_code:unrotated_memory_x",
    "surface_code:unrotated_memory_z",
]


@dataclass
class NoiseModel:
    """
    Noise model parameters for surface code circuit generation.

    All probabilities are per-operation unless noted.
    stim applies them as: before_round_data_depolarization,
    before_measure_flip_probability, after_reset_flip_probability.

    Attributes:
        depolarizing:   symmetric depolarizing noise on data qubits each round
        measurement:    bit-flip probability on ancilla measurement outcomes
        reset:          flip probability after qubit reset
    """
    depolarizing: float = 0.0
    measurement: float = 0.0
    reset: float = 0.0

    def __post_init__(self):
        for name, val in self.__dict__.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Probability '{name}' must be in [0, 1], got {val}")

    @classmethod
    def uniform(cls, p: float) -> "NoiseModel":
        """
        Standard uniform noise: same probability for all channels.
        Most common benchmark in the literature (e.g. Fowler et al. 2012).
        """
        return cls(depolarizing=p, measurement=p, reset=p)

    @classmethod
    def measurement_dominated(cls, p_gate: float, p_meas: float) -> "NoiseModel":
        """
        Separate gate and measurement noise.
        Relevant for platforms where readout errors dominate (e.g. some superconducting devices).
        """
        return cls(depolarizing=p_gate, measurement=p_meas, reset=p_gate)

    @classmethod
    def gate_only(cls, p: float) -> "NoiseModel":
        """
        Gate noise only, perfect measurements.
        Corresponds to the 'code-capacity' noise model studied theoretically.
        """
        return cls(depolarizing=p, measurement=0.0, reset=0.0)

    def describe(self) -> str:
        return (
            f"depol={self.depolarizing:.4f}, "
            f"meas={self.measurement:.4f}, "
            f"reset={self.reset:.4f}"
        )


# --- Circuit generation ---------------------------------------------------

@dataclass
class SurfaceCodeCircuit:
    """
    A surface code memory experiment circuit with metadata.

    Wraps stim.Circuit and keeps the configuration that produced it,
    so downstream code can always know what it's working with.
    """
    circuit: stim.Circuit
    distance: int
    rounds: int
    noise: NoiseModel
    code_type: CodeType

    @property
    def n_data_qubits(self) -> int:
        return self.distance ** 2

    @property
    def n_ancilla_qubits(self) -> int:
        return self.distance ** 2 - 1

    def detector_error_model(self, decompose_errors: bool = True) -> stim.DetectorErrorModel:
        return self.circuit.detector_error_model(decompose_errors=decompose_errors)

    def compile_sampler(self):
        return self.circuit.compile_detector_sampler()

    def __repr__(self) -> str:
        return (
            f"SurfaceCodeCircuit("
            f"type={self.code_type}, d={self.distance}, "
            f"rounds={self.rounds}, noise=[{self.noise.describe()}])"
        )


def build_surface_code(
    distance: int,
    rounds: int,
    noise: NoiseModel,
    code_type: CodeType = "surface_code:rotated_memory_x",
) -> SurfaceCodeCircuit:
    """
    Generate a surface code memory experiment circuit.

    Args:
        distance:  Code distance d. The rotated surface code uses d^2 data qubits
                   and d^2 - 1 ancilla qubits. Logical error rate suppression
                   scales as (p/p_th)^((d+1)/2).
        rounds:    Number of syndrome measurement rounds. Should be >= d
                   for a meaningful space-time decoding problem.
        noise:     NoiseModel instance specifying error probabilities.
        code_type: Which surface code variant and logical basis to use.
                   'rotated_memory_x' is standard for most benchmarks.

    Returns:
        SurfaceCodeCircuit with the generated stim circuit and metadata.

    Notes:
        The 'rotated' surface code (also called planar code in some literature)
        has ~half the qubit overhead of the 'unrotated' version for the same distance,
        which is why it's the standard for hardware experiments.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"Distance must be an odd integer >= 3, got {distance}")
    if rounds < 1:
        raise ValueError(f"Rounds must be >= 1, got {rounds}")

    circuit = stim.Circuit.generated(
        code_type,
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=noise.depolarizing,
        before_measure_flip_probability=noise.measurement,
        after_reset_flip_probability=noise.reset,
    )

    return SurfaceCodeCircuit(
        circuit=circuit,
        distance=distance,
        rounds=rounds,
        noise=noise,
        code_type=code_type,
    )
