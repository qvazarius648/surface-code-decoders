"""
Logical error rate estimation with proper statistical treatment.

Key design decisions:
- Error bars via Wilson score interval (better than naive binomial for small counts)
- Results returned as structured dataclass, not raw floats
- Decoder receives DEM once at construction, not per sample (performance)
"""

from dataclasses import dataclass
from typing import Type
import numpy as np
import pandas as pd

from ..circuits.surface_code import SurfaceCodeCircuit, NoiseModel, build_surface_code, CodeType
from ..decoders.base import BaseDecoder


# --- Result types --------------------------------------------------------

@dataclass
class DecoderResult:
    """
    Result of a single decoder experiment.

    Includes point estimate and confidence interval for the logical error rate.
    Uses Wilson score interval which is well-behaved even for small error counts
    (unlike the naive p ± sqrt(p(1-p)/n) which can give negative bounds).
    """
    distance: int
    rounds: int
    noise_level: float          # scalar summary (e.g. depolarizing p)
    noise_description: str      # full noise model description
    decoder_name: str
    n_samples: int
    n_errors: int

    @property
    def logical_error_rate(self) -> float:
        return self.n_errors / self.n_samples

    @property
    def wilson_interval(self) -> tuple[float, float]:
        """
        95% Wilson score confidence interval.
        Reference: Wilson (1927), also see Agresti & Coull (1998).
        """
        n, k = self.n_samples, self.n_errors
        z = 1.96  # 95% confidence
        p_hat = k / n
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    @property
    def error_bar(self) -> float:
        """Symmetric error bar (half-width of Wilson interval)."""
        lo, hi = self.wilson_interval
        return (hi - lo) / 2


# --- Core experiment function --------------------------------------------

def estimate_logical_error_rate(
    sc: SurfaceCodeCircuit,
    decoder_cls: Type[BaseDecoder],
    n_samples: int,
) -> DecoderResult:
    """
    Run one experiment: sample syndromes, decode, count logical errors.

    Args:
        sc:           SurfaceCodeCircuit to sample from
        decoder_cls:  Decoder class (not instance) to instantiate
        n_samples:    Number of Monte Carlo samples

    Returns:
        DecoderResult with logical error rate and confidence interval
    """
    # Build DEM once — used both for decoder construction and sampling
    dem = sc.detector_error_model(decompose_errors=True)

    # Initialize decoder (builds internal graph from DEM)
    decoder = decoder_cls(dem)

    # Sample syndromes and true logical outcomes
    sampler = sc.compile_sampler()
    detectors, actual_observables = sampler.sample(
        n_samples, separate_observables=True
    )

    # Decode
    predicted_observables = decoder.decode_batch(detectors)

    # Count logical errors (any observable mismatch = logical failure)
    n_errors = int(np.sum(
        np.any(predicted_observables != actual_observables, axis=1)
    ))

    return DecoderResult(
        distance=sc.distance,
        rounds=sc.rounds,
        noise_level=sc.noise.depolarizing,  # primary sweep parameter
        noise_description=sc.noise.describe(),
        decoder_name=decoder.name,
        n_samples=n_samples,
        n_errors=n_errors,
    )


# --- Sweep utilities -----------------------------------------------------

def compare_decoders(
    distances: list[int],
    noise_levels: list[float],
    decoder_classes: list[Type[BaseDecoder]],
    n_samples: int,
    rounds_per_distance: int | None = None,
    code_type: CodeType = "surface_code:rotated_memory_x",
    noise_factory=NoiseModel.uniform,
) -> pd.DataFrame:
    """
    Run sweep_noise_levels for multiple decoders and combine results.

    Useful for threshold plots where you want all decoders on the same axes.
    The DEM is built once per (distance, noise_level) and shared across decoders.

    Args:
        decoder_classes: list of decoder classes to compare

    Returns:
        DataFrame with same schema as sweep_noise_levels, with decoder_name
        column distinguishing results.
    """
    all_results = []

    for decoder_cls in decoder_classes:
        print(f"\n── Decoder: {decoder_cls.__name__} ──")
        df = sweep_noise_levels(
            distances=distances,
            noise_levels=noise_levels,
            decoder_cls=decoder_cls,
            n_samples=n_samples,
            rounds_per_distance=rounds_per_distance,
            code_type=code_type,
            noise_factory=noise_factory,
        )
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def sweep_noise_levels(
    distances: list[int],
    noise_levels: list[float],
    decoder_cls: Type[BaseDecoder],
    n_samples: int,
    rounds_per_distance: int | None = None,
    code_type: CodeType = "surface_code:rotated_memory_x",
    noise_factory=NoiseModel.uniform,
) -> pd.DataFrame:
    """
    Sweep over distances and noise levels, collect results into a DataFrame.

    Args:
        distances:            List of code distances to test
        noise_levels:         List of noise probabilities to sweep
        decoder_cls:          Decoder class to use
        n_samples:            Samples per data point
        rounds_per_distance:  If None, uses rounds = distance (standard choice)
        code_type:            Surface code variant
        noise_factory:        Callable p -> NoiseModel. Default: uniform noise.

    Returns:
        DataFrame with columns: distance, rounds, noise_level, noise_description,
        decoder_name, n_samples, n_errors, logical_error_rate, error_bar
    """
    records = []

    for d in distances:
        rounds = rounds_per_distance if rounds_per_distance is not None else d
        for p in noise_levels:
            noise = noise_factory(p)
            sc = build_surface_code(d, rounds, noise, code_type)
            result = estimate_logical_error_rate(sc, decoder_cls, n_samples)
            records.append({
                "distance": result.distance,
                "rounds": result.rounds,
                "noise_level": result.noise_level,
                "noise_description": result.noise_description,
                "decoder_name": result.decoder_name,
                "n_samples": result.n_samples,
                "n_errors": result.n_errors,
                "logical_error_rate": result.logical_error_rate,
                "error_bar": result.error_bar,
            })
            print(
                f"  d={d}, p={p:.4f} → "
                f"LER={result.logical_error_rate:.4f} "
                f"± {result.error_bar:.4f} "
                f"({result.n_errors}/{result.n_samples})"
            )

    return pd.DataFrame(records)
