"""
Logical error rate estimation with statistical confidence intervals.
 
Design decisions:
    - Confidence intervals via Wilson score interval rather than the naive
      binomial approximation p ± sqrt(p(1-p)/n), which can produce negative
      bounds and is unreliable for small error counts.
    - Results are returned as a structured DecoderResult dataclass rather than
      raw floats, so callers always have access to counts alongside rates.
    - The DEM is built once per experiment and shared between the decoder
      constructor and the sampler, avoiding redundant computation.
    - sweep_noise_levels and compare_decoders return pandas DataFrames for
      straightforward plotting and analysis downstream.
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
    Result of a single decoder experiment at one (distance, noise_level) point.
 
    Stores raw counts rather than just the rate so that confidence intervals
    can be computed correctly. The Wilson score interval is used throughout
    because it remains well-behaved when error counts are small or zero.
 
    Attributes:
        distance:          Code distance d.
        rounds:            Number of syndrome extraction rounds.
        noise_level:       Depolarizing probability p (primary sweep parameter).
        noise_description: Full noise model description string.
        decoder_name:      Human-readable decoder identifier.
        n_samples:         Total number of Monte Carlo shots.
        n_errors:          Number of shots where a logical error occurred.
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
        95% Wilson score confidence interval for the logical error rate.
 
        Preferred over the normal approximation interval because it does not
        produce negative lower bounds and is accurate for small counts.
 
        References:
            Wilson (1927). Probable inference, the law of succession, and
                statistical inference. JASA 22(158): 209-212.
            Agresti & Coull (1998). Approximate is better than exact for
                interval estimation of binomial proportions. TAS 52(2).
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
    Run one decoder experiment: sample syndromes, decode, count logical errors.
 
    The DEM is constructed once and used both to initialize the decoder and
    to drive the syndrome sampler. A logical error is counted whenever the
    decoder's predicted observable flips disagree with the true flips on any
    observable in a given shot.
 
    Args:
        sc:          SurfaceCodeCircuit defining the code and noise model.
        decoder_cls: Decoder class to instantiate (not an instance).
        n_samples:   Number of Monte Carlo shots to sample.
 
    Returns:
        DecoderResult with logical error rate and Wilson confidence interval.
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
    Sweep over code distances and noise levels for a single decoder.
 
    Args:
        distances:           Code distances to evaluate.
        noise_levels:        Physical error rates to sweep over.
        decoder_cls:         Decoder class to use for all experiments.
        n_samples:           Monte Carlo shots per data point.
        rounds_per_distance: Syndrome extraction rounds per experiment.
                             If None, uses rounds = distance, which gives a
                             balanced space-time decoding problem.
        code_type:           Surface code variant to generate.
        noise_factory:       Callable mapping p -> NoiseModel.
                             Defaults to uniform noise across all channels.
 
    Returns:
        DataFrame with columns: distance, rounds, noise_level, noise_description,
        decoder_name, n_samples, n_errors, logical_error_rate, error_bar.
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
    Run sweep_noise_levels for multiple decoders and combine into one DataFrame.
 
    Iterates over decoder_classes in order, runs a full noise sweep for each,
    and concatenates the results. The decoder_name column distinguishes results
    from different decoders in the output DataFrame.
 
    Args:
        distances:           Code distances to evaluate.
        noise_levels:        Physical error rates to sweep over.
        decoder_classes:     List of decoder classes to compare.
        n_samples:           Monte Carlo shots per data point.
        rounds_per_distance: Syndrome extraction rounds. If None, uses rounds = distance.
        code_type:           Surface code variant to generate.
        noise_factory:       Callable mapping p -> NoiseModel.
 
    Returns:
        DataFrame with the same schema as sweep_noise_levels, with decoder_name
        column identifying which decoder produced each row.
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