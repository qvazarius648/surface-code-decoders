"""
Smoke test: verify the full pipeline works end-to-end.
"""
import sys
sys.path.insert(0, "/home/claude")

from qec_surface.circuits import build_surface_code, NoiseModel
from qec_surface.decoders import MWPMDecoder
from qec_surface.benchmark import estimate_logical_error_rate, sweep_noise_levels

# 1. Build a single circuit
noise = NoiseModel.uniform(0.01)
sc = build_surface_code(distance=3, rounds=3, noise=noise)
print(f"Circuit: {sc}")
print(f"  Data qubits: {sc.n_data_qubits}")
print(f"  Ancilla qubits: {sc.n_ancilla_qubits}")

# 2. Run single experiment
result = estimate_logical_error_rate(sc, MWPMDecoder, n_samples=1000)
print(f"\nSingle experiment:")
print(f"  LER = {result.logical_error_rate:.4f} ± {result.error_bar:.4f}")
print(f"  95% CI: {result.wilson_interval}")

# 3. Small sweep
print("\nSmall sweep (should take ~10s):")
df = sweep_noise_levels(
    distances=[3, 5],
    noise_levels=[0.005, 0.01, 0.02],
    decoder_cls=MWPMDecoder,
    n_samples=2000,
)
print(df[["distance", "noise_level", "logical_error_rate", "error_bar"]])
print("\nAll tests passed.")
