"""LMGauss vs production codebook on REAL Qwen3-0.6B post-WHT K.

Tests Theorem 7 (Basat 2026) prediction on the actual case: 3-RHT + a
Lloyd-Max-Gaussian codebook should match Gaussian-input quantization error
(up to a vanishing additive term). The paper does not pin a specific
codebook; we construct LM-Gauss for N(0, 1/d) so it matches the per-row
L2-normalized post-WHT K scale.

Cross-product to find the actual best cell:

                    turbo4 (sub-Gaussian fit, ±0.174)   LM-Gauss (Gaussian fit, ±0.241)
  1-RHT (prod)      [production stack]                  ?
  2-RHT             ?                                   ?
  3-RHT             ?                                   [paper's theoretical optimum]

Data: /Users/tom/dev/mse_scripts/eden-investigation/kv_dump/
Layers 0, 4, 8, 12, 16, 20, 23 from Qwen3-0.6B, 512 tokens, 8 heads, d=128.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.stats import norm

D = 128
KV_DIR = Path("/Users/tom/dev/mse_scripts/eden-investigation/kv_dump")
LAYERS = [0, 4, 8, 12, 16, 20, 23]

# Production turbo4 codebook from ggml/src/ggml-cuda/turbo-quant.cuh.
# Header comment claims Lloyd-Max for N(0, 1/128), but the actual centroid
# spacing is tighter (outermost 0.174 vs 0.241 for true Lloyd-Max-Gaussian),
# so it is empirically sub-Gaussian fitted to the post-WHT K distribution.
PROD_CENTROIDS_4BIT = np.array([
    -0.173926, -0.117195, -0.089527, -0.068756,
    -0.051262, -0.035597, -0.020989, -0.006938,
     0.006938,  0.020989,  0.035597,  0.051262,
     0.068756,  0.089527,  0.117195,  0.173926,
])


def lloyd_max_gaussian(sigma: float, n_levels: int = 16, n_iter: int = 500) -> np.ndarray:
    """Lloyd-Max-optimal scalar quantizer for N(0, sigma^2)."""
    pts = norm.ppf(np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels), 0, sigma)
    for _ in range(n_iter):
        bnd = (pts[:-1] + pts[1:]) / 2
        edges = np.concatenate([[-np.inf], bnd, [np.inf]])
        new = np.empty_like(pts)
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            pa = norm.pdf(a, 0, sigma) if np.isfinite(a) else 0.0
            pb = norm.pdf(b, 0, sigma) if np.isfinite(b) else 0.0
            mass = norm.cdf(b, 0, sigma) - norm.cdf(a, 0, sigma)
            new[i] = -sigma * sigma * (pb - pa) / mass if mass > 1e-15 else pts[i]
        if np.max(np.abs(new - pts)) < 1e-12:
            pts = new
            break
        pts = new
    return pts


SIGMA = 1.0 / np.sqrt(D)
LMG_CENTROIDS = lloyd_max_gaussian(SIGMA)

# Walsh-Hadamard basis (orthonormal)
H = linalg.hadamard(D) / np.sqrt(D)


def apply_rht(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Apply k random Hadamard transforms (random signs + Hadamard) to X (..., d)."""
    rng = np.random.default_rng(seed)
    Y = X.copy()
    for _ in range(k):
        signs = rng.choice([-1.0, 1.0], size=D).astype(np.float64)
        Y = (Y * signs) @ H.T
    return Y


def quantize_per_row(X: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Production scheme: per-row L2 normalize, nearest-centroid lookup, dequantize."""
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Xn = X / norms
    flat = Xn.reshape(-1)
    idx = np.argmin(np.abs(flat[:, None] - codebook[None, :]), axis=1)
    return codebook[idx].reshape(Xn.shape) * norms


def load_layer_k(layer: int) -> np.ndarray:
    """Load real K for layer, flatten (1, H, T, D) -> (H*T, D)."""
    k = np.load(KV_DIR / f"layer{layer:02d}_k.npy").astype(np.float64)
    return k.reshape(-1, D)


# ---------------------------------------------------------------------------
# Print codebook overview
# ---------------------------------------------------------------------------

print("=== LMGauss vs Production codebook on REAL Qwen3-0.6B K ===")
print(f"  d={D}, layers={LAYERS}, ~{8 * 512} rows per layer")
print(f"  PROD turbo4 outermost:           ±{PROD_CENTROIDS_4BIT[-1]:.4f}")
print(f"  LM-Gauss(sigma=1/sqrt(d)) outer: ±{LMG_CENTROIDS[-1]:.4f}  (= {LMG_CENTROIDS[-1]/SIGMA:.3f} sigma)")
print()

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

results: dict[int, dict[tuple[int, str], float]] = {}
post_wht_stats: dict[int, dict[str, float]] = {}

for layer in LAYERS:
    K_raw = load_layer_k(layer)
    # Production: one fixed-seed WHT-with-signs. This is the "1-RHT" baseline.
    K_post_wht = apply_rht(K_raw, 1, seed=layer * 1000)
    post_wht_stats[layer] = {
        "std": float(K_post_wht.std()),
        "abs_max": float(np.abs(K_post_wht).max()),
        "frac_outside_turbo4": float(np.mean(np.abs(K_post_wht) > PROD_CENTROIDS_4BIT[-1] * np.linalg.norm(K_post_wht, axis=-1, keepdims=True).mean())),
    }

    layer_res = {}
    for k_extra in (0, 1, 2):
        if k_extra == 0:
            K_rot = K_post_wht
        else:
            K_rot = apply_rht(K_post_wht, k_extra, seed=42 + layer * 10 + k_extra)
        for cb_name, cb in (("turbo4", PROD_CENTROIDS_4BIT), ("LMGauss", LMG_CENTROIDS)):
            K_q = quantize_per_row(K_rot, cb)
            layer_res[(k_extra, cb_name)] = float(np.mean((K_q - K_rot) ** 2))
    results[layer] = layer_res

# ---------------------------------------------------------------------------
# Per-layer table
# ---------------------------------------------------------------------------

print(f"{'layer':<8}{'RHTs':<6}{'turbo4 MSE':<18}{'LMGauss MSE':<18}"
      f"{'turbo4 % vs 1-RHT':<22}{'LMGauss % vs 1-RHT':<22}")
print("-" * 110)
for layer in LAYERS:
    base_t = results[layer][(0, "turbo4")]
    base_l = results[layer][(0, "LMGauss")]
    for k_extra in (0, 1, 2):
        t = results[layer][(k_extra, "turbo4")]
        l = results[layer][(k_extra, "LMGauss")]
        dt = (t / base_t - 1) * 100 if k_extra > 0 else 0.0
        dl = (l / base_l - 1) * 100 if k_extra > 0 else 0.0
        rht_label = f"{k_extra + 1}-RHT"
        print(f"{layer:<8}{rht_label:<6}{t:<18.6e}{l:<18.6e}{dt:<+22.2f}{dl:<+22.2f}")
    print()

# ---------------------------------------------------------------------------
# Aggregate ranking
# ---------------------------------------------------------------------------

print("=== Mean MSE across layers (lowest is best) ===")
cells = []
for k_extra in (0, 1, 2):
    for cb_name in ("turbo4", "LMGauss"):
        vals = [results[l][(k_extra, cb_name)] for l in LAYERS]
        mean_mse = float(np.mean(vals))
        cells.append((k_extra + 1, cb_name, mean_mse))

cells.sort(key=lambda x: x[2])
prod_baseline = next(m for n, c, m in cells if n == 1 and c == "turbo4")

print(f"{'rank':<6}{'cell':<25}{'mean MSE':<18}{'vs prod (1-RHT × turbo4)':<26}")
print("-" * 80)
for i, (n_rht, cb, mse) in enumerate(cells):
    delta = (mse / prod_baseline - 1) * 100
    label = f"{n_rht}-RHT × {cb}"
    print(f"{i + 1:<6}{label:<25}{mse:<18.6e}{delta:<+26.2f}")

# ---------------------------------------------------------------------------
# Headline interpretation
# ---------------------------------------------------------------------------

best = cells[0]
prod = (1, "turbo4", prod_baseline)
gain = (prod_baseline / best[2] - 1) * 100
print()
print(f"Best: {best[0]}-RHT × {best[1]}  MSE={best[2]:.6e}")
print(f"Prod: 1-RHT × turbo4         MSE={prod_baseline:.6e}")
if best[0] == 1 and best[1] == "turbo4":
    print("=> Production stack is optimal. Paper's recommendation loses on real K.")
elif best[0] == 3 and best[1] == "LMGauss":
    print(f"=> Paper's recommended stack (3-RHT + Gaussian-fit codebook) wins by {gain:.1f}% over production.")
else:
    print(f"=> Best cell is {best[0]}-RHT × {best[1]}, gain {gain:.1f}% over production.")
