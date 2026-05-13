"""RHT-count sweep using the ACTUAL production turbo4 codebook.

Source: ggml/src/ggml-cuda/turbo-quant.cuh TURBO_CENTROIDS_4BIT (16 levels).
Block normalization: per-128-element L2 (QK_TURBO4 = 128 = head_dim per
ggml-common.h), so each row of K (one head's worth of values for one token)
is one normalization block.

This is the codebook the fork actually ships. v3 used Lloyd-Max-on-Gaussian
(±2.7σ extremes). Production is ±2σ-ish, sub-Gaussian-fitted. Result should
hold (same mechanism: production codebook lucky-aligns with sub-Gaussian K,
+RHT breaks alignment), but the magnitude could differ.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats

d = 128                    # = QK_TURBO4 = head_dim, also the L2-norm block size
T = 512
n_trials = 5
n_queries = 64
sigma = 1.0 / np.sqrt(d)
TARGET_KURT = -1.56


# Production turbo4 codebook (16 levels, 4-bit)
# From ggml/src/ggml-cuda/turbo-quant.cuh:
PROD_CENTROIDS_4BIT = np.array([
    -0.173926, -0.117195, -0.089527, -0.068756,
    -0.051262, -0.035597, -0.020989, -0.006938,
     0.006938,  0.020989,  0.035597,  0.051262,
     0.068756,  0.089527,  0.117195,  0.173926
])


# ---------------------------------------------------------------------------
# Source K matching §3 layer-0 stats
# ---------------------------------------------------------------------------

def gen_postwht_K(T, d, rng, noise_frac):
    bern = rng.choice([-1.0, 1.0], size=(T, d))
    noise = rng.normal(0.0, noise_frac, size=(T, d))
    raw = bern + noise
    return raw * (sigma / np.sqrt(np.var(raw)))


H = linalg.hadamard(d) / np.sqrt(d)


def apply_rht(X, k, seed):
    rng = np.random.default_rng(seed)
    Y = X.copy()
    sign_seqs = []
    for _ in range(k):
        signs = rng.choice([-1.0, 1.0], size=d).astype(np.float64)
        sign_seqs.append(signs)
        Y = (Y * signs) @ H.T
    return Y, sign_seqs


def invert_rht(Y, sign_seqs):
    X = Y.copy()
    for signs in reversed(sign_seqs):
        X = (X @ H) * signs
    return X


# ---------------------------------------------------------------------------
# Production-style block quantization
#
#   Per row (one head_dim block of size 128):
#     norm = ||K_row||_2  (then divide by sqrt(d) to get rms-scaled centroids
#                          relative to row magnitude — see kernel: kv = c * norm)
#   Then nearest-centroid lookup.
#   Dequantize: K_row[i] ≈ centroid[idx[i]] * norm
# ---------------------------------------------------------------------------

def quantize_prod(X, codebook):
    """X: (T, d). Per-row L2 normalize, nearest-centroid, dequantize."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)  # (T, 1)
    norms = np.maximum(norms, 1e-12)
    # The kernel does kv = centroid * norm. So centroids in the table are
    # scaled so that centroid * norm reconstructs the value. Therefore the
    # nearest centroid match is performed against (X / norm).
    X_n = X / norms  # (T, d), normalized rows
    flat = X_n.flatten()
    idx = np.argmin(np.abs(flat[:, None] - codebook[None, :]), axis=1)
    out_n = codebook[idx].reshape(X_n.shape)
    return out_n * norms  # dequantize


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attn_kl(K_ref, K_test, Q):
    sd = np.sqrt(d)
    p = softmax(Q @ K_ref.T / sd)
    q = softmax(Q @ K_test.T / sd)
    eps = 1e-12
    return (p * (np.log(p + eps) - np.log(q + eps))).sum(axis=-1)


# ---------------------------------------------------------------------------
# Find noise_frac that matches §3 layer-0 stats with production codebook
# ---------------------------------------------------------------------------

print(f"=== Production turbo4 codebook ===")
print(f"  centroids (16 levels): extremes ±{PROD_CENTROIDS_4BIT[-1]:.4f}")
print(f"  L2-normalized extreme: ±{PROD_CENTROIDS_4BIT[-1]:.4f} of row norm")
print(f"  vs Lloyd-Max-Gaussian extremes: ±0.240 (my v3 codebook)")
print()

# Sweep noise_frac to find §3 layer-0 match
print("Tuning noise_frac to match §3 layer-0 stats...")
print(f"  target: kurt = -1.56, KS = 0.155")
print()
chosen = None
for nf in [0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40]:
    K_try = gen_postwht_K(T, d, np.random.default_rng(42), nf)
    kurt = stats.kurtosis(K_try.flatten())
    ks, _ = stats.kstest(K_try.flatten(), 'norm', args=(0, sigma))
    print(f"  noise_frac={nf:.2f}  kurt={kurt:+.3f}  KS={ks:.4f}")
    if abs(kurt - TARGET_KURT) < 0.05:
        chosen = (nf, kurt, ks, K_try)
nf, kurt, ks, K_orig = chosen if chosen else (0.35, *stats.kurtosis(gen_postwht_K(T, d, np.random.default_rng(42), 0.35).flatten()), 0, gen_postwht_K(T, d, np.random.default_rng(42), 0.35))
print()
print(f"  selected: noise_frac={nf:.3f}, kurt={kurt:+.3f}, KS={ks:.4f}")
print()


# ---------------------------------------------------------------------------
# Sweep k_extra with production codebook
# ---------------------------------------------------------------------------

print("=== Sweep k_extra on production codebook ===")
print(f"{'k_extra':<10}{'post-kurt':<14}{'post-KS':<12}"
      f"{'MSE':<18}{'KL mean':<16}{'KL p99':<16}{'Cat rate':<12}")
print("-" * 100)

results = {}
all_kl_arrays = {}

for k_extra in [0, 1, 2]:
    mse_runs = []
    kl_per_query: list[np.ndarray] = []
    kurt_post = None
    ks_post = None
    for trial in range(n_trials):
        seed = 2000 + 100 * (k_extra + 1) + trial
        if k_extra == 0:
            K_rot, sign_seqs = K_orig.copy(), []
        else:
            K_rot, sign_seqs = apply_rht(K_orig, k_extra, seed)
        if trial == 0:
            kurt_post = stats.kurtosis(K_rot.flatten())
            ks_post, _ = stats.kstest(K_rot.flatten(), 'norm', args=(0, sigma))
        K_rot_q = quantize_prod(K_rot, PROD_CENTROIDS_4BIT)
        K_recon = invert_rht(K_rot_q, sign_seqs) if sign_seqs else K_rot_q
        mse_runs.append(np.mean((K_recon - K_orig) ** 2))
        q_rng = np.random.default_rng(seed + 50000)
        Q = q_rng.normal(0.0, sigma, size=(n_queries, d))
        kl_per_query.append(attn_kl(K_orig, K_recon, Q))

    mse_mean = float(np.mean(mse_runs))
    kl_concat = np.concatenate(kl_per_query)
    all_kl_arrays[k_extra] = kl_concat
    results[k_extra] = {
        "post_kurt": kurt_post,
        "post_ks": ks_post,
        "mse_mean": mse_mean,
        "kl_mean": float(kl_concat.mean()),
        "kl_p99": float(np.percentile(kl_concat, 99)),
    }

base_median = float(np.median(all_kl_arrays[0]))
for k_extra in [0, 1, 2]:
    r = results[k_extra]
    cat = float(np.mean(all_kl_arrays[k_extra] >= 1.10 * base_median))
    print(
        f"{k_extra:<10}{r['post_kurt']:<+14.4f}{r['post_ks']:<12.4f}"
        f"{r['mse_mean']:<18.4e}{r['kl_mean']:<16.4e}"
        f"{r['kl_p99']:<16.4e}{cat:<12.1%}"
    )

print()
print("=== Delta vs k_extra=0 (production baseline) ===")
print(f"  k_extra=1  MSE: {(results[1]['mse_mean']/results[0]['mse_mean'] - 1)*100:+.2f}%   "
      f"KL mean: {(results[1]['kl_mean']/results[0]['kl_mean'] - 1)*100:+.2f}%   "
      f"KL p99: {(results[1]['kl_p99']/results[0]['kl_p99'] - 1)*100:+.2f}%")
print(f"  k_extra=2  MSE: {(results[2]['mse_mean']/results[0]['mse_mean'] - 1)*100:+.2f}%   "
      f"KL mean: {(results[2]['kl_mean']/results[0]['kl_mean'] - 1)*100:+.2f}%   "
      f"KL p99: {(results[2]['kl_p99']/results[0]['kl_p99'] - 1)*100:+.2f}%")

# ---------------------------------------------------------------------------
# Side-by-side: production vs Lloyd-Max-on-Gaussian codebook
# ---------------------------------------------------------------------------

print()
print("=== Side-by-side: PROD codebook vs Lloyd-Max-Gaussian (v3) codebook ===")

from scipy.stats import norm
def lloyd_max_gaussian(sig, n_levels=16, n_iter=100):
    pts = norm.ppf(np.linspace(0.5/n_levels, 1-0.5/n_levels, n_levels), 0, sig)
    for _ in range(n_iter):
        bnd = (pts[:-1] + pts[1:]) / 2
        edges = np.concatenate([[-np.inf], bnd, [np.inf]])
        new = []
        for i in range(n_levels):
            a, b = edges[i], edges[i+1]
            pa = norm.pdf(a, 0, sig) if np.isfinite(a) else 0
            pb = norm.pdf(b, 0, sig) if np.isfinite(b) else 0
            phi_a, phi_b = norm.cdf(a, 0, sig), norm.cdf(b, 0, sig)
            mass = phi_b - phi_a
            new.append(-sig*sig*(pb-pa)/mass if mass > 1e-15 else pts[i])
        new = np.array(new)
        if np.max(np.abs(new - pts)) < 1e-9: break
        pts = new
    return pts

LM_CENTROIDS = lloyd_max_gaussian(sigma)


def quantize_per_coord(X, codebook):
    """Per-coord global codebook (the v3 method, no per-block norm)."""
    flat = X.flatten()
    idx = np.argmin(np.abs(flat[:, None] - codebook[None, :]), axis=1)
    return codebook[idx].reshape(X.shape)


print(f"{'codebook':<20}{'quant scheme':<20}{'k_extra=0 MSE':<16}"
      f"{'k_extra=1 MSE':<16}{'Δ MSE':<12}{'Δ KL':<12}{'cat 0→1':<14}")
print("-" * 110)

for cb_name, codebook, scheme_name, quant_fn in [
    ("PROD turbo4", PROD_CENTROIDS_4BIT, "per-row L2 norm", quantize_prod),
    ("Lloyd-Max Gauss", LM_CENTROIDS, "per-coord global", quantize_per_coord),
]:
    base_mse, base_kl, k1_mse, k1_kl, base_kls, k1_kls = (None,) * 6
    for k_extra in [0, 1]:
        mse_runs, kls_runs = [], []
        for trial in range(n_trials):
            seed = 3000 + 100*(k_extra+1) + trial
            if k_extra == 0:
                K_rot, sign_seqs = K_orig.copy(), []
            else:
                K_rot, sign_seqs = apply_rht(K_orig, k_extra, seed)
            K_rot_q = quant_fn(K_rot, codebook)
            K_recon = invert_rht(K_rot_q, sign_seqs) if sign_seqs else K_rot_q
            mse_runs.append(np.mean((K_recon - K_orig)**2))
            q_rng = np.random.default_rng(seed + 50000)
            Q = q_rng.normal(0.0, sigma, size=(n_queries, d))
            kls_runs.append(attn_kl(K_orig, K_recon, Q))
        mse = float(np.mean(mse_runs))
        kls = np.concatenate(kls_runs)
        if k_extra == 0:
            base_mse, base_kl, base_kls = mse, kls.mean(), kls
        else:
            k1_mse, k1_kl, k1_kls = mse, kls.mean(), kls
    base_median = float(np.median(base_kls))
    cat0 = float(np.mean(base_kls >= 1.10 * base_median))
    cat1 = float(np.mean(k1_kls >= 1.10 * base_median))
    d_mse = (k1_mse/base_mse - 1) * 100
    d_kl = (k1_kl/base_kl - 1) * 100
    print(
        f"{cb_name:<20}{scheme_name:<20}{base_mse:<16.3e}"
        f"{k1_mse:<16.3e}{d_mse:<+12.1f}{d_kl:<+12.1f}{cat0:.1%} → {cat1:.1%}"
    )
