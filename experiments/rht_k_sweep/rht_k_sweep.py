"""RHT-count sweep on sub-Gaussian K cache.

Tests Basat et al. 2026 (arXiv:2605.06014v1) "Quantizing With Randomized Hadamard
Transforms" against the §3 observation from `why-mse-fails-for-kv-quantization.md`
that real K cache post-WHT is sub-Gaussian (kurt ~ -1.56 at layer 0).

Question: does applying k=2 or k=3 RHTs (which Basat proves recovers the
URR/Gaussian-marginal guarantee) improve Lloyd-Max-on-Gaussian quantization
quality, measured by (a) per-coord MSE in the original space and (b) attention-
softmax KL divergence using random queries.

Synthetic K, d=128 (Qwen3-0.6B head_dim), T=512 tokens, Beta(0.5, 0.5)-shaped
to match the layer-0 K kurt observed in §3. Lloyd-Max codebook fit to N(0, 1/d)
at b=4 (16 levels).
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

d = 128            # head dim (Qwen3-0.6B)
T = 512            # tokens (matches §3 prefill length)
n_trials = 5       # different RHT sign-flip seeds per k
n_queries = 64     # queries for attention-softmax KL proxy
sigma = 1.0 / np.sqrt(d)
n_levels = 16      # b=4
master_rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Generate sub-Gaussian K matching §3 layer-0 stats
# ---------------------------------------------------------------------------

def gen_subgaussian_K(T: int, d: int, rng) -> np.ndarray:
    """Beta(0.5, 0.5) shifted to [-1, 1], scaled to variance 1/d. Produces
    kurt ≈ -1.5 (arcsine), close to the §3 layer-0 K observation (-1.56)."""
    raw = rng.beta(0.5, 0.5, size=(T, d))
    shifted = 2.0 * raw - 1.0
    scale = sigma / np.sqrt(np.var(shifted))
    return shifted * scale


K_orig = gen_subgaussian_K(T, d, master_rng)
ks_orig, _ = stats.kstest(K_orig.flatten(), 'norm', args=(0, sigma))
kurt_orig = stats.kurtosis(K_orig.flatten())
print("=== Source K stats (sub-Gaussian, simulating §3 layer-0) ===")
print(f"  shape       : {K_orig.shape}")
print(f"  variance    : {K_orig.var():.6f} (target 1/d = {1/d:.6f})")
print(f"  kurtosis    : {kurt_orig:.4f} (§3 layer-0 = -1.56)")
print(f"  KS-stat vs N(0, 1/d): {ks_orig:.4f} (§3 layer-0 = 0.155)")
print()


# ---------------------------------------------------------------------------
# Hadamard + randomized sign-flip rotation
# ---------------------------------------------------------------------------

H = linalg.hadamard(d) / np.sqrt(d)  # orthonormal


def apply_rht(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply k RHTs in sequence (sign-flip then Hadamard). Returns rotated X
    and the sign-flip vectors used, for invertibility."""
    rng = np.random.default_rng(seed)
    Y = X.copy()
    sign_seqs: list[np.ndarray] = []
    for _ in range(k):
        signs = rng.choice([-1.0, 1.0], size=d).astype(np.float64)
        sign_seqs.append(signs)
        Y = (Y * signs) @ H.T
    return Y, sign_seqs


def invert_rht(Y: np.ndarray, sign_seqs: list[np.ndarray]) -> np.ndarray:
    """Inverse of apply_rht: H is its own inverse (symmetric orthonormal),
    sign flips are self-inverse."""
    X = Y.copy()
    for signs in reversed(sign_seqs):
        X = (X @ H) * signs
    return X


# ---------------------------------------------------------------------------
# Lloyd-Max codebook for N(0, sigma^2), b=4 (16 levels)
# ---------------------------------------------------------------------------

def lloyd_max_gaussian(sigma: float, n_levels: int, n_iter: int = 80) -> np.ndarray:
    """Optimal scalar quantizer for a Gaussian source. Iterative Lloyd updates.
    This is the standard 'turbo' codebook calibration target."""
    points = norm.ppf(
        np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels), 0.0, sigma
    )
    for _ in range(n_iter):
        boundaries = (points[:-1] + points[1:]) / 2
        edges = np.concatenate([[-np.inf], boundaries, [np.inf]])
        new_pts = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            pa = norm.pdf(a, 0.0, sigma) if np.isfinite(a) else 0.0
            pb = norm.pdf(b, 0.0, sigma) if np.isfinite(b) else 0.0
            phi_a = norm.cdf(a, 0.0, sigma)
            phi_b = norm.cdf(b, 0.0, sigma)
            mass = phi_b - phi_a
            if mass < 1e-15:
                new_pts.append((a + b) / 2 if np.isfinite(a) and np.isfinite(b) else points[i])
                continue
            # E[X | a < X < b] under N(0, sigma^2) = -sigma^2 (phi(b) - phi(a)) / mass
            mean = -sigma * sigma * (pb - pa) / mass
            new_pts.append(mean)
        new_pts_arr = np.array(new_pts)
        if np.max(np.abs(new_pts_arr - points)) < 1e-9:
            break
        points = new_pts_arr
    return points


codebook = lloyd_max_gaussian(sigma, n_levels)
print("=== Lloyd-Max codebook (b=4, fit to N(0, 1/d)) ===")
print(f"  {codebook.round(5)}")
print()


def quantize(X: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Nearest-neighbor scalar quantize to codebook, vectorized."""
    flat = X.flatten()
    # broadcast: (N, 1) - (1, n_levels)
    idx = np.argmin(np.abs(flat[:, None] - codebook[None, :]), axis=1)
    return codebook[idx].reshape(X.shape)


# ---------------------------------------------------------------------------
# Attention-softmax KL proxy
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attn_kl(K_ref: np.ndarray, K_test: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """KL( softmax(Q @ K_ref^T / sqrt(d))  ||  softmax(Q @ K_test^T / sqrt(d)) )
    per-query (1D array of length n_queries)."""
    sd = np.sqrt(d)
    p = softmax(Q @ K_ref.T / sd)        # (Nq, T)
    q = softmax(Q @ K_test.T / sd)
    eps = 1e-12
    kl = (p * (np.log(p + eps) - np.log(q + eps))).sum(axis=-1)
    return kl


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

print("=== Sweep k_rht in {1, 2, 3} ===")
print()

results = {}

for k_rht in [1, 2, 3]:
    mse_runs = []
    kl_per_query: list[np.ndarray] = []
    kurt_post = None
    ks_post = None

    for trial in range(n_trials):
        seed = 1000 + 100 * k_rht + trial
        # Apply k RHTs
        K_rot, sign_seqs = apply_rht(K_orig, k_rht, seed)
        # Distributional measurement (only first trial)
        if trial == 0:
            kurt_post = stats.kurtosis(K_rot.flatten())
            ks_post, _ = stats.kstest(K_rot.flatten(), 'norm', args=(0, sigma))
        # Quantize in rotated space with Gaussian-fit codebook
        K_rot_q = quantize(K_rot, codebook)
        # Invert RHTs back to original space
        K_recon = invert_rht(K_rot_q, sign_seqs)
        # Per-coord MSE in the ORIGINAL space (what attention actually reads)
        mse_runs.append(np.mean((K_recon - K_orig) ** 2))
        # Attention KL proxy
        q_rng = np.random.default_rng(seed + 50000)
        Q = q_rng.normal(0.0, sigma, size=(n_queries, d))
        kl_per_query.append(attn_kl(K_orig, K_recon, Q))

    mse_mean = float(np.mean(mse_runs))
    mse_std = float(np.std(mse_runs))
    kl_concat = np.concatenate(kl_per_query)
    kl_mean = float(kl_concat.mean())
    kl_p50 = float(np.percentile(kl_concat, 50))
    kl_p99 = float(np.percentile(kl_concat, 99))

    results[k_rht] = {
        "post_kurt": kurt_post,
        "post_ks": ks_post,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "kl_mean": kl_mean,
        "kl_p50": kl_p50,
        "kl_p99": kl_p99,
    }

    print(
        f"k={k_rht}  | post-RHT kurt={kurt_post:+.4f}  KS={ks_post:.4f}  "
        f"|  MSE={mse_mean:.4e} ± {mse_std:.1e}  "
        f"|  KL mean={kl_mean:.4e}  KL p99={kl_p99:.4e}"
    )

# ---------------------------------------------------------------------------
# Catastrophic-rate proxy: queries where KL >= 1.10 * baseline (k=1) KL
# ---------------------------------------------------------------------------

print()
print("=== Catastrophic rate (>=10% worse than k=1 baseline KL per-query) ===")
# rebuild k=1 baseline KL distribution at same seeds for paired comparison
base_kls: list[float] = []
for trial in range(n_trials):
    seed = 1000 + 100 * 1 + trial
    K_rot, sign_seqs = apply_rht(K_orig, 1, seed)
    K_rot_q = quantize(K_rot, codebook)
    K_recon = invert_rht(K_rot_q, sign_seqs)
    q_rng = np.random.default_rng(seed + 50000)
    Q = q_rng.normal(0.0, sigma, size=(n_queries, d))
    base_kls.extend(attn_kl(K_orig, K_recon, Q).tolist())
base_arr = np.asarray(base_kls)
base_median = float(np.median(base_arr))
print(f"  k=1 baseline KL median: {base_median:.4e}")

for k_rht in [1, 2, 3]:
    # rebuild per-query KL for k_rht (same seeds, paired)
    kls: list[float] = []
    for trial in range(n_trials):
        seed = 1000 + 100 * k_rht + trial
        K_rot, sign_seqs = apply_rht(K_orig, k_rht, seed)
        K_rot_q = quantize(K_rot, codebook)
        K_recon = invert_rht(K_rot_q, sign_seqs)
        q_rng = np.random.default_rng(seed + 50000)
        Q = q_rng.normal(0.0, sigma, size=(n_queries, d))
        kls.extend(attn_kl(K_orig, K_recon, Q).tolist())
    arr = np.asarray(kls)
    cat = float(np.mean(arr >= 1.10 * base_median))
    print(f"  k={k_rht}  catastrophic_rate (KL > 1.10 * baseline median): {cat:.3%}")

print()
print("=== Sub-Gaussian-CONTROL: same experiment on Gaussian K (sanity) ===")
# Pure Gaussian source — Basat's prediction should be CLEANLY CORRECT here:
# more RHTs preserve Gaussian → no change in fit → MSE should be flat.
K_gauss = master_rng.normal(0.0, sigma, size=(T, d))
print(f"  K_gauss: kurt={stats.kurtosis(K_gauss.flatten()):+.4f}")
for k_rht in [1, 2, 3]:
    mses: list[float] = []
    for trial in range(n_trials):
        seed = 7000 + 100 * k_rht + trial
        K_rot, sign_seqs = apply_rht(K_gauss, k_rht, seed)
        K_rot_q = quantize(K_rot, codebook)
        K_recon = invert_rht(K_rot_q, sign_seqs)
        mses.append(np.mean((K_recon - K_gauss) ** 2))
    print(f"  k={k_rht}  MSE={np.mean(mses):.4e} ± {np.std(mses):.1e}")
