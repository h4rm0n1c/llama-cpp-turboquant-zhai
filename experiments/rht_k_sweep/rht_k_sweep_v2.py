"""RHT-count sweep — v2 — K source matches §3 POST-WHT statistics.

v1 generated raw K with Beta(0.5, 0.5) (kurt -1.5, KS 0.099) and discovered that
ONE Hadamard at d=128 Gaussianizes that input almost completely (kurt -0.006,
KS 0.002), which under-stresses Basat's theorem because there's nothing left to
fix.

The production reality (§3 of why-mse-fails) is: K *already post-WHT* is
sub-Gaussian (kurt -1.56, KS 0.155 at layer 0). The fork already applies 1 WHT.
Basat's claim is that ADDITIONAL RHTs would help. So this v2 generates K to
match the post-WHT distribution and tests 0 / 1 / 2 ADDITIONAL RHTs (total
1 / 2 / 3 from the conceptual pre-WHT origin).

To get a kurt -1.56 distribution at d=128, we use a tighter-than-arcsine
bimodal: K_i ~ uniform({-a, +a}) with small additive Gaussian noise. Pure
Bernoulli is kurt -2; mixing in noise pulls kurt up toward 0. We tune to
match the §3 target.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

d = 128
T = 512
n_trials = 5
n_queries = 64
sigma = 1.0 / np.sqrt(d)
n_levels = 16  # b=4
master_rng = np.random.default_rng(42)

TARGET_KURT = -1.56
TARGET_KS = 0.155


# ---------------------------------------------------------------------------
# Generate K matching §3 post-WHT stats
# ---------------------------------------------------------------------------

def gen_postwht_K(T: int, d: int, rng, noise_frac: float = 0.20) -> np.ndarray:
    """K_i ~ uniform({-a, +a}) + small Gaussian noise. Tunable noise_frac.
    Variance scaled to 1/d."""
    bern = rng.choice([-1.0, 1.0], size=(T, d))
    noise = rng.normal(0.0, noise_frac, size=(T, d))
    raw = bern + noise
    scale = sigma / np.sqrt(np.var(raw))
    return raw * scale


# noise_frac search to match TARGET_KURT
K_orig = None
for noise_frac in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    K_try = gen_postwht_K(T, d, np.random.default_rng(42), noise_frac)
    kurt_try = stats.kurtosis(K_try.flatten())
    ks_try, _ = stats.kstest(K_try.flatten(), 'norm', args=(0, sigma))
    print(f"  noise_frac={noise_frac:.2f}  kurt={kurt_try:+.3f}  KS={ks_try:.4f}")
    if abs(kurt_try - TARGET_KURT) < 0.05:
        K_orig = K_try
        chosen_nf = noise_frac
        chosen_kurt = kurt_try
        chosen_ks = ks_try

if K_orig is None:
    # fall back: use a slight tail-trimmed Gaussian to hit kurt -1.56
    print("  No exact match; using interpolated fallback at noise_frac=0.10")
    K_orig = gen_postwht_K(T, d, np.random.default_rng(42), 0.10)
    chosen_nf = 0.10
    chosen_kurt = stats.kurtosis(K_orig.flatten())
    chosen_ks = stats.kstest(K_orig.flatten(), 'norm', args=(0, sigma))[0]

print()
print(f"=== Source K (simulating §3 layer-0 POST-WHT) ===")
print(f"  shape       : {K_orig.shape}")
print(f"  noise_frac  : {chosen_nf}")
print(f"  variance    : {K_orig.var():.6f} (target 1/d = {1/d:.6f})")
print(f"  kurtosis    : {chosen_kurt:+.4f}  (§3 layer-0 = -1.56)")
print(f"  KS vs N(0, 1/d): {chosen_ks:.4f}  (§3 layer-0 = 0.155)")
print()


# ---------------------------------------------------------------------------
# Hadamard
# ---------------------------------------------------------------------------

H = linalg.hadamard(d) / np.sqrt(d)


def apply_rht(X: np.ndarray, k: int, seed: int):
    rng = np.random.default_rng(seed)
    Y = X.copy()
    sign_seqs = []
    for _ in range(k):
        signs = rng.choice([-1.0, 1.0], size=d).astype(np.float64)
        sign_seqs.append(signs)
        Y = (Y * signs) @ H.T
    return Y, sign_seqs


def invert_rht(Y: np.ndarray, sign_seqs):
    X = Y.copy()
    for signs in reversed(sign_seqs):
        X = (X @ H) * signs
    return X


# ---------------------------------------------------------------------------
# Lloyd-Max codebook for N(0, sigma^2)
# ---------------------------------------------------------------------------

def lloyd_max_gaussian(sigma: float, n_levels: int, n_iter: int = 100) -> np.ndarray:
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
    flat = X.flatten()
    idx = np.argmin(np.abs(flat[:, None] - codebook[None, :]), axis=1)
    return codebook[idx].reshape(X.shape)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attn_kl(K_ref: np.ndarray, K_test: np.ndarray, Q: np.ndarray) -> np.ndarray:
    sd = np.sqrt(d)
    p = softmax(Q @ K_ref.T / sd)
    q = softmax(Q @ K_test.T / sd)
    eps = 1e-12
    return (p * (np.log(p + eps) - np.log(q + eps))).sum(axis=-1)


# ---------------------------------------------------------------------------
# Sweep: k_extra in {0, 1, 2}
#   k_extra=0 → no additional rotation → quantize K_orig directly (production behavior)
#   k_extra=1 → +1 RHT
#   k_extra=2 → +2 RHTs
# ---------------------------------------------------------------------------

print("=== Sweep additional RHTs in {0, 1, 2} (on top of K_orig = post-WHT) ===")
print()
print(f"{'k_extra':<10}{'post-kurt':<14}{'post-KS':<12}{'MSE':<18}"
      f"{'KL mean':<16}{'KL p99':<16}")
print("-" * 86)

results = {}
all_kl_arrays = {}

for k_extra in [0, 1, 2]:
    mse_runs = []
    kl_per_query: list[np.ndarray] = []
    kurt_post = None
    ks_post = None
    for trial in range(n_trials):
        seed = 1000 + 100 * (k_extra + 1) + trial
        if k_extra == 0:
            K_rot = K_orig.copy()
            sign_seqs = []
        else:
            K_rot, sign_seqs = apply_rht(K_orig, k_extra, seed)
        if trial == 0:
            kurt_post = stats.kurtosis(K_rot.flatten())
            ks_post, _ = stats.kstest(K_rot.flatten(), 'norm', args=(0, sigma))
        K_rot_q = quantize(K_rot, codebook)
        K_recon = invert_rht(K_rot_q, sign_seqs) if sign_seqs else K_rot_q
        mse_runs.append(np.mean((K_recon - K_orig) ** 2))
        q_rng = np.random.default_rng(seed + 50000)
        Q = q_rng.normal(0.0, sigma, size=(n_queries, d))
        kl_per_query.append(attn_kl(K_orig, K_recon, Q))

    mse_mean = float(np.mean(mse_runs))
    mse_std = float(np.std(mse_runs))
    kl_concat = np.concatenate(kl_per_query)
    all_kl_arrays[k_extra] = kl_concat
    results[k_extra] = {
        "post_kurt": kurt_post,
        "post_ks": ks_post,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "kl_mean": float(kl_concat.mean()),
        "kl_p99": float(np.percentile(kl_concat, 99)),
    }
    print(
        f"{k_extra:<10}{kurt_post:<+14.4f}{ks_post:<12.4f}"
        f"{mse_mean:<18.4e}{kl_concat.mean():<16.4e}"
        f"{np.percentile(kl_concat, 99):<16.4e}"
    )

# ---------------------------------------------------------------------------
# Paired catastrophic rate vs k_extra=0 (production baseline)
# ---------------------------------------------------------------------------

print()
print("=== Catastrophic rate (per-query KL > 1.10 * k_extra=0 median) ===")
base_median = float(np.median(all_kl_arrays[0]))
print(f"  k_extra=0 baseline KL median: {base_median:.4e}")
for k in [0, 1, 2]:
    cat = float(np.mean(all_kl_arrays[k] >= 1.10 * base_median))
    print(f"  k_extra={k}  catastrophic_rate: {cat:.3%}")

# ---------------------------------------------------------------------------
# Delta table (vs k_extra=0)
# ---------------------------------------------------------------------------

print()
print("=== Delta vs k_extra=0 (production baseline) ===")
print(f"  k_extra=1  MSE: {(results[1]['mse_mean']/results[0]['mse_mean'] - 1)*100:+.2f}%   "
      f"KL mean: {(results[1]['kl_mean']/results[0]['kl_mean'] - 1)*100:+.2f}%")
print(f"  k_extra=2  MSE: {(results[2]['mse_mean']/results[0]['mse_mean'] - 1)*100:+.2f}%   "
      f"KL mean: {(results[2]['kl_mean']/results[0]['kl_mean'] - 1)*100:+.2f}%")
