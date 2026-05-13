"""Sensitivity check before public post: does the v3 result depend on the
specific Bernoulli+noise source shape, or does it hold for other sub-Gaussian
shapes at the same kurt?

If the result is brittle to source shape (e.g. holds only for bimodal-ish
sources, not uniform-ish), the public claim has to be qualified.

Test 4 sources at the same target kurt ≈ -1.56:
  (a) Bernoulli + Gaussian noise (the v3 source, peaks at ±a)
  (b) Uniform on [-a, +a] (flat, no peaks)
  (c) Mixture: 50% uniform + 50% Gaussian (intermediate shape, kurt tunable)
  (d) Truncated Gaussian (Gaussian clipped at ±a, kurt tunable)

Same d=128, b=4 Lloyd-Max-on-Gaussian, k_extra ∈ {0,1,2}.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats
from scipy.stats import norm, truncnorm

d = 128
T = 512
n_trials = 5
n_queries = 64
sigma = 1.0 / np.sqrt(d)
n_levels = 16
TARGET_KURT = -1.56


def normalize_to_sigma(raw):
    return raw * (sigma / np.sqrt(np.var(raw)))


def src_bern_noise(T, d, rng, nf):
    bern = rng.choice([-1.0, 1.0], size=(T, d))
    noise = rng.normal(0.0, nf, size=(T, d))
    return normalize_to_sigma(bern + noise)


def src_uniform(T, d, rng, half_width):
    raw = rng.uniform(-half_width, half_width, size=(T, d))
    return normalize_to_sigma(raw)


def src_unif_gauss_mix(T, d, rng, mix_frac, half_width):
    """mix_frac of uniform, (1-mix_frac) of Gaussian."""
    n_unif = int(T * d * mix_frac)
    n_gauss = T * d - n_unif
    unif = rng.uniform(-half_width, half_width, n_unif)
    gauss = rng.normal(0.0, half_width / 1.7, n_gauss)  # scaled to similar variance
    arr = np.concatenate([unif, gauss])
    rng.shuffle(arr)
    return normalize_to_sigma(arr.reshape(T, d))


def src_truncated_gauss(T, d, rng, trunc_sd):
    """Truncated Gaussian: clip at ±trunc_sd * sigma."""
    raw = rng.normal(0.0, 1.0, size=(T, d))
    raw = np.clip(raw, -trunc_sd, trunc_sd)
    return normalize_to_sigma(raw)


# Tune each source to kurt ≈ TARGET_KURT
def tune_param(gen_fn, T, d, target_kurt, param_range, n_search=20):
    best = None
    best_err = 1e9
    for p in np.linspace(*param_range, n_search):
        rng = np.random.default_rng(42)
        X = gen_fn(T, d, rng, p)
        k = stats.kurtosis(X.flatten())
        err = abs(k - target_kurt)
        if err < best_err:
            best_err = err
            best = (p, k, X)
    return best


print(f"Tuning sources to target kurt = {TARGET_KURT}...\n")
b_p, b_k, K_bern = tune_param(src_bern_noise, T, d, TARGET_KURT, (0.1, 1.5))
u_p, u_k, K_unif = tune_param(src_uniform, T, d, TARGET_KURT, (0.5, 5.0))
m_p, m_k, K_mix = tune_param(
    lambda T, d, rng, p: src_unif_gauss_mix(T, d, rng, p, 1.0),
    T, d, TARGET_KURT, (0.0, 1.0),
)
t_p, t_k, K_trunc = tune_param(src_truncated_gauss, T, d, TARGET_KURT, (0.5, 4.0))

print(f"  bernoulli+noise   noise={b_p:.3f}  kurt={b_k:+.3f}")
print(f"  uniform           half_w={u_p:.3f}  kurt={u_k:+.3f}")
print(f"  unif+gauss mix    mix_frac={m_p:.3f}  kurt={m_k:+.3f}")
print(f"  truncated gauss   trunc_sd={t_p:.3f}  kurt={t_k:+.3f}")
print()


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


def lloyd_max_gaussian(sigma, n_levels, n_iter=100):
    points = norm.ppf(np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels), 0.0, sigma)
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
            new_pts.append(-sigma * sigma * (pb - pa) / mass)
        new_pts_arr = np.array(new_pts)
        if np.max(np.abs(new_pts_arr - points)) < 1e-9:
            break
        points = new_pts_arr
    return points


codebook = lloyd_max_gaussian(sigma, n_levels)


def quantize(X, codebook):
    flat = X.flatten()
    idx = np.argmin(np.abs(flat[:, None] - codebook[None, :]), axis=1)
    return codebook[idx].reshape(X.shape)


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


def run_one_source(name, K_orig):
    print(f"=== source: {name}  (kurt={stats.kurtosis(K_orig.flatten()):+.3f}) ===")
    results = []
    base_kl_arr = None
    for k_extra in [0, 1, 2]:
        mse_runs = []
        kls_runs = []
        for trial in range(n_trials):
            seed = 1000 + 100 * (k_extra + 1) + trial
            if k_extra == 0:
                K_rot, sign_seqs = K_orig.copy(), []
            else:
                K_rot, sign_seqs = apply_rht(K_orig, k_extra, seed)
            K_rot_q = quantize(K_rot, codebook)
            K_recon = invert_rht(K_rot_q, sign_seqs) if sign_seqs else K_rot_q
            mse_runs.append(np.mean((K_recon - K_orig) ** 2))
            q_rng = np.random.default_rng(seed + 50000)
            Q = q_rng.normal(0.0, sigma, size=(n_queries, d))
            kls_runs.append(attn_kl(K_orig, K_recon, Q))
        kl_all = np.concatenate(kls_runs)
        if k_extra == 0:
            base_kl_arr = kl_all
        cat = float(np.mean(kl_all >= 1.10 * np.median(base_kl_arr)))
        results.append((k_extra, float(np.mean(mse_runs)), float(kl_all.mean()), cat))
    base_mse = results[0][1]
    base_kl = results[0][2]
    for k_extra, mse, kl_mean, cat in results:
        d_mse = (mse / base_mse - 1) * 100
        d_kl = (kl_mean / base_kl - 1) * 100
        print(
            f"  k_extra={k_extra}  MSE={mse:.3e} ({d_mse:+5.1f}%)  "
            f"KL={kl_mean:.3e} ({d_kl:+5.1f}%)  cat={cat:.1%}"
        )
    print()


run_one_source("bernoulli+noise (v3 source)", K_bern)
run_one_source("uniform (flat density)", K_unif)
run_one_source("uniform/gaussian mix", K_mix)
run_one_source("truncated gaussian", K_trunc)
