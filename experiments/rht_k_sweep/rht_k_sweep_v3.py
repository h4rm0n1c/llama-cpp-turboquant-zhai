"""RHT-count sweep — v3 — full kurt sweep across the realistic sub-Gaussian range.

v2 showed a 22-23% MSE + KL degradation from adding 1 RHT on top of K with
kurt -1.96. The §3 layer-0 measurement is kurt -1.56 — milder. v3 sweeps
kurt from -2.0 (pure Bernoulli) to -0.0 (Gaussian) to characterize where
the degradation lives and whether it persists at the realistic -1.56 point.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats
from scipy.stats import norm

d = 128
T = 512
n_trials = 5
n_queries = 64
sigma = 1.0 / np.sqrt(d)
n_levels = 16


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


codebook = lloyd_max_gaussian(sigma, n_levels)
print("=== RHT-count sweep across kurt range (sub-Gaussian → Gaussian) ===\n")
print(f"{'noise_frac':<12}{'kurt':<10}{'KS':<10}",
      f"{'MSE k=0':<14}{'MSE k=1':<14}{'MSE k=2':<14}",
      f"{'Δk=1 %':<10}{'Δk=2 %':<10}",
      f"{'KL k=0':<14}{'KL k=1':<14}{'KL k=2':<14}",
      f"{'Cat k=0':<10}{'Cat k=1':<10}{'Cat k=2':<10}")
print("-" * 188)

noise_fracs = [0.05, 0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.0]
for nf in noise_fracs:
    K_orig = gen_postwht_K(T, d, np.random.default_rng(42), nf)
    kurt = stats.kurtosis(K_orig.flatten())
    ks, _ = stats.kstest(K_orig.flatten(), 'norm', args=(0, sigma))

    row = []
    for k_extra in [0, 1, 2]:
        mse_runs = []
        kls_runs = []
        for trial in range(n_trials):
            seed = 1000 + 100 * (k_extra + 1) + trial + int(nf * 1000)
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
        mse = float(np.mean(mse_runs))
        kl_all = np.concatenate(kls_runs)
        row.append((mse, kl_all))

    mse0, kl0 = row[0]
    mse1, kl1 = row[1]
    mse2, kl2 = row[2]
    base_median = float(np.median(kl0))
    cat0 = float(np.mean(kl0 >= 1.10 * base_median))
    cat1 = float(np.mean(kl1 >= 1.10 * base_median))
    cat2 = float(np.mean(kl2 >= 1.10 * base_median))

    print(
        f"{nf:<12.2f}{kurt:<+10.3f}{ks:<10.4f}"
        f"{mse0:<14.3e}{mse1:<14.3e}{mse2:<14.3e}"
        f"{(mse1/mse0-1)*100:<+10.1f}{(mse2/mse0-1)*100:<+10.1f}"
        f"{kl0.mean():<14.3e}{kl1.mean():<14.3e}{kl2.mean():<14.3e}"
        f"{cat0:<10.1%}{cat1:<10.1%}{cat2:<10.1%}"
    )
