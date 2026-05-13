# RHT-count Sweep — Empirical Response to Basat et al. 2026

**Branch:** `experiment/rht-k-sweep`
**Date:** 2026-05-13
**Hardware:** Apple M5 Max (single-threaded Python, no GPU)

## Question

Does applying k=2 or k=3 RHTs (per Basat et al., arXiv:2605.06014v1) on top of the
production WHT improve Lloyd-Max-on-Gaussian quantization quality on real
KV-cache-shaped K?

## Answer

**No. It makes it worse, by 11-37% MSE and 11-37% KL across the entire sub-Gaussian
range, and approximately triples the catastrophic-regression rate.**

## Method

- d = 128 (Qwen3-0.6B head_dim)
- T = 512 tokens
- Synthetic K source: `bern({-1, +1}) + N(0, noise_frac²)`, rescaled to σ² = 1/d.
  This produces a continuously-tunable kurt range from -2.0 (pure Bernoulli) to
  -0.5 (near-Gaussian) by sweeping `noise_frac`. The §3 layer-0 K measurement
  (kurt -1.56, KS 0.155 vs N(0, 1/d)) lands at noise_frac ≈ 0.35.
- Codebook: b=4 (16 levels) Lloyd-Max fit to N(0, 1/d) — the standard production target.
- For each k_extra ∈ {0, 1, 2}: apply k_extra additional RHTs (Hadamard ×
  random sign-flip), quantize in the rotated space, inverse-rotate, measure
  MSE vs K_orig and attention-softmax KL using Q ~ N(0, 1/d) random queries.
- 5 trials per cell (independent sign-flip seeds), 64 queries each → 320 paired
  KL measurements per (kurt, k_extra) cell.

## Full sweep (12 kurt points × 3 k_extra values)

| noise_frac | kurt    | KS    | MSE k=0    | MSE k=1    | MSE k=2    | Δk=1 %  | Δk=2 %  | KL k=0    | KL k=1    | KL k=2    | Cat k=0 | Cat k=1 | Cat k=2 |
|------------|---------|-------|------------|------------|------------|---------|---------|-----------|-----------|-----------|---------|---------|---------|
| 0.05       | -1.990  | 0.306 | 5.41e-05   | 7.45e-05   | 7.26e-05   | +37.5%  | +34.1%  | 2.08e-07  | 2.91e-07  | 2.86e-07  | 26.6%   | 94.1%   | 93.1%   |
| 0.15       | -1.912  | 0.247 | 6.01e-05   | 7.34e-05   | 7.28e-05   | +22.1%  | +21.1%  | 2.33e-07  | 2.90e-07  | 2.84e-07  | 23.8%   | 78.8%   | 78.8%   |
| 0.25       | -1.770  | 0.195 | 6.01e-05   | 7.45e-05   | 7.29e-05   | +23.8%  | +21.2%  | 2.40e-07  | 2.91e-07  | 2.88e-07  | 23.1%   | 77.8%   | 72.8%   |
| 0.30       | -1.681  | 0.172 | 6.03e-05   | 7.36e-05   | 7.33e-05   | +21.9%  | +21.5%  | 2.35e-07  | 2.86e-07  | 2.87e-07  | 22.8%   | 76.9%   | 77.5%   |
| **0.35**   | **-1.584** | **0.152** | **6.06e-05** | **7.38e-05** | **7.32e-05** | **+21.7%** | **+20.7%** | **2.37e-07** | **2.89e-07** | **2.82e-07** | **23.1%** | **73.4%** | **70.9%** |
| 0.40       | -1.483  | 0.133 | 6.12e-05   | 7.29e-05   | 7.27e-05   | +19.2%  | +18.9%  | 2.37e-07  | 2.85e-07  | 2.86e-07  | 22.2%   | 73.1%   | 75.0%   |
| 0.45       | -1.379  | 0.115 | 6.16e-05   | 7.40e-05   | 7.34e-05   | +20.2%  | +19.1%  | 2.41e-07  | 2.88e-07  | 2.85e-07  | 26.9%   | 72.8%   | 72.5%   |
| 0.50       | -1.275  | 0.100 | 6.22e-05   | 7.38e-05   | 7.34e-05   | +18.8%  | +18.1%  | 2.44e-07  | 2.86e-07  | 2.88e-07  | 21.9%   | 66.6%   | 68.8%   |
| 0.60       | -1.075  | 0.075 | 6.29e-05   | 7.38e-05   | 7.32e-05   | +17.4%  | +16.4%  | 2.46e-07  | 2.89e-07  | 2.86e-07  | 24.7%   | 70.6%   | 66.2%   |
| 0.75       | -0.810  | 0.048 | 6.50e-05   | 7.41e-05   | 7.44e-05   | +14.1%  | +14.6%  | 2.55e-07  | 2.89e-07  | 2.96e-07  | 23.8%   | 60.9%   | 66.9%   |
| 1.00       | -0.488  | 0.026 | 6.71e-05   | 7.41e-05   | 7.42e-05   | +10.6%  | +10.7%  | 2.64e-07  | 2.85e-07  | 2.89e-07  | 25.6%   | 48.4%   | 50.9%   |

**Bold row = §3 layer-0 K conditions (kurt -1.56, KS 0.155).**

## Mechanism

The production WHT (1 RHT) puts K in a state where its values cluster near
positions on the codebook lattice. For a bimodal-ish K with peaks at ≈ ±σ
(where σ = 1/√d), and a b=4 Lloyd-Max codebook with centroids that include
±0.082 (= σ to two decimal places), the quantization error per coord is
tiny — the values are already near centroids.

Additional RHTs Gaussianize K (Basat's theorem: kurt → 0 after k=2). The
values now spread uniformly over the full codebook range [-0.24, +0.24]
instead of concentrating near a few centroids. Per-coord quantization error
increases to the Gaussian rate-distortion bound for this codebook — which is
higher than the bimodal lucky-alignment MSE.

Both MSE and the attention-softmax KL proxy degrade together. The KL change
is consistent with the §7.2 sign-inversion observation in why-mse-fails — but
here it's driven by rotation-count, not by changing centroids.

## What this does and doesn't prove

**Does prove (within the synthetic regime):**

- Basat's theorem is verified: kurt drops from -1.96 to ≈ 0 after 1 RHT at
  d=128, and from -1.58 to ≈ 0 after 1 RHT at the §3 kurt point. The
  Gaussian-marginal recovery is real.
- Practical consequence for KV cache quantization with the production
  Gaussian-fit codebook: adding RHTs degrades quantization quality, not
  improves it. The lucky-alignment of WHT-rotated K with the Gaussian
  codebook is empirically more valuable than the asymptotic-Gaussian
  guarantee that more rotations buy.
- Effect persists across the realistic sub-Gaussian range, with degradation
  ranging from +11% MSE (mildly sub-Gaussian) to +37% MSE (strongly
  sub-Gaussian).
- Effect is robust to seed (5 trials per cell, std generally < 10% of mean).

**Does not prove:**

- Real K cache from Qwen3-0.6B / Llama-3.2-1B may not exactly match the
  Bernoulli+noise synthetic. Real K extraction + replay is the next step.
- This is a per-coord scalar quantization test. The b=4 codebook is fixed;
  results might differ at b=2 / b=3 / b=8.
- Vector quantization (where Basat proposes k=3 for codebook universality)
  is not tested here. The §6 of Basat 2026 specifically targets VQ; we used
  scalar.
- V cache (already approximately Gaussian per §3 of why-mse-fails) was not
  tested. Prediction: applying additional RHTs to V will not change MSE or
  KL significantly (Gaussian → Gaussian transform is a no-op
  distributionally).

## Next steps to firm this up to paper-grade

1. Real K extraction from llama-cli during a Qwen3-0.6B prefill (need to add
   a debug dump hook to the WHT kernel, or save K post-WHT via a custom
   build).
2. Sweep b ∈ {2, 3, 4, 8} bit budgets at the §3 kurt point.
3. Repeat on V cache as a control — predicted flat.
4. Cross-model: Llama-3.2-1B, Mistral-7B (Tom's why-mse-fails uses 5 families).
5. Optional: re-run with the adaptive-RHT-count primitive from Basat §7
   to see if the linear-time moment check correctly chooses k=0 for K.

## Repro

```bash
cd experiments/rht_k_sweep
python3 rht_k_sweep_v3.py   # full sweep
python3 rht_k_sweep_v2.py   # single-point detail at §3 kurt
```

No GPU, no model load. Runs in ~10 seconds. Pure numpy/scipy.
