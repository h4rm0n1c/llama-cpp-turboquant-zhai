# q4_0 SET_ROWS Kernel Implementation Plan

**Branch:** `feat/q4_0-set-rows-kernel` (based on `merge/upstream-sync@3097b23c2`)

## Goal

Make `q4_0` KV V-cache as fast as `turbo3` for SET_ROWS operations, enabling 4-bit (3.56x) compression on KV caches with any head dimension (including head_dim=64 which turbo3's 128-block can't support).

This unlocks asymmetric KV cache experiments (e.g. `cache-type-k = q8_0`, `cache-type-v = q4_0`) for higher total compression with no quality loss on the key side.

## Background

### The speed problem

`q4_0` currently uses the generic `set_rows_cuda_quant` dispatch path (`set-rows.cu:1181`):

```cpp
} else if (dst->type == GGML_TYPE_Q4_0) {
    set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
        src0_d, src1_d, (block_q4_0*)dst->data, ...);
```

This launches `k_set_rows_quant`, a general template where each thread:
1. Reads 32 float values from source (serial loop)
2. Finds max absolute value (serial loop)
3. Computes scale
4. Scales and rounds each value (serial loop)
5. Packs nibbles (serial loop)
6. Writes 1 `block_q4_0` (18 bytes)

All 32 elements in a block are handled by one thread — serialized. No warp-cooperative parallelism, no vectorized memory access.

`turbo3`'s hand-tuned `k_set_rows_turbo3` uses 128/64 threads per block, doing parallel warp-reduce for norm, butterfly WHT, and warp-shuffle nibble packing. Despite the vastly more complex math (WHT + PolarQuant + norm correction), it's faster because it fully utilizes the GPU's parallel throughput.

### What q4_0 needs

q4_0 is a simpler format than turbo3 — no WHT, no PolarQuant, no norm correction. A hand-tuned kernel can be much shorter than turbo3's 190 lines while achieving the same speedup.

## The Fix: `k_set_rows_q4_0`

### Kernel design

One CUDA block per `QK4_0=32` element chunk, **32 threads per block**:

```
Thread j (0..31):
  1. Load element j from source row           // coalesced float read
  2. |val| = fabsf(value)
  3. Warp reduce: find max |val| across warp  // __shfl_xor_sync tree
  4. Thread 0: d = vmax / -8                  // scale (q4_0 symmetric format)
  5. Broadcast d via __shfl_sync
  6. q = round(value/d + 8.5)                 // quantize to [0,15]
  7. Clamp to uint4 (min(15, max(0, q)))
  8. Warp shuffle: pair with (j^1) to pack 2 nibbles into 1 byte
  9. Even threads write qs[j/2]
  10. Thread 0 writes d as fp16 (blk->d)
```

### q4_0 block format (from `ggml-common.h`)

```c
#define QK4_0 32
typedef struct {
    ggml_half  d;           // 2 bytes: scale (fp16)
    uint8_t    qs[QK4_0/2]; // 16 bytes: nibble-packed 4-bit indices
} block_q4_0;               // 18 bytes total
// 18 bytes / 32 values = 4.5 bits/value → 3.56x compression vs fp16
```

### Quantize math (from `cpy-utils.cuh:17`)

```c
amax = max(|v|) for all 32 elements
d = vmax / -8           // where vmax is the value with max |v|
id = 1.0f / d
for j in 0..31:
    xj = value[j] * id
    qj = min(15, (uint8_t)(xj + 8.5f))   // round to [0,15], symmetric offset
// pack: qs[j/2] = q[j*2] | (q[j*2+1] << 4)
```

### Files to modify

| File | Change | Lines |
|---|---|---|
| `ggml/src/ggml-cuda/set-rows.cu` | New kernel `k_set_rows_q4_0` + launcher `set_rows_cuda_q4_0` + dispatch | ~80 |
| `ggml/src/ggml-cuda/template-instances/set-rows-instance-q4_0.cu` | New template instance file | ~10 |

#### set-rows.cu additions

**1. Kernel (~50 lines)** — Add after the turbo4 section, before the tail functions:

```cuda
template <typename idx_t>
__launch_bounds__(32)
static __global__ void k_set_rows_q4_0(
    const float * __restrict__ src0,
    const idx_t * __restrict__ src1,
    block_q4_0 * __restrict__ dst,
    const int64_t ne00, const int64_t ne01,
    const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
    const int64_t s01, const int64_t s02, const int64_t s03,
    const int64_t s10, const int64_t s11, const int64_t s12,
    const int64_t s1,  const int64_t s2,  const int64_t s3) {

    const int j = threadIdx.x;  // element index within block (0..31)

    // blockIdx.x = flat block index
    const int64_t blocks_per_row = ne00 / QK4_0;
    const int64_t b = blockIdx.x;
    const int64_t i_blk = b % blocks_per_row;
    int64_t tmp = b / blocks_per_row;
    const int64_t i01 = tmp % ne01;
    tmp /= ne01;
    const int64_t i02 = tmp % ne12;
    const int64_t i03 = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_q4_0 * dst_row_ptr = (block_q4_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_q4_0 * blk = dst_row_ptr + i_blk;

    const float val = src_row[i_blk * QK4_0 + j];

    // Warp reduce: find |amax|
    float v = fabsf(val);
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));

    // Thread 0 computes scale, broadcast
    __shared__ float s_d;
    if (j == 0) {
        // Need vmax (signed) for scale formula. Re-find from lane 0's value.
        // Simpler: use the amax value and negate if needed.
        const float vmax = __shfl_sync(0xffffffff, val, 0);
        s_d = (vmax >= 0 ? v : -v) / -8;  // d = vmax / -8
    }
    __syncthreads();
    const float d = s_d;
    const float id = d != 0.0f ? 1.0f/d : 0.0f;

    // Quantize
    const float xf = val * id;
    const uint8_t q = min(15, (uint8_t)(xf + 8.5f));

    // Nibble pack: pair threads (even/odd) share one byte
    const uint8_t q_lo = __shfl_sync(0xffffffff, q, j & ~1);
    const uint8_t q_hi = j & 1 ? q << 4 : 0;
    if (j % 2 == 0) blk->qs[j/2] = q_lo | q_hi;

    // Write scale
    if (j == 0) blk->d = __float2half(d);

    GGML_UNUSED(ne10); GGML_UNUSED(ne13);
    GGML_UNUSED(ne00);
}
```

**2. Launcher (~20 lines)**

```cuda
static void set_rows_cuda_q4_0(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;
    block_q4_0 * dst_d = (block_q4_0 *)dst->data;

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];

    GGML_ASSERT(ne00 % QK4_0 == 0);

    const int64_t blocks_per_row = ne00 / QK4_0;
    const int64_t total_blocks = blocks_per_row * ne01 * ne02 * ne03;

    const dim3 block_dim(32);
    const dim3 grid_dim(total_blocks);

    k_set_rows_q4_0<<<grid_dim, block_dim, 0, ctx.stream()>>>(
        src0_d, src1_d, dst_d,
        ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13,
        nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3);
}
```

**3. Dispatch plug (2 lines)** — Replace the generic path at line 1181:

```cpp
} else if (dst->type == GGML_TYPE_Q4_0) {
    set_rows_cuda_q4_0<idx_t>(ctx, src0, src1, dst);
```

#### Template instance file (~10 lines)

New file `ggml/src/ggml-cuda/template-instances/set-rows-instance-q4_0.cu`:

```cpp
#include "set-rows.cu"

template void set_rows_cuda_q4_0<int64_t>(...);
template void set_rows_cuda_q4_0<int32_t>(...);
```

### What won't change

| Component | Status | Reason |
|---|---|---|
| VEC FA at D=64 | ✅ Already works | `FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)` at fattn.cu:286 |
| FA dispatch `case GGML_TYPE_Q4_0: break;` | ✅ Already passes | No head_dim restriction (line 506) |
| `ggml_type` registration | ✅ Already exists | In `ggml.c` |
| `kv_cache_types` | ✅ Already listed | In `common/arg.cpp:395` |
| Proxy VRAM registry | ✅ Already has q4_0 | In `qz_vram_snapshot.py:85` |
| `models-preset.ini` | ✅ Already configurable | `cache-type-v = q4_0` |
| V cache validation | ✅ Block 32 passes | `64 % 32 == 0` |

### Testing

1. Build with `cmake --build build --target llama-server`
2. Run with `--cache-type-v q4_0` on an LFM2.5-8B model (head_dim=64)
3. Verify model loads without `n_embd_head_v` error
4. Compare tokens/sec vs `q8_0` baseline
5. Try asymmetric: `--cache-type-k q8_0 --cache-type-v q4_0`

### Future expansions

Once the q4_0 kernel pattern is established, extending to other 32-block types is mechanical:

| Type | Block | Extra kernel complexity |
|---|---|---|
| `q4_1` | 32 | Add min/max dual reduce (vmin + vmax instead of |amax|) |
| `q5_0` | 32 | 5-bit packing (not nibble-aligned — 8 bytes per 5 elements) |
| `q5_1` | 32 | 5-bit + min/max |
