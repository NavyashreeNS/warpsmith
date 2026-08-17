# Deep dive: how each optimization works

This document explains the *why* behind every kernel in the repository. The
[benchmark report](../results/REPORT.md) has the numbers; this has the reasoning.

The through-line is a single idea, applied at every level of the machine:

> A GPU is not slow because it lacks arithmetic units. It is slow because the
> arithmetic units are waiting. Every optimization here is a way of making data
> arrive sooner, or of needing less of it.

---

## Table of contents

- [The two ceilings](#the-two-ceilings)
- [SGEMM: nine stages](#sgemm-nine-stages)
  - [Stage 1 → 2: coalescing](#stage-1--2-coalescing)
  - [Stage 2 → 3: shared memory tiling](#stage-2--3-shared-memory-tiling)
  - [Stage 3 → 4: register tiling in one dimension](#stage-3--4-register-tiling-in-one-dimension)
  - [Stage 4 → 5: the outer product](#stage-4--5-the-outer-product)
  - [Stage 5 → 6: vectorization and a transposed tile](#stage-5--6-vectorization-and-a-transposed-tile)
  - [Stage 6 → 7: warp tiling](#stage-6--7-warp-tiling)
  - [Stage 8: tensor cores](#stage-8-tensor-cores)
- [Reduction: the other shape of the problem](#reduction-the-other-shape-of-the-problem)
- [Transpose: shared memory banks](#transpose-shared-memory-banks)
- [Softmax and the online algorithm](#softmax-and-the-online-algorithm)
- [Attention: why FlashAttention wins](#attention-why-flashattention-wins)
- [How the measurements are taken](#how-the-measurements-are-taken)

---

## The two ceilings

Every kernel is bounded by one of two things, and knowing which one decides what
to optimize.

**The compute ceiling.** Each SM has a fixed number of FP32 lanes. Multiply by
the clock and by two (a fused multiply-add is two FLOPs) and you have the
arithmetic ceiling. Nothing that issues FFMA instructions can exceed it.

**The bandwidth ceiling.** DRAM delivers a fixed number of bytes per second.
Nothing that reads memory can exceed it.

Which ceiling binds depends on the kernel's **arithmetic intensity** - FLOPs
performed per byte of DRAM traffic. Set the two ceilings equal and solve, and you
get the **ridge point**: the intensity at which a kernel stops being starved by
memory and starts being limited by arithmetic. Below it, more arithmetic is free.
Above it, less memory traffic is free.

On the test GPU the ridge point is about **40 FLOP/byte**. That number explains
the whole of this repository:

| Kernel | Arithmetic intensity | Bound by |
|:---|---:|:---|
| Naive GEMM | ~0.25 FLOP/byte | memory, catastrophically |
| Tiled GEMM (32x32) | ~8 FLOP/byte | memory, still |
| Blocked GEMM (128x128) | ~32 FLOP/byte | approaching the ridge |
| Large GEMM overall | ~700 FLOP/byte | compute |
| Reduction, transpose, softmax, RMSNorm | < 1 FLOP/byte | memory, permanently |

A GEMM *algorithm* has enormous arithmetic intensity - `2*N^3` FLOPs over `3*N^2`
bytes. A naive GEMM *implementation* has almost none, because it re-reads the same
data from DRAM `N` times. **All of stages 2 through 7 are about recovering the
intensity that the algorithm always had.** Nothing changes the FLOP count; only
the byte count moves.

---

## SGEMM: nine stages

All stages compute the row-major operation `C = alpha*A@B + beta*C`, with
`A: MxK`, `B: KxN`, `C: MxN`.

### Stage 1 → 2: coalescing

The only difference between these two kernels is which index `threadIdx.x` feeds:

```cuda
// Stage 1 - threadIdx.x selects the row
const int row = blockIdx.x * 32 + threadIdx.x;
const int col = blockIdx.y * 32 + threadIdx.y;

// Stage 2 - threadIdx.x selects the column
const int col = blockIdx.x * 32 + threadIdx.x;
const int row = blockIdx.y * 32 + threadIdx.y;
```

Threads in a warp have consecutive `threadIdx.x`. The memory system serves a warp
by collecting its 32 addresses and issuing the smallest number of 32-byte sectors
that covers them.

In stage 1, lane `i` accesses `B[k*N + col]` where `col` is fixed per lane... but
`row` varies, so `A[row*K + k]` walks down a *column* of A: 32 addresses, each
`K*4` bytes apart, each in a different cache line. One warp instruction becomes 32
memory transactions, and of every 32-byte sector fetched, 4 bytes are used.

In stage 2, lane `i` reads `B[k*N + col + i]` - 32 consecutive floats, 128
contiguous bytes, **four** sectors instead of thirty-two. The read of A becomes
identical across all lanes, which the hardware serves as a broadcast.

The FLOP count is unchanged. The byte count falls by roughly 8x, and so does the
runtime. This is the single highest-leverage change in the entire progression, and
it is an index swap.

### Stage 2 → 3: shared memory tiling

Both previous kernels read `2K` floats per output element. The fix is reuse, and
reuse requires somewhere to keep data - shared memory, an explicitly managed
scratchpad private to a block, roughly as fast as L1.

A 32x32 block cooperatively loads a 32x32 tile of A and of B, then every thread
computes its partial dot product entirely out of shared memory:

```cuda
for (int k0 = 0; k0 < K; k0 += 32) {
  As[ty][tx] = A[row*K + k0 + tx];     // one global read per thread
  Bs[ty][tx] = B[(k0 + ty)*N + col];
  __syncthreads();
  for (int k = 0; k < 32; ++k)          // 32 uses of it
    acc += As[ty][k] * Bs[k][tx];
  __syncthreads();
}
```

Each element loaded from DRAM is consumed by 32 threads, so DRAM traffic drops
32-fold and arithmetic intensity rises from ~0.25 to ~8 FLOP/byte.

The kernel is now limited by a new bottleneck. Each thread holds *one*
accumulator, so it needs two shared-memory loads for every FMA. Shared memory is
fast, but a 2:1 load-to-math ratio means the load-store unit, not the FP32 units,
sets the pace.

### Stage 3 → 4: register tiling in one dimension

The same trick, one level further down: cache in **registers**, which are faster
than shared memory and (crucially) unlimited in bandwidth because each thread has
its own.

Each thread now owns 8 accumulators spanning 8 rows of one column of C. For each
step of the dot product it loads one element of B and reuses it against 8 elements
of A:

```cuda
float acc[8] = {0.0f};
for (int dot = 0; dot < BK; ++dot) {
  const float b = Bs[dot*BN + thread_col];   // 1 shared-memory load
  for (int m = 0; m < 8; ++m)
    acc[m] += As[(thread_row*8 + m)*BK + dot] * b;   // 8 FMAs
}
```

Nine shared-memory loads now feed eight FMAs instead of sixteen feeding eight.

### Stage 4 → 5: the outer product

Amortize in *both* directions. Each thread loads 8 elements of A and 8 of B into
registers, then performs their full **outer product**: 16 loads, 64 FMAs, a 4:1
math-to-load ratio.

```cuda
float acc[8][8], reg_m[8], reg_n[8];
for (int dot = 0; dot < BK; ++dot) {
  for (int m = 0; m < 8; ++m) reg_m[m] = As[...];   //  8 loads
  for (int n = 0; n < 8; ++n) reg_n[n] = Bs[...];   //  8 loads
  for (int m = 0; m < 8; ++m)
    for (int n = 0; n < 8; ++n)
      acc[m][n] += reg_m[m] * reg_n[n];             // 64 FMAs
}
```

This is the structural core of every fast dense GEMM ever written, cuBLAS and
CUTLASS included. It also introduces the tradeoff that dominates the rest of the
progression: 64 accumulators plus staging registers is a lot of register file, so
fewer warps fit on an SM. **Occupancy falls and throughput rises.** The usual
advice to maximize occupancy is wrong here - what matters is having enough
independent work in flight to hide latency, and 64 independent FMAs per thread is
a great deal of independent work.

### Stage 5 → 6: vectorization and a transposed tile

Two changes.

**`float4` everywhere.** One `LDG.E.128` instruction moves 16 bytes; four
`LDG.E.32` move the same bytes with four times the instruction issue, address
arithmetic and latency exposure. On a kernel whose inner loop is already
instruction-bound, this is close to free performance.

**A transposed A tile.** Stage 5 reads A down a column of the tile (stride `BK`),
which cannot be vectorized. Storing the tile transposed at load time - `As[k][m]`
instead of `As[m][k]` - makes the inner-loop read contiguous, so the `float4` load
becomes legal:

```cuda
const float4 v = *(const float4*)&A[row*K + col*4];
As[(k+0)*BM + m] = v.x;   // scatter into the transposed layout
As[(k+1)*BM + m] = v.y;   // costs nothing: the data is already in flight
As[(k+2)*BM + m] = v.z;
As[(k+3)*BM + m] = v.w;
```

The tile geometry is a template parameter here, not a constant, because the best
choice is a property of the specific GPU. `--suite tune` measures the whole search
space; see the autotune table in the report.

### Stage 6 → 7: warp tiling

An SM does not execute 256 independent threads. It executes 8 warps across 4
processing blocks, and the warp is the unit that shares an instruction stream and
issues a single shared-memory request. So there is a level of the hierarchy
between "block" and "thread" that stages 1-6 ignore.

Stage 7 partitions the 128x128 block tile into four 64x64 warp tiles, one per
warp, and covers each warp tile with 32 threads owning several small sub-tiles
rather than one large one. The 32 lanes of a warp then read the same narrow slice
of shared memory, so operands land in the register file once and are reused across
many more FMAs. Each thread accumulates 128 outputs.

This is the stage that reaches - and on larger problems exceeds - cuBLAS FP32.

### Stage 8: tensor cores

Everything so far issues FFMA on the FP32 pipelines, which is a hard ceiling.
Ampere has a second datapath: a tensor core consumes an entire small matrix
multiply as one instruction, and TF32 runs at twice the FP32 rate.

```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
wmma::load_matrix_sync(a, &As[...], BK);
for (int t = 0; t < a.num_elements; ++t)
  a.x[t] = wmma::__float_to_tf32(a.x[t]);   // 23-bit mantissa -> 10-bit
wmma::mma_sync(acc, a, b, acc);             // one instruction, 16x16x8 multiply
```

TF32 keeps FP32's 8-bit exponent, so dynamic range is unchanged and nothing
overflows that would not have overflowed before. The mantissa drops from 23 bits
to 10, and accumulation still happens in full FP32 inside the tensor core. The
measured relative error rises from ~5e-7 to ~3e-4, which is reported in the table
rather than hidden - this is exactly the tradeoff cuBLAS makes when TF32 math mode
is enabled, and is why stage 8 is benchmarked against the cuBLAS TF32 baseline as
well as the FP32 one.

**A known limitation, stated plainly.** This kernel reaches roughly a third of the
TF32 ceiling and is comfortably beaten by cuBLAS in TF32 mode. It demonstrates the
MMA programming model correctly, but it lacks the software pipelining a
competitive tensor-core GEMM needs: `cp.async` bulk copies with a multi-stage
shared-memory buffer, `ldmatrix` for fragment loads, and a swizzled layout to
avoid conflicts on the fragment reads. Closing that gap is the top item in
[future work](../README.md#what-is-not-here).

---

## Reduction: the other shape of the problem

GEMM is compute-bound and rewards blocking. Reduction is bandwidth-bound and
rewards getting out of the memory system's way: the theoretical best is simply
"read the array once at full bandwidth". Six formulations:

1. **Global atomics.** One `atomicAdd` per element. All 64 million threads
   serialize on the same 4 bytes. Correct, short, and about 80x slower than the
   answer.
2. **Shared-memory tree, interleaved addressing.** The classic first attempt. Its
   flaw is `if (tid % (2*stride) == 0)`: on the first iteration only even lanes are
   active, so every warp runs half-empty while still being issued, and the stride-2
   access pattern hits bank conflicts.
3. **Sequential addressing.** Same tree, same additions, but the active threads
   are the *first* half of the block. Whole warps retire together instead of every
   warp idling half its lanes.
4. **Grid-stride accumulation.** The tree is pure overhead - 8 barriers to turn
   256 numbers into 1. So do as much as possible before entering it: each thread
   accumulates many elements into a register first, and the grid is sized to the
   GPU rather than to the input.
5. **Warp shuffle.** `__shfl_xor_sync` lets a lane read another lane's register
   directly. Shared memory is then needed only to combine the 8 warps, removing
   most of the barriers and nearly all shared-memory traffic.
6. **Vectorized loads.** With the arithmetic free, only the load pipeline is left.
   `float4` accesses pin the kernel to the bandwidth roof, matching CUB.

A numerical result falls out for free. Floating-point addition is not associative,
so summation *order* changes the answer. A serial accumulation of `n` terms
accumulates error like `sqrt(n)`; a tree of depth `log2(n)` accumulates it like
`sqrt(log n)`. Measured against a Kahan-compensated double-precision sum, the tree
reductions are about three orders of magnitude more accurate than the atomic
version. **The parallel algorithm is not only faster, it is more correct.**

---

## Transpose: shared memory banks

Transpose does no arithmetic at all, which makes it the cleanest demonstration of
pure memory behaviour.

The naive kernel reads along rows and writes along columns; one of the two is
always strided. Staging a tile through shared memory makes *both* global accesses
contiguous - the transpose happens inside the SM.

But then a second effect appears. Shared memory is 32 banks of 4 bytes, striped
across addresses. A `float tile[32][32]` puts an entire column in a single bank,
so reading the tile column-wise serializes 32 ways.

```cuda
__shared__ float tile[32][32];      // column-wise read: 32-way bank conflict
__shared__ float tile[32][33];      // one word of padding: conflict-free
```

Padding the row stride to 33 shifts each row by one bank, so a column touches 32
distinct banks. **One character, measured.** A final variant processes 4 rows per
thread, which quarters the block count, index arithmetic and barriers, and brings
the kernel to within a percent of a pure copy - the true upper bound, since
transpose cannot beat moving the same bytes without rearranging them.

---

## Softmax and the online algorithm

Softmax as defined reads its input three times: once for the maximum (needed for
numerical stability), once for the sum of exponentials, once to normalize. On a
bandwidth-bound operation each pass is a direct multiplier on runtime, so the
entire optimization is "read the input fewer times".

The **online** formulation removes the dependency that seems to force two passes.
Carry a running maximum `m` and running sum `l`; when a larger element appears,
retroactively correct the sum by the exact factor `exp(m_old - m_new)`:

```cuda
const float m_new = fmaxf(m, x);
l = l * __expf(m - m_new) + __expf(x - m_new);
m = m_new;
```

One pass, and *numerically identical* to the two-pass version - not an
approximation. Nothing is ever exponentiated with a positive argument, so nothing
overflows. A final variant caches the row in shared memory so DRAM sees the
theoretical minimum of one read and one write, reaching ~93% of peak bandwidth.

This is the key that unlocks the next section.

---

## Attention: why FlashAttention wins

Scaled dot-product attention is `softmax(Q@K^T / sqrt(d)) @ V`. The textbook
implementation:

1. compute `S = Q@K^T`, an `S x S` matrix, and write it to DRAM;
2. read it back, softmax each row, write it again;
3. read it a third time to compute `P@V`.

Both implementations in this repository perform the same `4*S^2*D` multiply-adds.
The difference is entirely traffic and memory:

| | Materialized | Fused |
|:---|:---|:---|
| Scratch memory | `O(S^2)` - 64 MiB at S=4096 | `O(1)` |
| Round trips through the score matrix | 4 | 0 |
| Kernel launches per head | 3 | 1 |

The fused kernel tiles over key/value blocks, keeps the query row in registers and
the running softmax state `(m, l)` alongside it, and applies the online correction
to the *accumulator* as well as the denominator whenever the running maximum
changes:

```cuda
const float m_new = fmaxf(m, tile_max);
const float corr  = __expf(m - m_new);
l = l * corr + l_tile;
for (int d = 0; d < D; ++d) acc[d] *= corr;     // rescale the output so far
for (int j = 0; j < BN; ++j)                     // then add this tile
  for (int d = 0; d < D; ++d) acc[d] += p[j] * Vs[j*D + d];
```

The `S x S` matrix never exists.

**Causal masking is where the gap widens.** Autoregressive decoding only attends
backwards, so half the score matrix is thrown away. The fused kernel skips whole
key tiles above the diagonal before issuing a single dot product; the materialized
version must compute those scores and then discard them. Measured speedup rises
from ~1.7x to ~2.5x.

---

## How the measurements are taken

Benchmarking a GPU kernel badly is easy. The harness in
[`include/warpsmith/bench.cuh`](../include/warpsmith/bench.cuh) avoids the usual
mistakes:

- **CUDA events on the stream**, so host-side launch overhead and driver
  scheduling jitter are excluded.
- **Untimed warm-up launches** before the first sample, so SM and memory clocks
  have settled and the instruction cache is hot.
- **Many samples, median reported**, with p95 and the coefficient of variation
  published alongside. A measurement without a stability figure is an anecdote.
- **Correctness before speed.** Every kernel is validated against a trusted
  reference - cuBLAS for GEMM, CUB for reduction, double-precision host code
  elsewhere - and the benchmark exits non-zero if any kernel fails.
- **Tolerances that scale.** A `K`-term FP32 dot product accumulates rounding
  error like `sqrt(K)`, so a fixed absolute tolerance would wrongly fail large
  problems. The bar is a relative L2 norm scaled by `sqrt(K)`.

One caveat worth stating: `% of peak` is computed from the clock the driver
reports (`cudaDeviceProp::clockRate`). A laptop GPU boosts above that figure when
cool and throttles well below it under sustained load, so the denominator is a
nominal number rather than an instantaneous one. The `cv` column is where that
variability shows up, and it is the honest reason some rows have a high one.
