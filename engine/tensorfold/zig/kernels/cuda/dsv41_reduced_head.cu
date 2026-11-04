// Optional reduced heads only. Never rebuild the production dsv41_ops fatbin.
#include <cuda_runtime.h>
#include <stdint.h>
#include "dsv41_greedy_head.h"
#include "dsv41_sample_head.h"

// Current GEMM's fp32 bytes [rows,ld]; clip padding by global logical vocabulary.
// One 1024-thread block per row, same scan/warp/shared topology as full argmax.
extern "C" __global__ void __launch_bounds__(1024) tf_ds_greedy_head_local_kernel(
    const uint32_t* logits, unsigned long long ld, uint32_t vocab, uint32_t offset,
    TfDsHeadCandidate* out) {
    const uint32_t* row = logits + (unsigned long long)blockIdx.x * ld;
    TfDsHeadCandidate best = tf_ds_head_empty();
    const unsigned long long n = offset < vocab ? min(ld, (unsigned long long)(vocab - offset)) : 0;
    for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x)
        best = tf_ds_head_pick(best, {row[i], offset + (uint32_t)i});
    for (int off = 16; off > 0; off >>= 1) {
        TfDsHeadCandidate other = {
            __shfl_down_sync(0xffffffffu, best.bits, off),
            __shfl_down_sync(0xffffffffu, best.id, off)};
        best = tf_ds_head_pick(best, other);
    }
    __shared__ TfDsHeadCandidate shared[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = best;
    __syncthreads();
    if (warp != 0) return;
    best = lane < (blockDim.x + 31) / 32 ? shared[lane] : tf_ds_head_empty();
    for (int off = 16; off > 0; off >>= 1) {
        TfDsHeadCandidate other = {
            __shfl_down_sync(0xffffffffu, best.bits, off),
            __shfl_down_sync(0xffffffffu, best.id, off)};
        best = tf_ds_head_pick(best, other);
    }
    if (lane == 0) out[blockIdx.x] = best;
}

// Tiny rank-major gather [world,rows]; no host copies and no fp32 arithmetic.
extern "C" __global__ void tf_ds_greedy_head_merge_kernel(
    const TfDsHeadCandidate* all, uint32_t rows, uint32_t world, uint32_t* out) {
    if (threadIdx.x != 0) return;
    TfDsHeadCandidate best = tf_ds_head_empty();
    for (uint32_t rank = 0; rank < world; ++rank)
        best = tf_ds_head_pick(best, all[(unsigned long long)rank * rows + blockIdx.x]);
    out[blockIdx.x] = best.id;
}

// Exact local top-K, including original fp32 bits and lowest-ID tie breaking.
// K is today's candidate count (top_k + margin), not merely top_k.
// One extra 8-byte pair carries any-NaN and full local cutoff tie count.
extern "C" __global__ void __launch_bounds__(1024) tf_ds_sample_head_local_kernel(
    const uint32_t* logits, unsigned long long ld, uint32_t vocab, uint32_t offset,
    uint32_t count, TfDsHeadCandidate* out) {
    __shared__ TfDsHeadCandidate shared[32], previous;
    __shared__ uint32_t nan_flag, ties;
    const uint32_t* row = logits + (unsigned long long)blockIdx.x * ld;
    TfDsHeadCandidate* dst = out + (unsigned long long)blockIdx.x * (count + 1);
    const unsigned long long n = offset < vocab ? min(ld, (unsigned long long)(vocab - offset)) : 0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) { nan_flag = ties = 0; previous = tf_ds_head_empty(); }
    __syncthreads();
    for (uint32_t k = 0; k < count; ++k) {
        TfDsHeadCandidate best = tf_ds_head_empty();
        for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x) {
            const TfDsHeadCandidate c = {row[i], offset + (uint32_t)i};
            if (tf_ds_head_nan(c.bits)) { if (k == 0) atomicOr(&nan_flag, 1u); continue; }
            if (k > 0) {
                if (previous.id == UINT32_MAX) continue;
                const uint32_t ck = tf_ds_head_key(c.bits), pk = tf_ds_head_key(previous.bits);
                if (ck > pk || (ck == pk && c.id <= previous.id)) continue;
            }
            best = tf_ds_head_pick(best, c);
        }
        for (int off = 16; off > 0; off >>= 1) {
            TfDsHeadCandidate other = {__shfl_down_sync(0xffffffffu, best.bits, off),
                                      __shfl_down_sync(0xffffffffu, best.id, off)};
            best = tf_ds_head_pick(best, other);
        }
        if (lane == 0) shared[warp] = best;
        __syncthreads();
        if (warp == 0) {
            best = shared[lane];
            for (int off = 16; off > 0; off >>= 1) {
                TfDsHeadCandidate other = {__shfl_down_sync(0xffffffffu, best.bits, off),
                                          __shfl_down_sync(0xffffffffu, best.id, off)};
                best = tf_ds_head_pick(best, other);
            }
            if (lane == 0) { dst[k] = best; previous = best; }
        }
        __syncthreads();
    }
    uint32_t local_ties = 0;
    if (previous.id != UINT32_MAX) {
        const uint32_t cut = tf_ds_head_key(previous.bits);
        for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x)
            if (!tf_ds_head_nan(row[i]) && tf_ds_head_key(row[i]) == cut) ++local_ties;
    }
    atomicAdd(&ties, local_ties);
    __syncthreads();
    if (threadIdx.x == 0) dst[count] = {nan_flag, ties};
}

// The union theorem: a token outside a shard's top-K has K predecessors
// already on that shard and cannot enter the global top-K.
extern "C" __global__ void tf_ds_sample_head_merge_kernel(
    const TfDsHeadCandidate* all, uint32_t rows, uint32_t world, uint32_t count,
    uint32_t* vals, long long* ids, int* flags) {
    if (threadIdx.x == 0) tf_ds_sample_head_merge(all, blockIdx.x, rows, world, count, vals, ids, flags);
}
