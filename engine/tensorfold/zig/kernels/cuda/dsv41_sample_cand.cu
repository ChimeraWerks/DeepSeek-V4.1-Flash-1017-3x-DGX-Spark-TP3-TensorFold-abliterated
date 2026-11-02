// One sampled row per block: exact top-k plus margin by value descending, id ascending.
// Scan only the logical vocabulary (n) at the padded row stride (ld); flags 1 for NaN,
// 2 for a tie group too large to settle. Both flags require the original CPU row path.
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ unsigned cand_key(float v) {
    const unsigned u = __float_as_uint(v == 0.f ? 0.f : v); // signed zero is one value
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ bool cand_before(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

extern "C" __global__ void __launch_bounds__(1024) tf_ds_top_cand_rows_kernel(
    const float* logits, unsigned long long ld, int n, int count,
    float* vals, long long* ids, int* flags) {
    constexpr int ties_cap = 2048, warps = 32;
    __shared__ unsigned whist[warps][256];
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, mask_s, left_s, above_s, ties_s;
    __shared__ float chosen_val[64];
    __shared__ int chosen_id[64];
    __shared__ int tie_id[ties_cap];
    const float* row = logits + (unsigned long long)blockIdx.x * ld;
    const int k = count < n ? count : n;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x == 0) {
        prefix_s = mask_s = above_s = ties_s = 0u;
        left_s = (unsigned)k;
    }
    __syncthreads();
    // Four radix passes find the k-th largest float key. Warp-local histograms avoid
    // serializing repeated logits into one counter.
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = threadIdx.x; b < warps * 256; b += blockDim.x)
            (&whist[0][0])[b] = 0u;
        __syncthreads();
        const unsigned prefix = prefix_s, mask = mask_s;
        bool nan = false;
        const int trips = (n + (int)blockDim.x - 1) / (int)blockDim.x;
        for (int t = 0; t < trips; t++) {
            const int i = t * (int)blockDim.x + (int)threadIdx.x;
            unsigned bin = 256u;
            if (i < n) {
                const float v = row[i];
                if (shift == 24) nan |= v != v;
                const unsigned key = cand_key(v);
                if ((key & mask) == prefix) bin = (key >> shift) & 255u;
            }
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
            if (bin != 256u && lane == __ffs(peers) - 1)
                atomicAdd(&whist[warp][bin], (unsigned)__popc(peers));
        }
        if (shift == 24 && __syncthreads_or(nan)) {
            if (threadIdx.x == 0) flags[blockIdx.x] = 1;
            return;
        }
        __syncthreads();
        for (int b = threadIdx.x; b < 256; b += blockDim.x) {
            unsigned sum = 0u;
            for (int w = 0; w < warps; w++) sum += whist[w][b];
            hist[b] = sum;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned above = 0u;
            int b = 255;
            for (; b > 0; b--) {
                if (above + hist[b] >= left_s) break;
                above += hist[b];
            }
            left_s -= above;
            prefix_s = prefix | ((unsigned)b << shift);
            mask_s = mask | (255u << shift);
        }
        __syncthreads();
    }
    const unsigned cut = prefix_s;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned key = cand_key(row[i]);
        if (key > cut) {
            const unsigned at = atomicAdd(&above_s, 1u);
            if (at < 64u) {
                chosen_val[at] = row[i];
                chosen_id[at] = i;
            }
        } else if (key == cut) {
            const unsigned at = atomicAdd(&ties_s, 1u);
            if (at < (unsigned)ties_cap) tie_id[at] = i;
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const unsigned above = above_s, ties = ties_s, left = left_s;
    if (ties > (unsigned)ties_cap || above + left != (unsigned)k) {
        flags[blockIdx.x] = 2;
        return;
    }
    for (unsigned j = 0; j < left; j++) {
        unsigned best = j;
        for (unsigned q = j + 1; q < ties; q++)
            if (tie_id[q] < tie_id[best]) best = q;
        const int tmp = tie_id[j];
        tie_id[j] = tie_id[best];
        tie_id[best] = tmp;
        chosen_val[above + j] = row[tie_id[j]];
        chosen_id[above + j] = tie_id[j];
    }
    for (int a = 1; a < k; a++) {
        const float v = chosen_val[a];
        const int id = chosen_id[a];
        int b = a;
        while (b > 0 && cand_before(v, id, chosen_val[b - 1], chosen_id[b - 1])) {
            chosen_val[b] = chosen_val[b - 1];
            chosen_id[b] = chosen_id[b - 1];
            b--;
        }
        chosen_val[b] = v;
        chosen_id[b] = id;
    }
    for (int a = 0; a < k; a++) {
        vals[(size_t)blockIdx.x * count + a] = chosen_val[a];
        ids[(size_t)blockIdx.x * count + a] = chosen_id[a];
    }
    flags[blockIdx.x] = 0;
}
