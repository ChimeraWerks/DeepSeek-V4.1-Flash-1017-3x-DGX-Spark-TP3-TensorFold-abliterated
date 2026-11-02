// Integer-only worklist partition for TF_DS_PREFILL_SHARED_SPLIT.
// Independent module: never recompile/replace the measured W arithmetic cubin.
// Rebuild (CPU only), from engine/tensorfold:
// nvcc --cubin -arch=sm_121 -O3 -std=c++17 -Xptxas=-v \
//   zig/kernels/prefill_128/shared_split.cu \
//   -o zig/src/families/dsv41/fixtures/shared_split.cubin
#include <cuda_runtime.h>

__device__ bool is_shared(int2 pair, const int* ids, int used, int shared) {
    // Both original and measured W only reject u >= used, not negative u.
    if (pair.x < 0 || (pair.x < used && pair.y < 0)) __trap();
    return pair.x < used && ids[pair.x] == shared;
}

extern "C" __global__ void tf_shared_split_partition(
    int2* work_gu, int n_gu, int2* work_d, int n_d,
    const int* ids, const int* used_ptr, const int* counts,
    const int* members, int maxu, int maxm, int shared, int rows, int slots) {
    // Exactly two CTAs, one owner per disjoint list. All router checks remain
    // on EVERY call, including already-qualified and graph-captured calls.
    const int tid = threadIdx.x;
    const int used = *used_ptr;
    if (used <= 0 || used > maxu || counts[shared] != rows) __trap();
    __shared__ int shared_u, shared_count;
    if (tid == 0) { shared_u = -1; shared_count = 0; }
    __syncthreads();
    for (int u = tid; u < used; u += blockDim.x) {
        if (ids[u] == shared) {
            atomicExch(&shared_u, u);
            atomicAdd(&shared_count, 1);
        }
    }
    __syncthreads();
    if (shared_count != 1) __trap();
    // Do not assume shared's group-place equals its expert ID.
    for (int row = tid; row < rows; row += blockDim.x)
        if (members[shared_u * maxm + row] != row * 32 + slots - 1) __trap();

    int2* work = blockIdx.x ? work_d : work_gu;
    const int n = blockIdx.x ? n_d : n_gu;
    // Coalesced pair reads; one classification bit per pair, not a second list.
    // Every lane participates in every ballot, including the partial final tile.
    extern __shared__ unsigned shared_bits[];
    for (int base = 0; base < n; base += blockDim.x) {
        const int i = base + tid;
        const bool shared_pair = i < n && is_shared(work[i], ids, used, shared);
        const unsigned bits = __ballot_sync(0xffffffffu, shared_pair);
        if ((tid & 31) == 0 && i < n) shared_bits[i >> 5] = bits;
    }
    __syncthreads();
    if (tid != 0) return;

    // EXACT original two-pointer permutation, not merely the same sets.
    // Skip only runs the old loop would have advanced over without writing.
    // Each swap is identical; only the surviving left bit needs updating.
    int left = 0, right = n;
    while (left < right) {
        const unsigned bits = shared_bits[left >> 5] & (0xffffffffu << (left & 31));
        if (!bits) { left = (left | 31) + 1; continue; }
        left = (left & ~31) + __ffs(bits) - 1;
        if (left >= right) break;
        --right;
        if (left == right) break;
        const bool right_shared = (shared_bits[right >> 5] >> (right & 31)) & 1u;
        const int2 tmp = work[left];
        work[left] = work[right];
        work[right] = tmp;
        if (!right_shared) shared_bits[left >> 5] &= ~(1u << (left & 31));
    }
    // Inactive padding stays routed, with every pair and member code preserved.
    if (n - right != (rows + 63) / 64) __trap();
}
