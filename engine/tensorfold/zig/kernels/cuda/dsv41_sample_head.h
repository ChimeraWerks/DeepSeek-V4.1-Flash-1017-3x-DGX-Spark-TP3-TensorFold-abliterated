#ifndef TF_DS_SAMPLE_HEAD_H
#define TF_DS_SAMPLE_HEAD_H
#include "dsv41_greedy_head.h"
#ifdef __CUDACC__
#define TF_DS_SAMPLE_HD __host__ __device__ __forceinline__
#else
#define TF_DS_SAMPLE_HD inline
#endif
// Shared CPU/device merge, including the exact global cutoff fallback rule.
TF_DS_SAMPLE_HD void tf_ds_sample_head_merge(
    const TfDsHeadCandidate* all, uint32_t row, uint32_t rows, uint32_t world,
    uint32_t count, uint32_t* vals, long long* ids, int* flags) {
    uint32_t cursor[3] = {0, 0, 0};
    if (world != 3) { flags[row] = 2; return; }
    uint32_t nan = 0;
    for (uint32_t rank = 0; rank < world; ++rank)
        nan |= all[((unsigned long long)rank * rows + row) * (count + 1) + count].bits;
    if (nan) { flags[row] = 1; return; }
    TfDsHeadCandidate best = tf_ds_head_empty();
    for (uint32_t k = 0; k < count; ++k) {
        best = tf_ds_head_empty();
        uint32_t owner = 0;
        for (uint32_t rank = 0; rank < world; ++rank) {
            if (cursor[rank] == count) continue;
            const auto candidate = all[((unsigned long long)rank * rows + row) * (count + 1) + cursor[rank]];
            const auto picked = tf_ds_head_pick(best, candidate);
            if (picked.id != best.id) owner = rank;
            best = picked;
        }
        if (best.id == UINT32_MAX) { flags[row] = 2; return; }
        ++cursor[owner];
        vals[(unsigned long long)row * count + k] = best.bits;
        ids[(unsigned long long)row * count + k] = best.id;
    }
    uint32_t ties = 0;
    const uint32_t cut = tf_ds_head_key(best.bits);
    for (uint32_t rank = 0; rank < world; ++rank) {
        const auto* src = all + ((unsigned long long)rank * rows + row) * (count + 1);
        const auto local_cut = src[count - 1];
        if (local_cut.id != UINT32_MAX && tf_ds_head_key(local_cut.bits) == cut)
            ties += src[count].id;
        else
            for (uint32_t k = 0; k < count; ++k)
                if (src[k].id != UINT32_MAX && tf_ds_head_key(src[k].bits) == cut) ++ties;
    }
    flags[row] = ties > 2048 ? 2 : 0;
}
#undef TF_DS_SAMPLE_HD
#endif
