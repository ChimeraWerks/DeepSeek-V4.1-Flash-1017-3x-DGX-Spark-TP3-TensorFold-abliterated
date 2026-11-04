#ifndef TF_DS_GREEDY_HEAD_H
#define TF_DS_GREEDY_HEAD_H
#include <stdint.h>
#ifdef __CUDACC__
#define TF_DS_HD __host__ __device__ __forceinline__
#else
#define TF_DS_HD inline
#endif

// Raw fp32 bits, never a conversion/arithmetic result. UINT32_MAX is invalid.
struct TfDsHeadCandidate { uint32_t bits, id; };
static_assert(sizeof(TfDsHeadCandidate) == 8, "all-gather tuple ABI");

TF_DS_HD bool tf_ds_head_nan(uint32_t x) {
    return (x & 0x7fffffffu) > 0x7f800000u;
}
TF_DS_HD TfDsHeadCandidate tf_ds_head_empty() {
    return {0, UINT32_MAX};
}
TF_DS_HD bool tf_ds_head_valid(TfDsHeadCandidate x) {
    return x.id != UINT32_MAX && (!tf_ds_head_nan(x.bits) || x.id == 0);
}
// Order numeric IEEE32 encodings, but consider both zeros equal. Preserve the
// original bits in the candidate: this key is for comparison ONLY.
TF_DS_HD uint32_t tf_ds_head_key(uint32_t x) {
    if ((x & 0x7fffffffu) == 0) x = 0;
    return (x & 0x80000000u) ? ~x : (x ^ 0x80000000u);
}
// Exact tf_ds_argmax_rows_kernel semantics, not torch/MLX first-NaN:
// only a NaN at GLOBAL token zero is absorbing; all other NaNs are ignored.
TF_DS_HD TfDsHeadCandidate tf_ds_head_pick(TfDsHeadCandidate a, TfDsHeadCandidate b) {
    if (!tf_ds_head_valid(a)) return tf_ds_head_valid(b) ? b : tf_ds_head_empty();
    if (!tf_ds_head_valid(b)) return a;
    if (tf_ds_head_nan(a.bits)) return a;
    if (tf_ds_head_nan(b.bits)) return b;
    const uint32_t ak = tf_ds_head_key(a.bits), bk = tf_ds_head_key(b.bits);
    if (ak != bk) return ak > bk ? a : b;
#ifdef TF_DS_HEAD_REVERSE_TIE
    return a.id > b.id ? a : b;
#else
    return a.id < b.id ? a : b;
#endif
}
#undef TF_DS_HD
#endif
