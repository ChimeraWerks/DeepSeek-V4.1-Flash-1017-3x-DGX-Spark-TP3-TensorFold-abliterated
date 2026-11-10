// Isolated opt-in logit penalties. No production EXL3 binary changes.
#include <stdint.h>
#include <math.h>
#ifndef TF_PENALTIES_HOST
#include <cuda_runtime.h>
#define TF_HD __host__ __device__
#else
#define TF_HD
#endif
struct TfPenaltyCount { uint32_t count, repeats; };
struct TfPenaltyParams { double presence, frequency, repetition; };
static_assert(sizeof(TfPenaltyCount) == 8);
static_assert(sizeof(TfPenaltyParams) == 24);
TF_HD inline float tf_penalty(float value, TfPenaltyCount c, TfPenaltyParams p) {
    if (!isfinite(value)) return value;
    if (p.presence == 0 && p.frequency == 0 && p.repetition == 1) return value;
    double x = value;
    if (c.repeats && p.repetition != 1) x = x > 0 ? x / p.repetition : x * p.repetition;
    double additive = (c.count ? p.presence : 0.0) + p.frequency * double(c.count);
    return float(x - additive);
}
#ifndef TF_PENALTIES_HOST
extern "C" __global__ void tf_ds_penalties_kernel(const float* src, float* dst,
    const TfPenaltyCount* counts, const TfPenaltyParams* params, uint32_t width, uint32_t rows) {
    for (uint64_t i = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < uint64_t(width)*rows;
         i += uint64_t(gridDim.x)*blockDim.x) dst[i] = tf_penalty(src[i], counts[i], params[i/width]);
}
#endif
