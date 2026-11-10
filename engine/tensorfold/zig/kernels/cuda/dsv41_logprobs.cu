// One fixed block per raw vocabulary shard row. The split LSE avoids losing a small log mass at a large offset.
#include <cuda_runtime.h>
#ifndef CUDART_INF_F // (the CPU emulation shim defines it; real CUDA keeps it in math_constants.h)
#include <math_constants.h>
#endif
#include <stdint.h>

namespace {
constexpr unsigned threads = 256, top_cap = 20;
struct Candidate { float value; uint32_t id; };
struct Summary {
    float maximum, mass, chosen;
    uint32_t owns, flags, count;
    Candidate top[top_cap];
};
static_assert(sizeof(Summary) == 184);
__device__ bool before(Candidate a, Candidate b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}
}

extern "C" __global__ void __launch_bounds__(threads) tf_ds_logprob_shard_kernel(
    const float* logits, unsigned long long ld, unsigned n, unsigned offset,
    const unsigned* chosen, unsigned k, Summary* summaries) {
    __shared__ float values[threads];
    __shared__ unsigned ids[threads], bad[threads];
    const unsigned lane = threadIdx.x;
    const float* row = logits + (unsigned long long)blockIdx.x * ld;
    Summary* out = summaries + blockIdx.x;
    if (lane < top_cap) out->top[lane] = {0.f, 0};
    float maximum = -CUDART_INF_F;
    unsigned flags = k > top_cap;
    for (unsigned i = lane; i < n; i += threads) {
        const float v = row[i];
        flags |= isnan(v) || v == CUDART_INF_F;
        maximum = fmaxf(maximum, v);
    }
    values[lane] = maximum;
    bad[lane] = flags;
    __syncthreads();
    for (unsigned d = threads / 2; d; d /= 2) {
        if (lane < d) { values[lane] = fmaxf(values[lane], values[lane + d]); bad[lane] |= bad[lane + d]; }
        __syncthreads();
    }
    maximum = values[0];
    flags = bad[0];
    if (lane == 0) {
        const unsigned token = chosen[blockIdx.x];
        out->maximum = maximum;
        out->mass = 0;
        out->owns = token >= offset && token - offset < n;
        out->chosen = out->owns ? row[token - offset] : -CUDART_INF_F;
        out->flags = flags;
        out->count = flags ? 0 : min(k, n);
    }
    if (flags) return;
    float sum = 0, correction = 0;
    if (maximum != -CUDART_INF_F) for (unsigned i = lane; i < n; i += threads) {
        const float x = expf(row[i] - maximum);
        const float y = x - correction, t = sum + y;
        correction = (t - sum) - y;
        sum = t;
    }
    __syncthreads();
    values[lane] = sum;
    __syncthreads();
    for (unsigned d = threads / 2; d; d /= 2) {
        if (lane < d) values[lane] += values[lane + d];
        __syncthreads();
    }
    if (lane == 0) out->mass = values[0];
    Candidate previous{CUDART_INF_F, 0};
    for (unsigned j = 0; j < min(k, n); ++j) {
        Candidate best{-CUDART_INF_F, UINT32_MAX};
        for (unsigned i = lane; i < n; i += threads) {
            const Candidate value{row[i], offset + i};
            if ((j == 0 || before(previous, value)) && before(value, best)) best = value;
        }
        values[lane] = best.value;
        ids[lane] = best.id;
        __syncthreads();
        for (unsigned d = threads / 2; d; d /= 2) {
            if (lane < d) {
                Candidate a{values[lane], ids[lane]}, b{values[lane + d], ids[lane + d]};
                if (before(b, a)) { values[lane] = b.value; ids[lane] = b.id; }
            }
            __syncthreads();
        }
        previous = {values[0], ids[0]};
        if (lane == 0) out->top[j] = previous;
        __syncthreads();
    }
}
