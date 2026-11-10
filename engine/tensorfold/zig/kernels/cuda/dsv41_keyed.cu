// Isolated keyed draft noise and top-k filter. Never rebuild production dsv41_ops for this feature.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

// TF_DS_KEYED_DRAFT: a draft step's logits plus the served sampler's keyed Gumbel noise. sampling.zig's score of a
// candidate is value / T + gumbel(seed, position, id), so the target's pick maximises value + T * gumbel; this adds
// T * gumbel(seed, pos + step, col0 + v) to logits row r (rows l_row floats apart, `width` columns, global column ids
// from col0) of every stream r with T > 0 and leaves the others alone. keyBits is sampling.keyBits (splitmix64 of
// seed, position, id) in u64 ops, the uniform and the Gumbel draw are double as the host takes them; the only
// divergence from the host score is the f64 -> f32 rounding of T * gumbel and the device log against libm's.
struct TfDsKeyedRow {
    uint64_t seed;
    double temp;
    int64_t pos;  // the absolute position of step 0's draft
    uint32_t top_k;
    uint32_t reserved;
};
static_assert(sizeof(TfDsKeyedRow) == 32);

__device__ __forceinline__ uint64_t tf_ds_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// grid (ceil(width / blockDim), streams); the row table is device memory so a captured graph reads each pass's seeds.
extern "C" __global__ void tf_ds_keyed_noise_kernel(float* logits, uint64_t l_row, uint32_t width, uint64_t col0,
                                                    const TfDsKeyedRow* rows, uint32_t step) {
    const TfDsKeyedRow k = rows[blockIdx.y];
    if (!(k.temp > 0.0)) return;
    const double t = fmax(k.temp, 1e-6);
    const uint64_t pos = uint64_t(k.pos) + step;
    float* row = logits + uint64_t(blockIdx.y) * l_row;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < width; v += gridDim.x * blockDim.x) {
        uint64_t x = tf_ds_mix64(k.seed + 0x9E3779B97F4A7C15ull);
        x = tf_ds_mix64(x ^ (pos * 0xD1B54A32D192ED03ull));
        x = tf_ds_mix64(x ^ (col0 + v));
        const double u = double(x >> 11) * 0x1p-53 + 0x1p-54;
        // Preserve masked/padded -infinity even at the rare rounded uniform of 1.
        if (row[v] != -INFINITY) row[v] += float(t * -log(-log(u)));
    }
}

// NaN scores sort below every numeric score, including negative infinity.
__device__ __forceinline__ uint32_t tf_ds_score_key(float score) {
    if (isnan(score)) return 0u;
    const uint32_t bits = __float_as_uint(score);
    return bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

// grid (1, streams), any power-of-two blockDim.x <= 1024. Bias selects the mask only.
// Count all keys at or above each radix trial; threshold ties all survive.
extern "C" __global__ void tf_ds_keyed_filter_kernel(
    float* logits, uint64_t l_row, uint32_t width, uint64_t col0,
    const TfDsKeyedRow* rows, uint32_t step,
    const int64_t* tok, uint64_t t_stride,
    const int32_t* slot,
    const __half* cache, uint64_t c_row,
    const __half* stage, uint64_t s_row) {
    const uint32_t r = blockIdx.y;
    const TfDsKeyedRow k = rows[r];
    if (!(k.temp > 0.0)) return;
    const double t = fmax(k.temp, 1e-6);
    const uint64_t pos = uint64_t(k.pos) + step;
    float* row = logits + uint64_t(r) * l_row;
    const int64_t id = tok[uint64_t(r) * t_stride];
    const int32_t sl = slot[id];
    const __half* bias = sl >= 0 ? cache + uint64_t(sl) * c_row : stage + uint64_t(r) * s_row;
    __shared__ uint32_t count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    uint32_t local = 0;
    for (uint64_t v = threadIdx.x; v < width; v += blockDim.x)
        local += row[v] != -INFINITY;
    atomicAdd(&count, local);
    __syncthreads();
    const bool mask = k.top_k != 0 && k.top_k < count;
    // All threads must finish reading count before its first reset.
    __syncthreads();
    uint32_t threshold = 0;
    if (mask) {
        for (uint32_t bit = 0x80000000u; bit != 0; bit >>= 1) {
            const uint32_t trial = threshold | bit;
            if (threadIdx.x == 0) count = 0;
            __syncthreads();
            local = 0;
            for (uint64_t v = threadIdx.x; v < width; v += blockDim.x) {
                if (row[v] != -INFINITY)
                    local += tf_ds_score_key(row[v] + __half2float(bias[v])) >= trial;
            }
            atomicAdd(&count, local);
            __syncthreads();
            if (count >= k.top_k) threshold = trial;
            // All threads must consume this count before the next bit resets it.
            __syncthreads();
        }
    }
    for (uint64_t v = threadIdx.x; v < width; v += blockDim.x) {
        if (row[v] == -INFINITY) continue;
        if (mask && tf_ds_score_key(row[v] + __half2float(bias[v])) < threshold) {
            row[v] = -INFINITY;
        } else {
            uint64_t x = tf_ds_mix64(k.seed + 0x9E3779B97F4A7C15ull);
            x = tf_ds_mix64(x ^ (pos * 0xD1B54A32D192ED03ull));
            x = tf_ds_mix64(x ^ (col0 + v));
            const double u = double(x >> 11) * 0x1p-53 + 0x1p-54;
            row[v] += float(t * -log(-log(u)));
        }
    }
}
