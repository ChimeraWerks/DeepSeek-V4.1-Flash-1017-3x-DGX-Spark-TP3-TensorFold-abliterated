// Standalone GPU/CPU candidate-order check. Compile with nvcc -arch=sm_121 on GB10,
// or sm_120 locally; run the executable through gpu-lease on the workstation.
#include "../zig/kernels/cuda/dsv41_sample_cand.cu"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static void gpu(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}

static bool before(float a, int ia, float b, int ib) {
    return a > b || (a == b && ia < ib);
}

static unsigned bits(float v) {
    unsigned result;
    std::memcpy(&result, &v, sizeof result);
    return result;
}

static void check(const char* name, const std::vector<float>& rows, int ld, int n, int count,
                  const std::vector<int>& expected_flags, int& checks, int& failures) {
    const int r = (int)(rows.size() / ld);
    const int first_failure = failures;
    float *dev_rows = nullptr, *dev_vals = nullptr;
    long long *dev_ids = nullptr;
    int *dev_flags = nullptr;
    gpu(cudaMalloc(&dev_rows, rows.size() * sizeof(float)));
    gpu(cudaMalloc(&dev_vals, r * count * sizeof(float)));
    gpu(cudaMalloc(&dev_ids, r * count * sizeof(long long)));
    gpu(cudaMalloc(&dev_flags, r * sizeof(int)));
    gpu(cudaMemcpy(dev_rows, rows.data(), rows.size() * sizeof(float), cudaMemcpyHostToDevice));
    tf_ds_top_cand_rows_kernel<<<r, 1024>>>(dev_rows, ld, n, count, dev_vals, dev_ids, dev_flags);
    gpu(cudaGetLastError());
    gpu(cudaDeviceSynchronize());
    std::vector<float> values(r * count);
    std::vector<long long> ids(r * count);
    std::vector<int> flags(r);
    gpu(cudaMemcpy(values.data(), dev_vals, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpu(cudaMemcpy(ids.data(), dev_ids, ids.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    gpu(cudaMemcpy(flags.data(), dev_flags, flags.size() * sizeof(int), cudaMemcpyDeviceToHost));
    gpu(cudaFree(dev_rows));
    gpu(cudaFree(dev_vals));
    gpu(cudaFree(dev_ids));
    gpu(cudaFree(dev_flags));
    for (int row = 0; row < r; row++) {
        checks++;
        bool ok = flags[row] == expected_flags[row];
        if (ok && flags[row] == 0) {
            // sampling.topCandidates: ordered insertion, larger value then lower id.
            // The stride's padding is masked to -inf by roundLogits; its IDs cannot
            // enter the first 64 while n >= 64, even when all logical logits are -inf.
            std::vector<float> ref(count);
            std::vector<int> ref_id(count);
            int found = 0;
            for (int i = 0; i < n; i++) {
                float v = rows[row * ld + i];
                if (found == count && !before(v, i, ref[found - 1], ref_id[found - 1])) continue;
                int j = found < count ? found : count - 1;
                while (j > 0 && before(v, i, ref[j - 1], ref_id[j - 1])) {
                    ref[j] = ref[j - 1];
                    ref_id[j] = ref_id[j - 1];
                    j--;
                }
                ref[j] = v;
                ref_id[j] = i;
                if (found < count) found++;
            }
            for (int j = 0; j < found; j++) {
                if (ids[row * count + j] != ref_id[j] || bits(values[row * count + j]) != bits(ref[j])) {
                    ok = false;
                    std::fprintf(stderr, "mismatch %s row=%d count=%d j=%d: got (%lld,%08x) expected (%d,%08x)\n",
                                 name, row, count, j, ids[row * count + j], bits(values[row * count + j]), ref_id[j], bits(ref[j]));
                    break;
                }
            }
        }
        if (!ok) {
            failures++;
            std::fprintf(stderr, "FAIL %s row=%d count=%d flag=%d expected=%d\n",
                         name, row, count, flags[row], expected_flags[row]);
        }
    }
    std::printf("%s rows=%d count=%d: %d/%d passed\n", name, r, count, r - (failures - first_failure), r);
}

int main() {
    constexpr int n = 129280, ld = 129408;
    const float inf = std::numeric_limits<float>::infinity();
    std::mt19937 gen(7);
    std::normal_distribution<float> normal(0.0f, 4.0f);
    int checks = 0, failures = 0;
    std::vector<float> rows(6 * ld, -inf);
    for (int r = 0; r < 6; r++) for (int j = 0; j < n; j++) rows[r * ld + j] = normal(gen);
    for (int j = 0; j < 40; j++) rows[ld + j] = 20.0f;
    for (int j = 0; j < n; j++) rows[2 * ld + j] = normal(gen) - 10.0f;
    for (int j = 0; j < 32; j++) rows[2 * ld + j] = j & 1 ? -0.0f : 0.0f;
    rows[3 * ld + 5] = inf;
    rows[3 * ld + 9] = inf;
    for (int j = 0; j < 100; j++) rows[3 * ld + 100 + j] = -inf;
    for (int j = 0; j < n; j++) rows[4 * ld + j] = std::round(rows[4 * ld + j] * 10.f) / 10.f;
    for (int j = 0; j < n; j++) rows[5 * ld + j] = (float)(__half2float(__float2half_rn(rows[5 * ld + j])));
    for (int count : {1, 8, 28, 64}) check("representative", rows, ld, n, count, std::vector<int>(6, 0), checks, failures);
    std::vector<float> pad(ld, -inf);
    for (int j = 0; j < n; j++) pad[j] = normal(gen);
    for (int j = n; j < ld; j++) pad[j] = inf; // masked padding must not become a candidate
    check("padded", pad, ld, n, 64, {0}, checks, failures);
    std::vector<float> one(ld, -inf);
    one[3] = 1.0f;
    check("one finite", one, ld, n, 1, {0}, checks, failures);
    for (int count : {8, 28, 64}) check("ties overflow", one, ld, n, count, {2}, checks, failures);
    std::vector<float> small(2000, -inf);
    for (int j = 0; j < 10; j++) small[j] = normal(gen);
    check("ties within buffer", small, 2000, 2000, 28, {0}, checks, failures);
    std::vector<float> short_row(10, -inf);
    for (float& v : short_row) v = normal(gen);
    check("short row", short_row, 10, 10, 28, {0}, checks, failures);
    std::vector<float> nan_rows(2 * ld, -inf);
    for (int j = 0; j < n; j++) nan_rows[j] = nan_rows[ld + j] = normal(gen);
    nan_rows[77] = std::numeric_limits<float>::quiet_NaN();
    check("NaN neighbour", nan_rows, ld, n, 28, {1, 0}, checks, failures);
    for (int trial = 0; trial < 8; trial++) {
        std::vector<float> batch(48 * ld, -inf);
        for (int r = 0; r < 48; r++) for (int j = 0; j < n; j++) batch[r * ld + j] = normal(gen);
        check("random 48 rows", batch, ld, n, 28, std::vector<int>(48, 0), checks, failures);
    }
    std::printf("candidate checks: %d passed, %d failed\n", checks - failures, failures);
    return failures ? 1 : 0;
}
