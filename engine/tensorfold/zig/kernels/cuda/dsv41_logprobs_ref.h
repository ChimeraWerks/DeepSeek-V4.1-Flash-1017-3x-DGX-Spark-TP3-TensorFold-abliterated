// Compact raw-model summaries. No sampling or grammar state enters the reduction.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tf_ds_logprob {
constexpr unsigned top_cap = 20, threads = 256;
struct Candidate { float value; uint32_t id; };
struct Summary {
    float maximum, mass, chosen;
    uint32_t owns, flags, count;
    Candidate top[top_cap];
};
static_assert(sizeof(Summary) == 184);
inline bool before(Candidate a, Candidate b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}
// Fixed lane partition and tree match the CUDA reduction, independent of batch size.
inline Summary local(const float* row, unsigned n, unsigned offset, unsigned chosen, unsigned k) {
    if (k > top_cap) throw std::invalid_argument("top_logprobs");
    Summary out{};
    out.maximum = -INFINITY;
    out.chosen = -INFINITY;
    out.owns = chosen >= offset && chosen - offset < n;
    if (out.owns) out.chosen = row[chosen - offset];
    float maxima[threads], sums[threads];
    for (unsigned lane = 0; lane < threads; ++lane) {
        float m = -INFINITY;
        for (unsigned i = lane; i < n; i += threads) {
            const float v = row[i];
            if (std::isnan(v) || v == INFINITY) out.flags |= 1;
            m = std::max(m, v);
        }
        maxima[lane] = m;
    }
    for (unsigned d = threads / 2; d; d /= 2)
        for (unsigned lane = 0; lane < d; ++lane) maxima[lane] = std::max(maxima[lane], maxima[lane + d]);
    out.maximum = maxima[0];
    if (out.flags) return out;
    for (unsigned lane = 0; lane < threads; ++lane) {
        float sum = 0, correction = 0;
        if (out.maximum != -INFINITY) for (unsigned i = lane; i < n; i += threads) {
            const float x = std::exp(row[i] - out.maximum);
            const float y = x - correction, t = sum + y;
            correction = (t - sum) - y;
            sum = t;
        }
        sums[lane] = sum;
    }
    for (unsigned d = threads / 2; d; d /= 2)
        for (unsigned lane = 0; lane < d; ++lane) sums[lane] += sums[lane + d];
    out.mass = sums[0];
    out.count = std::min(k, n);
    std::vector<Candidate> all;
    all.reserve(n);
    for (unsigned i = 0; i < n; ++i) all.push_back({row[i], offset + i});
    std::partial_sort(all.begin(), all.begin() + out.count, all.end(), before);
    for (unsigned i = 0; i < out.count; ++i) out.top[i] = all[i];
    return out;
}
struct Result { double chosen; std::vector<std::pair<uint32_t, double>> top; };
inline Result merge(const std::vector<Summary>& shards, unsigned k) {
    if (k > top_cap || shards.empty()) throw std::invalid_argument("top_logprobs");
    float maximum = -INFINITY, chosen = -INFINITY;
    unsigned owners = 0;
    for (const auto& s : shards) {
        if (s.flags || std::isnan(s.maximum) || s.maximum == INFINITY || !std::isfinite(s.mass) || s.mass < 0)
            throw std::invalid_argument("invalid logits");
        maximum = std::max(maximum, s.maximum);
        if (s.owns) { owners += s.owns; chosen = s.chosen; }
    }
    if (maximum == -INFINITY || owners != 1) throw std::invalid_argument("invalid row");
    double mass = 0;
    std::vector<Candidate> all;
    for (const auto& s : shards) {
        if (s.maximum != -INFINITY) mass += double(s.mass) * std::exp(double(s.maximum) - double(maximum));
        all.insert(all.end(), s.top, s.top + s.count);
    }
    if (!(mass > 0)) throw std::invalid_argument("invalid mass");
    const double correction = std::log(mass);
    Result result{(double(chosen) - double(maximum)) - correction, {}};
    std::sort(all.begin(), all.end(), before);
    for (unsigned i = 0; i < std::min(k, unsigned(all.size())); ++i)
        result.top.push_back({all[i].id, (double(all[i].value) - double(maximum)) - correction});
    return result;
}
}
