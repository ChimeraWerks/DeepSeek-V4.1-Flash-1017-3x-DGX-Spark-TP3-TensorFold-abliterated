//! The served target sampler of a decode or verify row (tensorfold.cuda.sampling.sample_rows with
//! exact_sampling.choose_rows): temperature 0 takes the row's argmax (the first of equal values). Otherwise, with top_k
//! > 0, the row's top_k + MARGIN largest logits (torch.topk's set; of equal values the lower ids) ordered by value, then
//! id (np.lexsort), the first top_k of them; each candidate's score is its logit over the temperature plus a Gumbel draw
//! keyed by (seed, position, token id) through a splitmix64 chain, so a verify row draws as the serial run at its
//! position. top_p sets the scores past the smallest prefix whose probability mass reaches top_p to -inf (the scaled
//! logits' exp over their sum: numpy's pairwise sum, then cumulative sums in order); min_p those under ln(min_p) of the
//! top. The token is the first best score. float64 as numpy computes it on the served image (numpy 2.1: its exp and log
//! are libm's, a row's sum of 20 its 8-accumulator pairwise sum; zrec np_libm.py).
const std = @import("std");

/// exact_sampling.Sampling (the served lane's defaults: temperature 1.0, top_k 20, top_p 0.95, min_p 0).
pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: usize = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,
};

/// exact_sampling.MARGIN: candidates beyond top_k read from the GPU.
pub const margin = 8;
/// The candidates a row takes at most here (top_k + MARGIN).
pub const max_candidates = 64;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

fn mix(x0: u64) u64 {
    var x = x0;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// exact_sampling.uniform: a (0, 1) double from a splitmix64 hash of (seed, position, token id).
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

/// numpy's pairwise sum of a contiguous float64 row (umath pairwise_sum, under its 128-element block).
pub fn pairwiseSum(a: []const f64) f64 {
    const n = a.len;
    if (n < 8) {
        var res: f64 = 0.0;
        for (a) |v| res += v;
        return res;
    }
    if (n > 128) {
        const n2 = (n / 2) / 8 * 8;
        return pairwiseSum(a[0..n2]) + pairwiseSum(a[n2..]);
    }
    var r: [8]f64 = a[0..8].*;
    var i: usize = 8;
    while (i < n - n % 8) : (i += 8) {
        for (0..8) |j| r[j] += a[i + j];
    }
    var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    while (i < n) : (i += 1) res += a[i];
    return res;
}

/// a before b in np.lexsort((ids, -values))'s order: the larger value, then the lower id.
fn before(va: f32, ia: i64, vb: f32, ib: i64) bool {
    return va > vb or (va == vb and ia < ib);
}

/// The row's first `count` (value, id) in that order: torch.topk's set (of equal values the lower ids) as np.lexsort
/// orders it.
pub fn topCandidates(row: []const f32, count: usize, values: []f32, ids: []i64) usize {
    const k = @min(count, row.len);
    var n: usize = 0;
    for (row, 0..) |v, i| {
        const id: i64 = @intCast(i);
        if (n == k and !before(v, id, values[k - 1], ids[k - 1])) continue;
        // insert in order, dropping the last when full
        var j: usize = if (n < k) n else k - 1;
        while (j > 0 and before(v, id, values[j - 1], ids[j - 1])) : (j -= 1) {
            values[j] = values[j - 1];
            ids[j] = ids[j - 1];
        }
        values[j] = v;
        ids[j] = id;
        if (n < k) n += 1;
    }
    return n;
}

/// The first index of the largest value (np.argmax / torch.argmax on a row without NaN).
pub fn argmax(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |v, i| {
        if (v > row[best]) best = i;
    }
    return best;
}

/// sample_rows for one row of fp32 logits at absolute position `position`: the token.
fn sampleBoundedRow(row: []const f32, position: u64, s: Sampling) !i64 {
    if (s.temperature <= 0) return @intCast(argmax(row));
    const count = @min(row.len, s.top_k + margin);
    if (count > max_candidates) return error.TooManyCandidates;
    var values: [max_candidates]f32 = undefined;
    var ids: [max_candidates]i64 = undefined;
    const width = topCandidates(row, count, &values, &ids);
    return sampleCandidates(values[0..width], ids[0..width], position, s);
}

/// Draw from an already sorted bounded top-k plus margin, with the CPU sampler's exact arithmetic.
pub fn sampleCandidates(values: []const f32, ids: []const i64, position: u64, s: Sampling) !i64 {
    if (values.len == 0 or values.len != ids.len or values.len > max_candidates or s.top_k == 0) return error.BadSamplingCandidates;
    const width = values.len;
    // choose_rows on the candidates (already in lexsort's order)
    const k = @max(1, @min(s.top_k, width));
    const t = @max(s.temperature, 1e-6);
    var scaled: [max_candidates]f64 = undefined;
    var score: [max_candidates]f64 = undefined;
    for (0..k) |j| {
        scaled[j] = @as(f64, values[j]) / t;
        const u = uniform(s.seed, position, @bitCast(ids[j]));
        score[j] = scaled[j] - log(-log(u));
    }
    if (0.0 < s.top_p and s.top_p < 1.0) {
        var mx = scaled[0];
        for (scaled[1..k]) |v| mx = @max(mx, v);
        var probs: [max_candidates]f64 = undefined;
        for (0..k) |j| probs[j] = exp(scaled[j] - mx);
        const sum = pairwiseSum(probs[0..k]);
        for (0..k) |j| probs[j] /= sum;
        var acc: f64 = 0.0;
        var below: usize = 0;
        for (0..k) |j| {
            acc += probs[j];
            if (acc < s.top_p) below += 1;
        }
        const keep = below + 1;
        for (keep..k) |j| score[j] = -std.math.inf(f64);
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + log(s.min_p);
        for (0..k) |j| {
            if (scaled[j] < floor) score[j] = -std.math.inf(f64);
        }
    }
    var best: usize = 0;
    for (0..k) |j| {
        if (score[j] > score[best]) best = j;
    }
    return ids[best];
}

test "splitmix64 draws as exact_sampling.uniform makes them" {
    // (seed, position, id) -> the bits of exact_sampling.uniform's double
    try std.testing.expectEqual(@as(u64, 0x3fe8cd8867d6a69c), @as(u64, @bitCast(uniform(101, 2229, 2581))));
    try std.testing.expectEqual(@as(u64, 0x3fd08f785f86da1b), @as(u64, @bitCast(uniform(404, 59, 129279))));
    try std.testing.expectEqual(@as(u64, 0x3fc9ff45ea7ce2be), @as(u64, @bitCast(uniform(0, 0, 0))));
}

test "numpy's pairwise sum" {
    const a = [_]f64{ 1e16, 1.0, -1e16, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 0.5, 0.25 };
    // r = a[0..8] (no whole second block of 8); ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), then the rest in order
    var want = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
    for (a[8..]) |v| want += v;
    try std.testing.expectEqual(want, pairwiseSum(&a));
}

test "top candidates by value, then id" {
    const row = [_]f32{ 1, 5, 3, 5, 2, 5, 0 };
    var v: [4]f32 = undefined;
    var ids: [4]i64 = undefined;
    try std.testing.expectEqual(@as(usize, 4), topCandidates(&row, 4, &v, &ids));
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 5, 2 }, &ids);
    try std.testing.expectEqualSlices(f32, &.{ 5, 5, 5, 3 }, &v);
}

/// Reused by the lane worker: no allocation per verification row after growth.
const Candidate = struct { value: f32, id: i64 };
pub const Workspace = struct {
    penalty_cache: ?PenaltyCache = null,
    order: std.ArrayList(Candidate) = .empty,
    probs: std.ArrayList(f64) = .empty,

    pub fn deinit(w: *Workspace, a: std.mem.Allocator) void {
        w.order.deinit(a);
        w.probs.deinit(a);
        w.* = .{};
    }
};

fn candidateBefore(_: void, a: Candidate, b: Candidate) bool {
    return before(a.value, a.id, b.value, b.id);
}

fn validLogit(v: f32) bool {
    return std.math.isFinite(v) or v == -std.math.inf(f32);
}

/// Certify the same nucleus prefix as the reference full sort, without ordering
/// the low-logit tail. A failed certificate always falls back to that reference.
/// The denominator is order-sensitive in f64: never use an approximate cutoff.
fn prefilteredNucleus(w: *Workspace, a: std.mem.Allocator, t: f64, p: f64) !?usize {
    const order = w.order.items;
    // Keep the roundoff proof deliberately bounded; the served vocab is 129408.
    if (order.len < 128 or order.len > (1 << 24)) return null;
    var mx: f32 = -std.math.inf(f32);
    for (order) |c| mx = @max(mx, c.value);
    if (!std.math.isFinite(mx)) return error.NoSamplingCandidates;
    const top = @as(f64, mx) / t;
    const cutoff = top - 12.0;
    var count: usize = 0;
    for (order) |c| {
        if (std.math.isFinite(c.value) and @as(f64, c.value) / t >= cutoff) count += 1;
    }
    // Diffuse rows use the original path without allocating a second workspace.
    if (count == 0 or count >= order.len / 2) return null;
    try w.probs.resize(a, order.len);
    for (w.probs.items, order) |*weight, c| weight.* = exp(@as(f64, c.value) / t - top);
    const sum = pairwiseSum(w.probs.items);
    // Both permutations sum the SAME stored nonnegative f64 weights, including
    // a maximum weight of 1. With n <= 2^24, 8*n*epsilon bounds the two sums'
    // relative rounding difference (also dwarfs subnormal-addition error).
    // This is intentionally looser than pairwiseSum's logarithmic-depth bound.
    const err = sum * (8.0 * @as(f64, @floatFromInt(order.len)) * std.math.floatEps(f64));
    const lo = sum - err;
    const hi = sum + err;
    var n: usize = 0;
    for (0..order.len) |i| {
        if (std.math.isFinite(order[i].value) and @as(f64, order[i].value) / t >= cutoff) {
            std.mem.swap(Candidate, &order[n], &order[i]);
            n += 1;
        }
    }
    std.sort.pdq(Candidate, order[0..n], {}, candidateBefore);
    // Rounded division and positive addition are monotone. These two actual
    // cumulative sequences enclose the reference sequence at every position.
    var lower: f64 = 0;
    var upper: f64 = 0;
    for (order[0..n], 0..) |c, i| {
        const weight = exp(@as(f64, c.value) / t - top);
        lower += weight / hi;
        upper += weight / lo;
        if (upper < p) continue;
        if (lower >= p) return i + 1;
        return null; // rounding boundary: preserve the old arithmetic exactly
    }
    return null; // insufficient prefix mass: retain the full vocabulary
}

/// A full-vocabulary draw. No implicit top-k limit; padding at -inf cannot win.
/// The unconstrained path needs no sort/scratch. Ties still use lexsort order.
fn sampleFull(w: *Workspace, a: std.mem.Allocator, row: []const f32, position: u64, s: Sampling) !i64 {
    const t = @max(s.temperature, 1e-6);
    if (!std.math.isFinite(t)) return error.InvalidSamplingTemperature;
    const nucleus = 0.0 < s.top_p and s.top_p < 1.0;
    if ((s.top_k == 0 or s.top_k >= row.len) and !nucleus) {
        var mx: f32 = -std.math.inf(f32);
        for (row) |v| {
            if (!validLogit(v)) return error.InvalidSamplingLogits;
            mx = @max(mx, v);
        }
        if (!std.math.isFinite(mx)) return error.NoSamplingCandidates;
        const floor = if (s.min_p > 0) @as(f64, mx) / t + log(s.min_p) else -std.math.inf(f64);
        const cached = if (w.penalty_cache) |cache| cache.lookup(cache.ptr, s.seed, position, row.len) else null;
        var best: ?usize = null;
        var best_score: f64 = -std.math.inf(f64);
        for (row, 0..) |v, id| {
            if (!std.math.isFinite(v)) continue;
            const scaled = @as(f64, v) / t;
            if (scaled < floor) continue;
            const score = scaled - (if (cached) |values| values[id] else keyedPenalty(s.seed, position, id));
            if (best == null or score > best_score or
                (score == best_score and before(v, @intCast(id), row[best.?], @intCast(best.?))))
            {
                best = id;
                best_score = score;
            }
        }
        return @intCast(best orelse return error.NoSamplingCandidates);
    }
    try w.order.resize(a, row.len);
    for (row, w.order.items, 0..) |v, *c, id| {
        if (!validLogit(v)) return error.InvalidSamplingLogits;
        c.* = .{ .value = v, .id = @intCast(id) };
    }
    const k = if (s.top_k == 0) row.len else @min(s.top_k, row.len);
    const certified = if (nucleus and k == row.len) try prefilteredNucleus(w, a, t, s.top_p) else null;
    if (certified == null) std.sort.pdq(Candidate, w.order.items, {}, candidateBefore);
    if (!std.math.isFinite(w.order.items[0].value)) return error.NoSamplingCandidates;
    const top = @as(f64, w.order.items[0].value) / t;
    var keep = certified orelse k;
    if (nucleus and certified == null) {
        try w.probs.resize(a, k);
        for (w.probs.items, w.order.items[0..k]) |*p, c| p.* = exp(@as(f64, c.value) / t - top);
        const sum = pairwiseSum(w.probs.items);
        var acc: f64 = 0;
        for (w.probs.items, 0..) |p, i| {
            acc += p / sum;
            if (acc >= s.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    const floor = if (s.min_p > 0) top + log(s.min_p) else -std.math.inf(f64);
    var best: ?i64 = null;
    var best_score: f64 = -std.math.inf(f64);
    for (w.order.items[0..keep]) |c| {
        if (!std.math.isFinite(c.value)) continue;
        const scaled = @as(f64, c.value) / t;
        if (scaled < floor) continue;
        const score = scaled - log(-log(uniform(s.seed, position, @bitCast(c.id))));
        if (best == null or score > best_score) {
            best = c.id;
            best_score = score;
        }
    }
    return best orelse error.NoSamplingCandidates;
}

/// All requested sampling modes; the original small-top-k arithmetic is retained.
pub fn sampleRowWithWorkspace(w: *Workspace, a: std.mem.Allocator, row: []const f32, position: u64, s: Sampling) !i64 {
    if (row.len == 0) return error.EmptySamplingRow;
    if (s.temperature <= 0) return @intCast(argmax(row));
    if (s.top_k > 0 and @min(row.len, s.top_k + margin) <= max_candidates)
        return sampleBoundedRow(row, position, s);
    return sampleFull(w, a, row, position, s);
}

/// Standalone gates keep their existing API. Serving uses one retained workspace.
pub fn sampleRow(row: []const f32, position: u64, s: Sampling) !i64 {
    var w: Workspace = .{};
    defer w.deinit(std.heap.page_allocator);
    return sampleRowWithWorkspace(&w, std.heap.page_allocator, row, position, s);
}

test "full vocabulary is not clipped and rejects masked padding" {
    var row: [129408]f32 = @splat(-std.math.inf(f32));
    row[129279] = 10;
    try std.testing.expectEqual(@as(i64, 129279), try sampleRow(&row, 401000, .{ .seed = 2026, .temperature = 0.6, .top_k = 0, .top_p = 1 }));
    try std.testing.expectEqual(@as(i64, 129279), try sampleRow(&row, 401000, .{ .seed = 1708, .temperature = 0.6, .top_k = 0, .top_p = 0.95 }));
}

test "full-vocabulary nucleus and large top-k share exact position-keyed draws" {
    const a = std.testing.allocator;
    var w: Workspace = .{};
    defer w.deinit(a);
    var row: [1024]f32 = undefined;
    for (&row, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) / 7;
    for ([_]Sampling{
        .{ .seed = 2026, .temperature = 0.6, .top_k = 0, .top_p = 0.9 },
        .{ .seed = 1708, .temperature = 0.6, .top_k = 0, .top_p = 1 },
        .{ .seed = 42, .temperature = 0.8, .top_k = 1000, .top_p = 0.95, .min_p = 0.1 },
    }) |s| {
        var serial: [16]i64 = undefined;
        for (&serial, 0..) |*v, j| v.* = try sampleRowWithWorkspace(&w, a, &row, 131000 + j, s);
        const capacity = w.order.capacity;
        for (0..4) |batch| for (0..4) |offset| {
            const j = batch * 4 + offset;
            try std.testing.expectEqual(serial[j], try sampleRowWithWorkspace(&w, a, &row, 131000 + j, s));
        };
        try std.testing.expectEqual(capacity, w.order.capacity);
    }
}

test "full sampler validates empty invalid and all-masked rows" {
    try std.testing.expectError(error.EmptySamplingRow, sampleRow(&.{}, 0, .{ .seed = 1, .top_k = 0 }));
    try std.testing.expectError(error.InvalidSamplingLogits, sampleRow(&.{std.math.nan(f32)}, 0, .{ .seed = 1, .top_k = 0 }));
    try std.testing.expectError(error.NoSamplingCandidates, sampleRow(&.{-std.math.inf(f32)}, 0, .{ .seed = 1, .top_k = 0 }));
}

/// Sample only the reachable target path of an explicitly eligible chain.
/// Skipped rows are a sentinel, never a valid token. Forced windows use false.
/// All full-sampler logit validation still runs, including unreachable rows.
pub fn sampleWindow(w: *Workspace, a: std.mem.Allocator, logits: []const f32, vocab: usize, positions: []const u64, drafts: []const u32, s: ?Sampling, prefix_only: bool, out: []u32) !usize {
    if (vocab == 0 or out.len == 0 or positions.len != out.len or drafts.len + 1 != out.len or logits.len != vocab * out.len) return error.BadSamplingWindow;
    var reached: usize = 0;
    var reachable = true;
    for (out, positions, 0..) |*pick, position, r| {
        const row = logits[r * vocab ..][0..vocab];
        if (!reachable) {
            if (s) |sm| if (sm.temperature > 0 and !(sm.top_k > 0 and @min(vocab, sm.top_k + margin) <= max_candidates)) {
                var any = false;
                for (row) |v| {
                    if (!validLogit(v)) return error.InvalidSamplingLogits;
                    any = any or std.math.isFinite(v);
                }
                if (!any) return error.NoSamplingCandidates;
            };
            pick.* = std.math.maxInt(u32);
            continue;
        }
        pick.* = if (s) |sm| @intCast(try sampleRowWithWorkspace(w, a, row, position, sm)) else @intCast(argmax(row));
        reached += 1;
        if (prefix_only and r < drafts.len and pick.* != drafts[r]) reachable = false;
    }
    return reached;
}

/// The same reachable-row rule with GPU-sorted candidates for bounded sampling.
/// Greedy rows use the first candidate, equal to argmax on a NaN-free row.
pub fn sampleCandidateWindow(values: []const f32, ids: []const i64, count: usize, vocab: usize, positions: []const u64, drafts: []const u32, s: ?Sampling, prefix_only: bool, out: []u32) !usize {
    if (count == 0 or count > max_candidates or vocab == 0 or out.len == 0 or positions.len != out.len or drafts.len + 1 != out.len or values.len != count * out.len or ids.len != values.len) return error.BadSamplingWindow;
    var reached: usize = 0;
    var reachable = true;
    for (out, positions, 0..) |*pick, position, r| {
        if (!reachable) {
            pick.* = std.math.maxInt(u32);
            continue;
        }
        const row_ids = ids[r * count ..][0..count];
        if (s) |sm| {
            if (sm.temperature > 0) {
                if (sm.top_k == 0 or sm.top_k > max_candidates - margin) return error.BadSamplingCandidates;
                const width = @min(vocab, sm.top_k + margin);
                if (width > count) return error.BadSamplingCandidates;
                pick.* = @intCast(try sampleCandidates(values[r * count ..][0..width], row_ids[0..width], position, sm));
            } else pick.* = @intCast(row_ids[0]);
        } else pick.* = @intCast(row_ids[0]);
        reached += 1;
        if (prefix_only and r < drafts.len and pick.* != drafts[r]) reachable = false;
    }
    return reached;
}

test "prefix sampling preserves every reachable target across modes seeds and mismatch depths" {
    const a = std.testing.allocator;
    var w: Workspace = .{};
    defer w.deinit(a);
    var logits: [6 * 256]f32 = undefined;
    for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 17) % 71)) / 8;
    const positions = [_]u64{ 1023, 1024, 1025, 1026, 1027, 1028 };
    for ([_]Sampling{
        .{ .seed = 42, .temperature = 0.6, .top_k = 0, .top_p = 1 },
        .{ .seed = 0, .temperature = 0.8, .top_k = 0, .top_p = 0.9 },
        .{ .seed = 99, .temperature = 1, .top_k = 20, .top_p = 0.95, .min_p = 0.1 },
        .{ .seed = 77, .temperature = 0 },
    }) |sm| {
        var full: [6]u32 = undefined;
        var lazy: [6]u32 = undefined;
        var proposals: [5]u32 = @splat(0);
        _ = try sampleWindow(&w, a, &logits, 256, &positions, &proposals, sm, false, &full);
        for (0..6) |mismatch| {
            @memcpy(&proposals, full[0..5]);
            if (mismatch < 5) proposals[mismatch] = (full[mismatch] + 1) % 256;
            const n = try sampleWindow(&w, a, &logits, 256, &positions, &proposals, sm, true, &lazy);
            try std.testing.expectEqual(@min(mismatch + 1, 6), n);
            try std.testing.expectEqualSlices(u32, full[0..n], lazy[0..n]);
            for (lazy[n..]) |v| try std.testing.expectEqual(std.math.maxInt(u32), v);
            // The acceptance chain and final correction can only read [0,n).
            var accepted: usize = 0;
            while (accepted < proposals.len and proposals[accepted] == lazy[accepted]) : (accepted += 1) {}
            try std.testing.expectEqual(mismatch, accepted);
        }
        // Forced output needs the last row even after a mismatch.
        @memset(&proposals, std.math.maxInt(u32));
        try std.testing.expectEqual(@as(usize, 6), try sampleWindow(&w, a, &logits, 256, &positions, &proposals, sm, false, &lazy));
        try std.testing.expectEqualSlices(u32, &full, &lazy);
    }
}

test "bounded candidate windows keep CPU draws and prefix acceptance" {
    const vocab = 128;
    const count = max_candidates;
    var logits: [4 * vocab]f32 = undefined;
    var values: [4 * count]f32 = undefined;
    var ids: [4 * count]i64 = undefined;
    for (&logits, 0..) |*v, i| v.* = if (i % vocab >= 100) -std.math.inf(f32) else @as(f32, @floatFromInt((i * 19 + i / vocab) % 37)) / 4;
    logits[3] = -0.0;
    logits[4] = 0.0;
    for (0..4) |r| _ = topCandidates(logits[r * vocab ..][0..vocab], count, values[r * count ..][0..count], ids[r * count ..][0..count]);
    const positions = [_]u64{ 1100, 1101, 1102, 1103 };
    var drafts: [3]u32 = undefined;
    var expected: [4]u32 = undefined;
    var actual: [4]u32 = undefined;
    var w: Workspace = .{};
    defer w.deinit(std.testing.allocator);
    for ([_]?Sampling{
        .{ .seed = 123, .temperature = 0.7, .top_k = 1, .top_p = 0.95 },
        .{ .seed = 123, .temperature = 0.7, .top_k = 20, .top_p = 0.8, .min_p = 0.1 },
        .{ .seed = 888, .temperature = 1.2, .top_k = 56, .top_p = 1, .min_p = 0.2 },
        .{ .seed = 888, .temperature = 0 },
        null,
    }) |sm| {
        _ = try sampleWindow(&w, std.testing.allocator, &logits, vocab, &positions, &.{ 0, 0, 0 }, sm, false, &expected);
        _ = try sampleCandidateWindow(&values, &ids, count, vocab, &positions, &.{ 0, 0, 0 }, sm, false, &actual);
        try std.testing.expectEqualSlices(u32, &expected, &actual);
        for (0..4) |mismatch| {
            @memcpy(&drafts, expected[0..3]);
            if (mismatch < 3) drafts[mismatch] +%= 1;
            const n = try sampleWindow(&w, std.testing.allocator, &logits, vocab, &positions, &drafts, sm, true, &expected);
            try std.testing.expectEqual(n, try sampleCandidateWindow(&values, &ids, count, vocab, &positions, &drafts, sm, true, &actual));
            try std.testing.expectEqualSlices(u32, &expected, &actual);
        }
    }
}

test "unreachable full-sampler errors remain visible" {
    const sm: Sampling = .{ .seed = 1, .top_k = 0, .top_p = 1 };
    var w: Workspace = .{};
    defer w.deinit(std.testing.allocator);
    var out: [2]u32 = undefined;
    try std.testing.expectError(error.InvalidSamplingLogits, sampleWindow(&w, std.testing.allocator, &.{ 1, 0, std.math.nan(f32), 0 }, 2, &.{ 0, 1 }, &.{99}, sm, true, &out));
    try std.testing.expectError(error.NoSamplingCandidates, sampleWindow(&w, std.testing.allocator, &.{ 1, 0, -std.math.inf(f32), -std.math.inf(f32) }, 2, &.{ 0, 1 }, &.{99}, sm, true, &out));
}

/// A nonblocking, identity-checked view. Cache misses retain original math.
pub const PenaltyCache = struct {
    ptr: *anyopaque,
    lookup: *const fn (*anyopaque, u64, u64, usize) ?[]const f64,
};

pub fn keyedPenalty(seed: u64, position: u64, id: u64) f64 {
    return log(-log(uniform(seed, position, id)));
}

test "nucleus certificate accepts a narrow head and rejects ambiguous cutoffs" {
    var w: Workspace = .{};
    const a = std.testing.allocator;
    defer w.deinit(a);
    try w.order.resize(a, 1024);
    for (w.order.items, 0..) |*c, id| c.* = .{ .value = if (id < 8) 0 else -30, .id = @intCast(id) };
    try std.testing.expectEqual(@as(?usize, 8), try prefilteredNucleus(&w, a, 1, 0.95));
    // At exactly 1/8 with a zero tail the interval straddles the cutoff.
    for (w.order.items[8..]) |*c| c.value = -std.math.inf(f32);
    try std.testing.expectEqual(@as(?usize, null), try prefilteredNucleus(&w, a, 1, 0.125));
    for (w.order.items) |*c| c.value = 0;
    try std.testing.expectEqual(@as(?usize, null), try prefilteredNucleus(&w, a, 1, 0.95));
    // A small retained head with too much aggregate tail mass also falls back.
    for (w.order.items, 0..) |*c, id| c.* = .{ .value = if (id == 0) 0 else -12.01, .id = @intCast(id) };
    try std.testing.expectEqual(@as(?usize, null), try prefilteredNucleus(&w, a, 1, 0.9999));
}
