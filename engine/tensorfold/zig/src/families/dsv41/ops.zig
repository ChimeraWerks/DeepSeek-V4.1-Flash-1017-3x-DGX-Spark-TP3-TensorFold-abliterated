//! The torch operations of the served forward that only move or widen bytes, on TensorFold's own torch-op kernels
//! (zig/kernels/cuda/torch_ops: pointwise.cu, movement.cu; ours in dsv41_ops.cu) with their development launchers'
//! geometry: `.float()` of bf16 and fp16 (exact), `.to(bf16)` of fp32 (round to nearest even), `.contiguous()` and row
//! copies, row gathers and scatters by index (`t[idx]`, `t[idx] = rows`), and the integer selections of the indexer
//! (torch's top-k of int64 keys, kernels.topk_indices). Their bytes are the same as torch's by construction; the layer
//! gate checks them.
const std = @import("std");
const cuda = @import("cuda");

/// launch_shape.h's tf_launch_blocks: a block a `threads` elements, at most 65535.
fn blocks(count: usize, threads: usize) u32 {
    return @intCast(@min((count + threads - 1) / threads, 65535));
}

/// one stream's keyed kernel row: seed, temperature, first draft position and draft top_k; `reserved` (upload 0) is
/// where the select kernel leaves the row's cut for apply.
pub const KeyedRow = extern struct { seed: u64, temp: f64, pos: i64, top_k: u32 = 0, reserved: u32 = 0 };

/// tf_ds_l2_prefetch_kernel's argument: up to 16 device ranges (their first byte and length).
pub const Prefetch = extern struct { ptr: [16]u64 = @splat(0), bytes: [16]u64 = @splat(0), n: c_int = 0 };

pub const Ops = struct {
    pointwise: cuda.Module,
    movement: cuda.Module,
    family: cuda.Module,
    candidates: ?cuda.Module = null,
    reduced_head: ?cuda.Module = null,
    top_cand_rows: ?cuda.Function = null,
    to_f32: cuda.Function,
    f16_to_f32: cuda.Function,
    add2_bf16: cuda.Function,
    add3_bf16: cuda.Function,
    topk_i64: cuda.Function,
    topk_indices: cuda.Function,
    scatter: cuda.Function,
    cand_fast: cuda.Function,
    iota_vis: cuda.Function,
    l2_prefetch: cuda.Function,
    block_max: cuda.Function,
    pool_pick: cuda.Function,
    apply_pool: cuda.Function,
    to_bf16: cuda.Function,
    strided: cuda.Function,
    gather: cuda.Function,
    copy16: cuda.Function,
    argmax_rows: cuda.Function,
    greedy_head_local: ?cuda.Function = null,
    greedy_head_merge: ?cuda.Function = null,
    sample_head_local: ?cuda.Function = null,
    sample_head_merge: ?cuda.Function = null,
    gpu_clock: cuda.Function,
    penalties: ?cuda.Module = null,
    penalty_apply: ?cuda.Function = null,
    keyed: ?cuda.Module = null,
    keyed_noise: ?cuda.Function = null,
    keyed_select: ?cuda.Function = null,
    keyed_apply: ?cuda.Function = null,

    /// The three images (cuda.kernels.torch_pointwise, torch_movement and dsv41_ops, or the fatbins' bytes).
    pub fn load(d: *const cuda.Driver, pointwise: []const u8, movement: []const u8, family: []const u8) !Ops {
        var pw = try cuda.Module.load(d, pointwise);
        errdefer pw.unload();
        var mv = try cuda.Module.load(d, movement);
        errdefer mv.unload();
        var fm = try cuda.Module.load(d, family);
        errdefer fm.unload();
        return .{
            .pointwise = pw,
            .movement = mv,
            .family = fm,
            .to_f32 = try pw.function("tf_bf16_to_f32_kernel"),
            .f16_to_f32 = try fm.function("tf_ds_f16_to_f32_kernel"),
            .add2_bf16 = try fm.function("tf_ds_add2_bf16_kernel"),
            .add3_bf16 = try fm.function("tf_ds_add3_bf16_kernel"),
            .topk_i64 = try fm.function("tf_ds_topk_i64_kernel"),
            .topk_indices = try fm.function("tf_ds_topk_indices_kernel"),
            .scatter = try fm.function("tf_ds_scatter_rows_kernel"),
            .cand_fast = try fm.function("tf_ds_cand_fast_kernel"),
            .iota_vis = try fm.function("tf_ds_iota_vis_kernel"),
            .l2_prefetch = try fm.function("tf_ds_l2_prefetch_kernel"),
            .block_max = try fm.function("tf_ds_block_max_kernel"),
            .pool_pick = try fm.function("tf_ds_pool_pick_kernel"),
            .apply_pool = try fm.function("tf_ds_apply_pool_kernel"),
            .to_bf16 = try pw.function("tf_f32_to_bf16_kernel"),
            .strided = try mv.function("tf_strided_copy_kernel"),
            .gather = try mv.function("tf_gather_rows_kernel"),
            .copy16 = try fm.function("tf_ds_copy_rows16_kernel"),
            .argmax_rows = try fm.function("tf_ds_argmax_rows_kernel"),
            .gpu_clock = try fm.function("tf_ds_clock_kernel"),
        };
    }

    /// Called only for the enabled feature, never through the production family module.
    pub fn enableKeyed(o: *Ops, d: *const cuda.Driver, image: []const u8) !void {
        if (image.len == 0) return error.NoKeyedDraftKernel;
        var module = try cuda.Module.load(d, image);
        errdefer module.unload();
        const f = try module.function("tf_ds_keyed_noise_kernel");
        const select = try module.function("tf_ds_keyed_select_kernel");
        const apply = try module.function("tf_ds_keyed_apply_kernel");
        o.keyed = module;
        o.keyed_noise = f;
        o.keyed_select = select;
        o.keyed_apply = apply;
    }

    /// A separate image: existing production CUDA images remain byte-identical.
    pub fn enableGreedyHead(o: *Ops, d: *const cuda.Driver, image: []const u8) !void {
        if (image.len == 0) return error.NoGreedyHeadKernel;
        var module = try cuda.Module.load(d, image);
        errdefer module.unload();
        const local = try module.function("tf_ds_greedy_head_local_kernel");
        const merge = try module.function("tf_ds_greedy_head_merge_kernel");
        const sample_local = try module.function("tf_ds_sample_head_local_kernel");
        const sample_merge = try module.function("tf_ds_sample_head_merge_kernel");
        o.sample_head_local = sample_local;
        o.sample_head_merge = sample_merge;
        o.reduced_head = module;
        o.greedy_head_local = local;
        o.greedy_head_merge = merge;
    }

    pub fn greedyHeadLocal(o: *const Ops, s: cuda.Stream, logits: u64, ld: usize, vocab: usize, offset: usize, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, ld));
        a.add(@as(u32, @intCast(vocab)));
        a.add(@as(u32, @intCast(offset)));
        a.add(out);
        try go(o.greedy_head_local orelse return error.NoGreedyHeadKernel, s, rows, 1024, &a);
    }

    pub fn greedyHeadMerge(o: *const Ops, s: cuda.Stream, all: u64, rows: usize, world: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(all);
        a.add(@as(u32, @intCast(rows)));
        a.add(@as(u32, @intCast(world)));
        a.add(out);
        try go(o.greedy_head_merge orelse return error.NoGreedyHeadKernel, s, rows, 32, &a);
    }


    pub fn sampleHeadLocal(o: *const Ops, s: cuda.Stream, logits: u64, ld: usize, vocab: usize, offset: usize, rows: usize, count: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, ld));
        a.add(@as(u32, @intCast(vocab)));
        a.add(@as(u32, @intCast(offset)));
        a.add(@as(u32, @intCast(count)));
        a.add(out);
        try go(o.sample_head_local orelse return error.NoGreedyHeadKernel, s, rows, 1024, &a);
    }

    pub fn sampleHeadMerge(o: *const Ops, s: cuda.Stream, all: u64, rows: usize, world: usize, count: usize, vals: u64, ids: u64, flags: u64) !void {
        var a: cuda.Args = .{};
        a.add(all);
        a.add(@as(u32, @intCast(rows)));
        a.add(@as(u32, @intCast(world)));
        a.add(@as(u32, @intCast(count)));
        a.add(vals);
        a.add(ids);
        a.add(flags);
        try go(o.sample_head_merge orelse return error.NoGreedyHeadKernel, s, rows, 32, &a);
    }

    pub fn enableCandidates(o: *Ops, d: *const cuda.Driver, image: []const u8) !void {
        var module = try cuda.Module.load(d, image);
        errdefer module.unload();
        const f = try module.function("tf_ds_top_cand_rows_kernel");
        o.candidates = module;
        o.top_cand_rows = f;
    }

    pub fn enablePenalties(o: *Ops, d: *const cuda.Driver, image: []const u8) !void {
        if (image.len == 0) return error.NoPenaltyKernel;
        var module = try cuda.Module.load(d, image);
        errdefer module.unload();
        const f = try module.function("tf_ds_penalties_kernel");
        o.penalties = module;
        o.penalty_apply = f;
    }
    pub fn applyPenalties(o: *const Ops, s: cuda.Stream, src: u64, dst: u64, counts: u64, params: u64, width: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(src); a.add(dst); a.add(counts); a.add(params);
        a.add(@as(u32,@intCast(width))); a.add(@as(u32,@intCast(rows)));
        try go(o.penalty_apply orelse return error.NoPenaltyKernel, s, blocks(width*rows,256),256,&a);
    }
    pub fn unload(o: *Ops) void {
        const cleanup = cuda.context_health.cleanupMark();
        if (o.penalties) |*module| {
            module.unload();
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
        }
        if (o.keyed) |*module| {
            module.unload();
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
        }
        if (o.reduced_head) |*module| {
            module.unload();
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
        }
        if (o.candidates) |*module| {
            module.unload();
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
        }
        o.pointwise.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return;
        o.movement.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return;
        o.family.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return;
        if (!cuda.context_health.cleanupReleased(cleanup)) return;
        o.* = undefined;
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: usize, block: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = block } }, s, args);
    }

    /// `.float()` of `count` bf16 values (exact).
    pub fn toF32(o: *const Ops, s: cuda.Stream, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.to_f32, s, blocks(count, 256), 256, &a);
    }

    /// `.float()` of `count` fp16 values (exact).
    pub fn f16ToF32(o: *const Ops, s: cuda.Stream, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.f16_to_f32, s, blocks(count, 256), 256, &a);
    }

    /// Comm.sum of two ranks' fp32 partials then `.to(bf16)`: out = bf16(g0 + g1), one fp32 add in rank order.
    pub fn add2Bf16(o: *const Ops, s: cuda.Stream, g0: u64, g1: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(g0);
        a.add(g1);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.add2_bf16, s, blocks(count, 256), 256, &a);
    }

    /// Rank-major FP32 Engram partials; preserve TP2 arithmetic and extend in rank order for TP3.
    pub fn sumRanksBf16(o: *const Ops, s: cuda.Stream, gathered: u64, out: u64, count: usize, ranks: usize) !void {
        if (ranks == 2) return o.add2Bf16(s, gathered, gathered + count * 4, out, count);
        if (ranks != 3) return error.UnsupportedEngramRanks;
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(gathered);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.add3_bf16, s, blocks(count, 256), 256, &a);
    }

    /// keys.topk(k, sorted=False).values of unique int64 keys [rows, n] (row stride ks) into top [rows, k], the set in no
    /// particular order (_topk_finish sorts the indices).
    pub fn topkI64(o: *const Ops, s: cuda.Stream, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) !void {
        if (rows == 0) return;
        if (k > n) return error.BadTopK;
        var a: cuda.Args = .{};
        a.add(keys);
        a.add(@as(c_longlong, @intCast(ks)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        a.add(top);
        try cuda.launch.launch(o.topk_i64, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// kernels.topk_indices(score, k) with torch.where(top < vis, top, -1): each row's k highest fp32 scores' columns
    /// ascending (ties to the lower column), a column at or past vis[r] as -1. score [rows, n] (row stride ss), vis
    /// [rows] int64, out [rows, k] int64.
    pub fn topkIndices(o: *const Ops, s: cuda.Stream, score: u64, ss: usize, rows: usize, n: usize, k: usize, vis: u64, out: u64) !void {
        if (rows == 0) return;
        if (k > n or k > 2048) return error.BadTopK;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ss)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        a.add(vis);
        a.add(out);
        try cuda.launch.launch(o.topk_indices, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// dst row idx[i] = src row i (`t[idx] = rows`, distinct int64 indices): `rows` rows of `bytes` bytes, src rows
    /// `src_ld` bytes apart, dst rows `dst_ld`.
    pub fn scatterRows(o: *const Ops, s: cuda.Stream, src: u64, src_ld: usize, idx: u64, dst: u64, dst_ld: usize, bytes: usize, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_longlong, @intCast(src_ld)));
        a.add(idx);
        a.add(dst);
        a.add(@as(c_longlong, @intCast(dst_ld)));
        a.add(@as(c_longlong, @intCast(bytes)));
        a.add(@as(c_int, @intCast(rows)));
        try go(o.scatter, s, rows, 256, &a);
    }

    /// rounds.py _candidates_fast's mask when the pool takes every block: out [rows, nb] u8 (row stride os) = the block
    /// of bsize scores (row stride ss) has a maximum above -inf, or is the row's newest, (vis - 1) // bsize.
    pub fn candFast(o: *const Ops, s: cuda.Stream, score: u64, ss: usize, nb: usize, bsize: usize, vis: u64, out: u64, os: usize, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ss)));
        a.add(@as(c_int, @intCast(nb)));
        a.add(@as(c_int, @intCast(bsize)));
        a.add(vis);
        a.add(out);
        a.add(@as(c_longlong, @intCast(os)));
        try cuda.launch.launch(o.cand_fast, .{ .grid = .{ .x = blocks(nb, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Every 128-byte line of `pf`'s ranges touched into L2 (one block of 128 threads); writes nothing.
    pub fn l2Prefetch(o: *const Ops, s: cuda.Stream, pf: Prefetch) !void {
        if (pf.n == 0) return;
        // the served decode settings (exl3/prefetch.py): 3 us before the first wave, waves of 128 chunks of 8 KiB
        // at 230 GB/s (TF_EXL3_L2_DELAY_NS, TF_EXL3_L2_RATE_GBPS), one block
        const delay_ns: c_longlong = 3000;
        const wave_ns: c_longlong = @intFromFloat(128.0 * 8192.0 / 230.0);
        var a: cuda.Args = .{};
        a.add(pf);
        a.add(delay_ns);
        a.add(wave_ns);
        try cuda.launch.launch(o.l2_prefetch, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, s, &a);
    }

    /// The indexer's selection of every scanned key (its top-k takes all nb): out [rows, nb] int64 = j where j <
    /// vis[r], else -1 (vis [rows] int64), on the device.
    pub fn iotaVis(o: *const Ops, s: cuda.Stream, vis: u64, out: u64, nb: usize, rows: usize) !void {
        if (rows == 0 or nb == 0) return;
        var a: cuda.Args = .{};
        a.add(vis);
        a.add(out);
        a.add(@as(c_int, @intCast(nb)));
        try cuda.launch.launch(o.iota_vis, .{ .grid = .{ .x = blocks(nb, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The candidate pool's block maxima (model.py _candidates, rounds.py _candidate_blocks): out [rows, nb] fp32 (row
    /// stride os, nb = ceil(width / bsize)) = each block's maximum of the scores [rows, width] (row stride ss; the last
    /// block padded with -inf, a NaN wins), the row's newest block (vis - 1) // bsize pinned to +inf.
    pub fn blockMax(o: *const Ops, s: cuda.Stream, score: u64, ss: usize, width: usize, bsize: usize, vis: u64, out: u64, os: usize, rows: usize) !void {
        if (rows == 0) return;
        const nb = (width + bsize - 1) / bsize;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ss)));
        a.add(@as(c_int, @intCast(width)));
        a.add(@as(c_int, @intCast(bsize)));
        a.add(vis);
        a.add(out);
        a.add(@as(c_longlong, @intCast(os)));
        a.add(@as(c_int, @intCast(nb)));
        try cuda.launch.launch(o.block_max, .{ .grid = .{ .x = blocks(nb, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The pool from each row's top blocks idx [rows, k] int64 of the block maxima bmax [rows, nb] (row stride bs):
    /// _candidates' mask u8 [rows, nb] (row stride ms; zeros but the picked blocks above -inf) and/or
    /// _candidate_blocks' list int32 [rows, k] (the block, or -1 when its maximum is not above -inf); 0 leaves one out.
    pub fn poolPick(o: *const Ops, s: cuda.Stream, bmax: u64, bs: usize, nb: usize, idx: u64, k: usize, mask: u64, ms: usize, cblk: u64, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(bmax);
        a.add(@as(c_longlong, @intCast(bs)));
        a.add(@as(c_int, @intCast(nb)));
        a.add(idx);
        a.add(@as(c_int, @intCast(k)));
        a.add(mask);
        a.add(@as(c_longlong, @intCast(ms)));
        a.add(cblk);
        try cuda.launch.launch(o.pool_pick, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// model.py apply_candidates, in place: -inf at every column of score [rows, width] (row stride ss) whose block of
    /// bsize the row's pool mask (u8, row stride ms) leaves out.
    pub fn applyPool(o: *const Ops, s: cuda.Stream, score: u64, ss: usize, width: usize, bsize: usize, mask: u64, ms: usize, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ss)));
        a.add(@as(c_int, @intCast(width)));
        a.add(@as(c_int, @intCast(bsize)));
        a.add(mask);
        a.add(@as(c_longlong, @intCast(ms)));
        try cuda.launch.launch(o.apply_pool, .{ .grid = .{ .x = blocks(width, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// `.to(torch.bfloat16)` of `count` fp32 values (round to nearest even).
    pub fn toBf16(o: *const Ops, s: cuda.Stream, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.to_bf16, s, blocks(count, 256), 256, &a);
    }

    /// `rows` rows of `bytes` bytes from src (rows `src_ld` bytes apart) to dst (rows `dst_ld` bytes apart).
    pub fn copyRows(o: *const Ops, s: cuda.Stream, src: u64, src_ld: usize, dst: u64, dst_ld: usize, bytes: usize, rows: usize) !void {
        if (rows == 0) return;
        if (src_ld < bytes or dst_ld < bytes) return error.Overlap;
        if ((src | dst | src_ld | dst_ld | bytes) % 16 == 0 and rows <= 65535) {
            // 16-byte words, a word a thread (the same bytes)
            var w: cuda.Args = .{};
            w.add(src);
            w.add(@as(u64, src_ld / 16));
            w.add(dst);
            w.add(@as(u64, dst_ld / 16));
            w.add(@as(u64, bytes / 16));
            return cuda.launch.launch(o.copy16, .{ .grid = .{ .x = blocks(bytes / 16, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &w);
        }
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        // tf_strided_copy (outer, middle, inner, source_outer, source_middle, destination_outer, destination_middle):
        // a row a block
        for ([_]u64{ rows, 1, bytes, src_ld, bytes, dst_ld, bytes }) |v| a.add(v);
        try go(o.strided, s, rows, 256, &a);
    }

    /// Each row's greedy token (sampling.argmax's: the first largest, a NaN never larger, 0 when the first value is
    /// NaN): out [rows] u32 of logits [rows, n] fp32 (row stride ld floats).
    pub fn argmaxRows(o: *const Ops, s: cuda.Stream, logits: u64, ld: usize, n: usize, rows: usize, out: u64) !void {
        if (rows == 0 or n == 0) return;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, ld));
        a.add(@as(c_int, @intCast(n)));
        a.add(out);
        try go(o.argmax_rows, s, rows, 1024, &a);
    }

    /// Bounded sampled candidates: logical vocabulary n at physical row stride ld.
    pub fn topCandRows(o: *const Ops, s: cuda.Stream, logits: u64, ld: usize, n: usize, rows: usize, count: usize, vals: u64, ids: u64, flags: u64) !void {
        if (rows == 0 or n == 0 or count == 0 or count > 64) return error.BadCandidateShape;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, ld));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(count)));
        a.add(vals);
        a.add(ids);
        a.add(flags);
        try go(o.top_cand_rows orelse return error.NoSampleCandidateKernel, s, rows, 1024, &a);
    }

    /// Adds T * gumbel(seed, pos + step, col0 + v) to column v of each of `rows` streams' logits row (fp32, rows `l_row`
    /// floats apart, `width` columns; `table` the device KeyedRow of each stream), skipping streams with T <= 0.
    pub fn keyedNoise(o: *const Ops, s: cuda.Stream, logits: u64, l_row: usize, width: usize, col0: usize, table: u64, step: usize, rows: usize) !void {
        const f = o.keyed_noise orelse return error.KeyedDraftOff;
        if (rows == 0 or width == 0) return;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, l_row));
        a.add(@as(u32, @intCast(width)));
        a.add(@as(u64, col0));
        a.add(table);
        a.add(@as(u32, @intCast(step)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = blocks(width, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// noise restricted to each row's draft top_k after bias, before scoring; zero top_k is unrestricted. One block a
    /// row selects the cut into the table, then a grid as wide as keyedNoise's masks and adds the noise.
    pub fn keyedFilter(o: *const Ops, s: cuda.Stream, logits: u64, l_row: usize, width: usize, col0: usize, table: u64, step: usize, rows: usize, tok: u64, t_stride: usize, slot: u64, cache: u64, c_row: usize, stage: u64, s_row: usize) !void {
        const sel = o.keyed_select orelse return error.KeyedDraftOff;
        const app = o.keyed_apply orelse return error.KeyedDraftOff;
        if (rows == 0 or width == 0) return;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, l_row));
        a.add(@as(u32, @intCast(width)));
        a.add(table);
        a.add(tok);
        a.add(@as(u64, t_stride));
        a.add(slot);
        a.add(cache);
        a.add(@as(u64, c_row));
        a.add(stage);
        a.add(@as(u64, s_row));
        try cuda.launch.launch(sel, .{ .grid = .{ .x = 1, .y = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
        var b: cuda.Args = .{};
        b.add(logits);
        b.add(@as(u64, l_row));
        b.add(@as(u32, @intCast(width)));
        b.add(@as(u64, col0));
        b.add(table);
        b.add(@as(u32, @intCast(step)));
        b.add(tok);
        b.add(@as(u64, t_stride));
        b.add(slot);
        b.add(cache);
        b.add(@as(u64, c_row));
        b.add(stage);
        b.add(@as(u64, s_row));
        try cuda.launch.launch(app, .{ .grid = .{ .x = blocks(width, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &b);
    }

    /// The GPU clock (ns) into buf[i] when the stream reaches it (a profile).
    pub fn gpuClock(o: *const Ops, s: cuda.Stream, buf: u64, tag: u32, reset: bool, cap: u32) !void {
        var a: cuda.Args = .{};
        a.add(buf);
        a.add(tag);
        a.add(@as(u32, @intFromBool(reset)));
        a.add(cap);
        try go(o.gpu_clock, s, 1, 32, &a);
    }

    /// dst row i = src row idx[i] (int64 indices, rows of `bytes` bytes); an index outside [0, src_rows) sets the
    /// caller's zeroed uint32 flag `invalid` and leaves that row as it was.
    pub fn gatherRows(o: *const Ops, s: cuda.Stream, src: u64, src_rows: usize, idx: u64, dst: u64, bytes: usize, rows: usize, invalid: u64) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(idx);
        a.add(@as(u64, rows));
        a.add(@as(u64, src_rows));
        a.add(@as(u64, bytes));
        a.add(invalid);
        try go(o.gather, s, rows, 256, &a);
    }
};

test "launch geometry of the development launchers" {
    try std.testing.expectEqual(@as(u32, 1), blocks(1, 256));
    try std.testing.expectEqual(@as(u32, 2), blocks(257, 256));
    try std.testing.expectEqual(@as(u32, 65535), blocks(1 << 40, 256));
}

test "a keyed row is the kernel's 32-byte TfDsKeyedRow with top_k" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(KeyedRow));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(KeyedRow, "seed"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(KeyedRow, "top_k"));
    try std.testing.expectEqual(@as(usize, 28), @offsetOf(KeyedRow, "reserved"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(KeyedRow, "temp"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(KeyedRow, "pos"));
}


test "decode greedy head actual Ops launcher ABI and grids without CUDA" {
    const Fake = struct {
        var calls: usize = 0;
        var good: bool = true;
        fn arg(comptime T: type, params: ?[*]?*anyopaque, i: usize) T {
            return @as(*const T, @ptrCast(@alignCast(params.?[i].?))).*;
        }
        fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint,
            bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint,
            _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            calls += 1;
            good = good and gx == 3 and gy == 1 and gz == 1 and by == 1 and bz == 1 and shared == 0;
            if (@intFromPtr(f) == 101) {
                good = good and bx == 1024 and arg(u64, params, 0) == 0x11000 and
                    arg(u64, params, 1) == 43136 and arg(u32, params, 2) == 129280 and
                    arg(u32, params, 3) == 86272 and arg(u64, params, 4) == 0x22000;
            } else if (@intFromPtr(f) == 102) {
                good = good and bx == 32 and arg(u64, params, 0) == 0x33000 and
                    arg(u32, params, 1) == 3 and arg(u32, params, 2) == 3 and arg(u64, params, 3) == 0x44000;
            } else if (@intFromPtr(f) == 103) {
                good = good and bx == 1024 and arg(u64, params, 0) == 0x11000 and
                    arg(u64, params, 1) == 43136 and arg(u32, params, 2) == 129280 and
                    arg(u32, params, 3) == 86272 and arg(u32, params, 4) == 28 and arg(u64, params, 5) == 0x22000;
            } else if (@intFromPtr(f) == 104) {
                good = good and bx == 32 and arg(u64, params, 0) == 0x33000 and arg(u32, params, 1) == 3 and
                    arg(u32, params, 2) == 3 and arg(u32, params, 3) == 28 and arg(u64, params, 4) == 0x44000 and
                    arg(u64, params, 5) == 0x55000 and arg(u64, params, 6) == 0x66000;
            } else good = false;
            return cuda.abi.success;
        }
    };
    Fake.calls = 0;
    Fake.good = true;
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Fake.launch;
    // Pure host registry fixture; no Driver.open, cuInit or GPU context creation.
    const health = cuda.context_health;
    const fake_context: cuda.abi.Context = @ptrFromInt(0x7dd85001);
    try health.retain(@intFromPtr(&driver), fake_context);
    defer health.release(@intFromPtr(&driver), fake_context);
    health.setCurrent(@intFromPtr(&driver), fake_context);
    defer health.clearCurrent(@intFromPtr(&driver), fake_context);
    var owner = health.Owner.capture();
    defer owner.release();
    const stream: cuda.Stream = .{ .d = &driver, .owner = owner, .handle = null };
    var ops: Ops = undefined;
    ops.greedy_head_local = .{ .d = &driver, .owner = owner, .handle = @ptrFromInt(101) };
    ops.greedy_head_merge = .{ .d = &driver, .owner = owner, .handle = @ptrFromInt(102) };
    try ops.greedyHeadLocal(stream, 0x11000, 43136, 129280, 86272, 3, 0x22000);
    try ops.greedyHeadMerge(stream, 0x33000, 3, 3, 0x44000);
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    try std.testing.expect(Fake.good);
    ops.greedy_head_local = null;
    try std.testing.expectError(error.NoGreedyHeadKernel, ops.greedyHeadLocal(stream, 0x11000, 43136, 129280, 86272, 3, 0x22000));
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    ops.sample_head_local = .{ .d = &driver, .owner = owner, .handle = @ptrFromInt(103) };
    ops.sample_head_merge = .{ .d = &driver, .owner = owner, .handle = @ptrFromInt(104) };
    try ops.sampleHeadLocal(stream, 0x11000, 43136, 129280, 86272, 3, 28, 0x22000);
    try ops.sampleHeadMerge(stream, 0x33000, 3, 3, 28, 0x44000, 0x55000, 0x66000);
    try std.testing.expectEqual(@as(usize, 4), Fake.calls);
    try std.testing.expect(Fake.good);
    ops.sample_head_local = null;
    try std.testing.expectError(error.NoGreedyHeadKernel, ops.sampleHeadLocal(stream, 0x11000, 43136, 129280, 86272, 3, 28, 0x22000));
    try std.testing.expectEqual(@as(usize, 4), Fake.calls);
}


test "keyed filter actual launcher ABI and grid without CUDA" {
    const Fake = struct {
        var calls: usize = 0;
        var good: bool = true;
        fn arg(comptime T: type, params: ?[*]?*anyopaque, i: usize) T {
            return @as(*const T, @ptrCast(@alignCast(params.?[i].?))).*;
        }
        fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint,
            bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint,
            _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            calls += 1;
            good = good and gy == 3 and gz == 1 and by == 1 and bz == 1 and shared == 0;
            // select first, then apply, on the same stream
            if (calls == 1) {
                good = good and @intFromPtr(f) == 105 and gx == 1 and bx == 1024 and
                    arg(u64, params, 0) == 0x11000 and arg(u64, params, 1) == 215680 and
                    arg(u32, params, 2) == 43136 and arg(u64, params, 3) == 0x22000 and
                    arg(u64, params, 4) == 0x33010 and arg(u64, params, 5) == 6 and
                    arg(u64, params, 6) == 0x44000 and arg(u64, params, 7) == 0x55000 and
                    arg(u64, params, 8) == 43136 and arg(u64, params, 9) == 0x66000 and
                    arg(u64, params, 10) == 43136;
            } else if (calls == 2) {
                good = good and @intFromPtr(f) == 106 and gx == 169 and bx == 256 and
                    arg(u64, params, 0) == 0x11000 and arg(u64, params, 1) == 215680 and
                    arg(u32, params, 2) == 43136 and arg(u64, params, 3) == 86272 and
                    arg(u64, params, 4) == 0x22000 and arg(u32, params, 5) == 2 and
                    arg(u64, params, 6) == 0x33010 and arg(u64, params, 7) == 6 and
                    arg(u64, params, 8) == 0x44000 and arg(u64, params, 9) == 0x55000 and
                    arg(u64, params, 10) == 43136 and arg(u64, params, 11) == 0x66000 and
                    arg(u64, params, 12) == 43136;
            } else good = false;
            return cuda.abi.success;
        }
    };
    Fake.calls = 0;
    Fake.good = true;
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Fake.launch;
    // host-only fixture: no driver loading or GPU context creation.
    const health = cuda.context_health;
    const fake_context: cuda.abi.Context = @ptrFromInt(0x7dd85002);
    try health.retain(@intFromPtr(&driver), fake_context);
    defer health.release(@intFromPtr(&driver), fake_context);
    health.setCurrent(@intFromPtr(&driver), fake_context);
    defer health.clearCurrent(@intFromPtr(&driver), fake_context);
    var owner = health.Owner.capture();
    defer owner.release();
    const stream: cuda.Stream = .{ .d = &driver, .owner = owner, .handle = null };
    var ops: Ops = undefined;
    ops.keyed_select = .{ .d = &driver, .owner = owner, .handle = @ptrFromInt(105) };
    ops.keyed_apply = .{ .d = &driver, .owner = owner, .handle = @ptrFromInt(106) };
    try ops.keyedFilter(stream, 0x11000, 215680, 43136, 86272, 0x22000, 2, 3, 0x33010, 6, 0x44000, 0x55000, 43136, 0x66000, 43136);
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    try std.testing.expect(Fake.good);
    try ops.keyedFilter(stream, 0x11000, 215680, 43136, 86272, 0x22000, 2, 0, 0x33010, 6, 0x44000, 0x55000, 43136, 0x66000, 43136);
    try ops.keyedFilter(stream, 0x11000, 215680, 0, 86272, 0x22000, 2, 3, 0x33010, 6, 0x44000, 0x55000, 43136, 0x66000, 43136);
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    ops.keyed_apply = null;
    try std.testing.expectError(error.KeyedDraftOff, ops.keyedFilter(stream, 0x11000, 215680, 43136, 86272, 0x22000, 2, 3, 0x33010, 6, 0x44000, 0x55000, 43136, 0x66000, 43136));
    ops.keyed_select = null;
    try std.testing.expectError(error.KeyedDraftOff, ops.keyedFilter(stream, 0x11000, 215680, 43136, 86272, 0x22000, 2, 3, 0x33010, 6, 0x44000, 0x55000, 43136, 0x66000, 43136));
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
}
