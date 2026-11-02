//! The routed experts of a prompt chunk (exl3/experts.py routed(), the fast path the served build takes at 64 rows and
//! more, with its switches at their served values: the "mma" prompt kernel, work lists, no shared-dense GEMM): eight
//! launches of the served build's kernels (tensorfold_exl3_experts_v19: experts.cu's cubin and the codebook's
//! experts_cb*.cu cubin), as experts.cpp / experts.cu / experts_grouped.cuh launch them:
//! group_count, group_place, work_list, rot_in, grouped_mma (gate and up, the K splits folded), gateup_epilogue,
//! grouped_rows (TP2 down) or grouped_mma (TP3 down), and down_combine. And of a decode window (fewer rows: _routed_fused, the path routed() takes
//! with DECODE "fused", DECODE_PDL, DECODE_READY and the L2 discards at their served values): three launches,
//! decode_prep (the grouping and gate / up's rotated rows), grouped_cp_kernel for gate / up with their epilogue and
//! grouped_cp_kernel for down with the rows' combine, the last two programmatic dependents.
//! The map: pipeline/EXL3-EXPERTS-LAUNCHES.md.
const std = @import("std");
const cuda = @import("cuda");
const elf = @import("elf.zig");
const weights = @import("weights.zig");

/// experts.py's settings (GLM_GATEUP, GLM_DOWN by default_config; PROMPT_TILES; MMA_KB; EXACT_ROWS; DECODE_STAGES).
pub const exact_rows = 64;
pub const place_threads = 256; // experts.cu PLACE_THREADS
pub const list_threads = 1024; // experts.cu LIST_THREADS
pub const mma_cols = 64; // experts_grouped.cuh MMA_COLS
pub const mma_rows_gu = 64; // work list rows of the mma kernel
pub const rows_rows_d = 32; // work list rows of grouped_rows: 16 * PROMPT_TILES["down"][2]
pub const act_f32 = 1; // ACT_F32
pub const prep_threads = 256; // experts.cu PREP_THREADS
pub const decode_stages = 3; // DECODE_STAGES: each warp's ring of trellis copies in grouped_cp_kernel (its S)

/// A K2 range's template: (8, 8), (2, 10) or (2, 16), as every grouped launcher picks it.
pub fn range(lo: u32, hi: u32) usize {
    if (lo == 8 and hi == 8) return 0;
    if (lo >= 2 and hi <= 10) return 1;
    return 2;
}

/// default_config(K, N): (n tiles, warps, K splits, tiles in flight), GLM's where it divides.
pub fn config(k: usize, n: usize, gateup: bool) ![4]usize {
    const cands = [_][4]usize{ if (gateup) .{ 8, 4, 4, 1 } else .{ 8, 4, 1, 1 }, .{ 8, 4, 2, 1 }, .{ 8, 4, 1, 1 }, .{ 4, 4, 2, 2 }, .{ 4, 4, 1, 2 } };
    for (cands) |c| if (k % (16 * c[2] * c[1]) == 0 and n % (16 * c[0]) == 0) return c;
    return error.NoTileSetting;
}

/// _list_len: a work list's pairs for P member slots over at most maxu places in groups of `rows`.
pub fn listLen(p: usize, maxu: usize, rows: usize) usize {
    return (p + maxu * (rows - 1)) / rows;
}

/// The member lists' stride at R rows with work lists (busiest = R): whole 32-row tiles, at least one.
pub fn memberStride(r: usize) usize {
    const tile = 32;
    return @max(tile, (r + tile - 1) / tile * tile);
}

pub const Kernels = struct {
    base: cuda.Module,
    cb: cuda.Module,
    ragged_cb: ?cuda.Module = null,
    shared_cb: ?cuda.Module = null,
    shared_partition: ?cuda.Function = null,
    ragged_gu: [3]?cuda.Function = @splat(null),
    ragged_down: [3]?cuda.Function = @splat(null),
    group_count: cuda.Function,
    group_place: cuda.Function,
    work_list: cuda.Function,
    rot_in: cuda.Function, // <__nv_bfloat16>
    gateup_epilogue: cuda.Function,
    down_combine: cuda.Function,
    decode_prep: cuda.Function, // <__nv_bfloat16>
    mma: [3]?cuda.Function = @splat(null), // grouped_mma_kernel<CB, LO, HI, FOLD true, KB 4> by range()
    mma_down: [3]?cuda.Function = @splat(null), // FOLD false, KB 4: TP3's one-split down
    rows: [3]?cuda.Function = @splat(null), // grouped_rows_kernel<CB, 8, 4, 1, LO, HI, 2, FOLD false> by range()
    cp: [3][2]?cuda.Function = @splat(@splat(null)), // grouped_cp_kernel<CB, 8, 4, 3, LO, HI, EPI> by range(), EPI - 1

    /// experts.cu's cubin and the codebook's (experts_cb<codebook>.cu); kernels by name (the anonymous namespace's
    /// hash changes with every build) and template arguments.
    pub fn load(d: *const cuda.Driver, base_cubin: []const u8, cb_cubin: []const u8, codebook: u32) !Kernels {
        var base = try cuda.Module.load(d, base_cubin);
        errdefer base.unload();
        var cb = try cuda.Module.load(d, cb_cubin);
        errdefer cb.unload();
        var k: Kernels = .{ .base = base, .cb = cb, .group_count = undefined, .group_place = undefined, .work_list = undefined, .rot_in = undefined, .gateup_epilogue = undefined, .down_combine = undefined, .decode_prep = undefined };
        var found: u7 = 0;
        var names = elf.Symbols.init(base_cubin) orelse return error.BadCubin;
        while (names.next()) |name| {
            const plain = [_]struct { []const u8, u7 }{ .{ "18group_count_kernelE", 1 }, .{ "18group_place_kernelE", 2 }, .{ "16work_list_kernelE", 4 }, .{ "13rot_in_kernelI13__nv_bfloat16E", 8 }, .{ "22gateup_epilogue_kernelE", 16 }, .{ "19down_combine_kernelE", 32 }, .{ "18decode_prep_kernelI13__nv_bfloat16E", 64 } };
            for (plain) |e| {
                if (std.mem.indexOf(u8, name, e[0]) == null) continue;
                const f = try fnByName(base, name);
                switch (e[1]) {
                    1 => k.group_count = f,
                    2 => k.group_place = f,
                    4 => k.work_list = f,
                    8 => k.rot_in = f,
                    16 => k.gateup_epilogue = f,
                    32 => k.down_combine = f,
                    else => k.decode_prep = f,
                }
                found |= e[1];
            }
        }
        if (found != 127) return error.MissingKernel;
        names = elf.Symbols.init(cb_cubin) orelse return error.BadCubin;
        while (names.next()) |name| {
            if (std.mem.indexOf(u8, name, "18grouped_mma_kernelI")) |at| {
                var t: [5]i64 = undefined;
                if (!templateArgs(name[at + "18grouped_mma_kernel".len ..], &t)) continue;
                // <CB, LO, HI, FOLD, KB>
                if (t[0] != codebook or t[4] != 4) continue;
                const index = range(@intCast(t[1]), @intCast(t[2]));
                if (t[3] == 1) k.mma[index] = try fnByName(cb, name);
                if (t[3] == 0) k.mma_down[index] = try fnByName(cb, name);
            } else if (std.mem.indexOf(u8, name, "19grouped_rows_kernelI")) |at| {
                var t: [8]i64 = undefined;
                if (!templateArgs(name[at + "19grouped_rows_kernel".len ..], &t)) continue;
                // <CB, NT, W, PF, LO, HI, G, FOLD>
                if (t[0] != codebook or t[1] != 8 or t[2] != 4 or t[3] != 1 or t[6] != 2 or t[7] != 0) continue;
                k.rows[range(@intCast(t[4]), @intCast(t[5]))] = try fnByName(cb, name);
            } else if (std.mem.indexOf(u8, name, "17grouped_cp_kernelI")) |at| {
                var t: [7]i64 = undefined;
                if (!templateArgs(name[at + "17grouped_cp_kernel".len ..], &t)) continue;
                // <CB, NT, W, S, LO, HI, EPI>: grouped_cp_launch's one setting, the gate / up (1) and down (2) epilogues
                if (t[0] != codebook or t[1] != 8 or t[2] != 4 or t[3] != decode_stages or t[6] < 1 or t[6] > 2) continue;
                k.cp[range(@intCast(t[4]), @intCast(t[5]))][@intCast(t[6] - 1)] = try fnByName(cb, name);
            }
        }
        return k;
    }

    /// Separate module; original prompt/decode kernels and kit remain intact.
    pub fn loadRagged(k: *Kernels, d: *const cuda.Driver, blob: []const u8) !void {
        if (k.ragged_cb != null) return error.RaggedAlreadyLoaded;
        const cleanup = cuda.context_health.cleanupMark();
        var module = try cuda.Module.load(d, blob);
        errdefer if (cuda.context_health.cleanupReleased(cleanup)) module.unload();
        var names = elf.Symbols.init(blob) orelse return error.BadCubin;
        var gu: [3]?cuda.Function = @splat(null);
        var dn: [3]?cuda.Function = @splat(null);
        while (names.next()) |name| {
            if (std.mem.indexOf(u8, name, "18grouped_mma_kernelI")) |at| {
                var t: [5]i64 = undefined;
                if (!templateArgs(name[at + "18grouped_mma_kernel".len ..], &t)) continue;
                if (t[0] != 2 or t[4] != 4) return error.UnexpectedRaggedKernel;
                const index = range(@intCast(t[1]), @intCast(t[2]));
                if (!raggedModuleRangeAllowed(prompt_ragged_range16, prompt_shared_split, index)) return error.UnexpectedRaggedRange;
                if (index == 2 and (t[1] != 2 or t[2] != 16)) return error.UnexpectedRaggedRange;
                const f = try fnByName(module, name);
                if (index == 2) try f.allowDynamicShared(ragged_range16_shared);
                if (t[3] == 1) gu[index] = f else dn[index] = f;
            }
        }
        if (gu[0] == null or gu[1] == null or dn[0] == null or dn[1] == null) return error.MissingRaggedKernel;
        if (prompt_ragged_range16 and (gu[2] == null or dn[2] == null)) return error.MissingRaggedKernel;
        // A separate embedded integer-only module leaves either measured HI8/10
        // or HI16 W cubin untouched. Off performs exactly the original loads.
        if (prompt_shared_split) {
            var partition_module = try cuda.Module.load(d, &shared_split_image);
            errdefer if (cuda.context_health.cleanupReleased(cleanup)) partition_module.unload();
            const partition = try partition_module.function("tf_shared_split_partition");
            k.shared_cb = partition_module;
            k.shared_partition = partition;
        }
        split_checked = @splat(false);
        k.ragged_cb = module;
        k.ragged_gu = gu;
        k.ragged_down = dn;
        ragged_checked = @splat(@splat(false));
    }

    pub fn unload(k: *Kernels) void {
        const cleanup = cuda.context_health.cleanupMark();
        split_checked = @splat(false);
        if (k.shared_cb) |*module| module.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return;
        if (k.ragged_cb) |*module| module.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return;
        k.base.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return;
        k.cb.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return;
        if (!cuda.context_health.cleanupReleased(cleanup)) return;
        k.* = undefined;
    }
};

fn fnByName(m: cuda.Module, name: []const u8) !cuda.Function {
    var buf: [512]u8 = undefined;
    return m.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
}

/// The integer and bool template arguments of a mangled kernel name's "I...E" list ("ILi2ELi8ELb1ELi4EE" -> 2, 8, 1,
/// 4); false unless there are exactly out.len of them.
pub fn templateArgs(s: []const u8, out: []i64) bool {
    if (s.len == 0 or s[0] != 'I') return false;
    var i: usize = 1;
    var n: usize = 0;
    while (i < s.len and s[i] == 'L') {
        if (i + 2 >= s.len) return false;
        const kind = s[i + 1];
        const e = std.mem.indexOfScalarPos(u8, s, i + 2, 'E') orelse return false;
        if (n == out.len) return false;
        out[n] = switch (kind) {
            'i', 'b' => std.fmt.parseInt(i64, s[i + 2 .. e], 10) catch return false,
            else => return false,
        };
        n += 1;
        i = e + 1;
    }
    return n == out.len and i < s.len and s[i] == 'E';
}

/// A prompt Scratch (experts.py Scratch(prompt=True)) for up to `rows` rows of `slots` slots, and a chunk's work lists:
/// device addresses the caller allocated with these sizes and initial contents.
pub const Scratch = struct {
    rows: usize,
    slots: usize,
    xg: u64, // fp16 [rows * slots, D], zeros
    xu: u64, // fp16 [rows * slots, D], zeros
    xd: u64, // fp16 [rows * slots, I], zeros
    z: u64, // fp32 [promptZ(D, I) * rows * slots], zeros
    no_y: u64, // fp32 [1], zeros
    ids: u64, // int32 [min(rows * slots, E)], zeros
    count: u64, // int32 [1], zeros
    counts: u64, // int32 [E], zeros
    members: u64, // int32 [min(rows * slots, E) * ceil(rows / 128) * 128], -1
    work_gu: u64, // int32 [listLen(rows * slots, maxu, 64) * 2]
    work_d: u64, // int32 [listLen(rows * slots, maxu, 32) * 2]

    pub fn promptZ(d: usize, i: usize) usize {
        return @max(2 * i + d / 2, d);
    }

    /// Bytes of each buffer, in the field order above (xg .. work_d).
    pub fn sizes(rows: usize, slots: usize, d: usize, i: usize, e: usize) [11]usize {
        const p = rows * slots;
        const maxu = @min(p, e);
        return .{ 2 * p * d, 2 * p * d, 2 * p * i, 4 * promptZ(d, i) * p, 4, 4 * maxu, 4, 4 * e, 4 * maxu * ((rows + 127) / 128 * 128), 8 * listLen(p, maxu, mma_rows_gu), 8 * listLen(p, maxu, rows_rows_d) };
    }
};

fn i32a(a: *cuda.Args, v: usize) void {
    a.add(@as(c_int, @intCast(v)));
}

/// routed(x, pick, wts, ex, s, out, R, limit, ACT_F32) on a prompt chunk (R >= 64): out fp32 [R, D] = each row's slots'
/// expert outputs weighted by wts, summed in slot order. x bf16 [R, D] (row stride x_stride elements), pick int32 and
/// wts fp32 [R, slots] contiguous; every pick an expert id (the shared expert is the last).
pub fn prompt(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: Scratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, r: usize, limit: f32) !void {
    var ls = try promptLaunches(ex, sc, x, x_stride, pick, wts, out, r, limit);
    // Resolve the entire sequence before the first GPU launch.
    var fs: [8]cuda.Function = undefined;
    for (&ls, &fs) |*l, *f| f.* = switch (l.kernel) {
        .group_count => k.group_count,
        .group_place => k.group_place,
        .work_list => k.work_list,
        .rot_in => k.rot_in,
        .gate_up => k.mma[l.range] orelse return error.MissingKernel,
        .gateup_epilogue => k.gateup_epilogue,
        .down_mma => k.mma_down[l.range] orelse return error.MissingKernel,
        .down_rows => k.rows[l.range] orelse return error.MissingKernel,
        .down_combine => k.down_combine,
    };
    const enabled = raggedEligible(prompt_ragged_mma, r, ex.dims, ex.width, sc.slots) and k.ragged_cb != null;
    if (sharedSplitRanges(prompt_shared_split and enabled, ex, sc.slots)) |ranges| {
        splitPrompt(k, s, ex, sc, r, &ls, fs, ranges) catch |err| return @import("prefill_128.zig").qualificationFault(err);
        return;
    }
    for (&ls, fs) |*l, f| {
        if (enabled and raggedRangeAllowed(prompt_ragged_range16, l.range) and (l.kernel == .gate_up or l.kernel == .down_mma)) {
            const stage: usize = if (l.kernel == .gate_up) 0 else 1;
            const candidate = (if (stage == 0) k.ragged_gu[l.range] else k.ragged_down[l.range]) orelse return error.MissingRaggedKernel;
            raggedLaunch(s, l, f, candidate, stage, r * sc.slots, if (stage == 0) 2 * ex.width else ex.dims, sc.z) catch |err| return @import("prefill_128.zig").qualificationFault(err);
        } else try cuda.launch.launch(f, l.cfg, s, &l.args);
    }
}

pub const PromptLaunch = struct {
    kernel: enum { group_count, group_place, work_list, rot_in, gate_up, gateup_epilogue, down_mma, down_rows, down_combine },
    range: usize = 0,
    cfg: cuda.Config,
    args: cuda.Args,
};

fn promptLaunch(kernel: @FieldType(PromptLaunch, "kernel"), grid: [3]usize, block: usize, args: cuda.Args) PromptLaunch {
    return .{ .kernel = kernel, .cfg = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, .args = args };
}

/// CPU-visible launch plan shared with production prompt(). Python's mma path
/// uses 64-row work lists for TP3 down (I=768), but 32 rows for TP2 (I=1152).
pub fn promptLaunches(ex: weights.Experts, sc: Scratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, r: usize, limit: f32) ![8]PromptLaunch {
    var ls: [8]PromptLaunch = undefined;
    const d: usize = ex.dims;
    const i: usize = ex.width;
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r < exact_rows or r > sc.rows) return error.NotAPromptChunk;
    if (slots == 0 or slots > 32 or e == 0 or i == 0 or d == 0 or i % 128 != 0 or d % 128 != 0) return error.UnsupportedShape;
    const cgu = try config(d, i, true);
    const cd = try config(i, d, false);
    // the served fast path: mma for gate / up (K chains a multiple of MMA_KB), grouped_rows for down
    const chain_gu = (d / 16) / (cgu[1] * cgu[2]);
    if ((d / 16) % (cgu[1] * cgu[2]) != 0 or chain_gu % 4 != 0) return error.UnsupportedShape;
    const chain_d = (i / 16) / (cd[1] * cd[2]);
    const mma_d = (i / 16) % (cd[1] * cd[2]) == 0 and chain_d % 4 == 0;
    if (cd[2] != 1) return error.UnsupportedShape;
    const p = r * slots;
    const maxu = @min(p, e);
    const maxm = memberStride(r);
    const n_gu = listLen(p, maxu, mma_rows_gu);
    const down_rows: usize = if (mma_d) mma_rows_gu else rows_rows_d;
    const n_d = listLen(p, maxu, down_rows);

    var a: cuda.Args = .{};
    a.add(pick);
    a.add(sc.counts);
    i32a(&a, p);
    ls[0] = promptLaunch(.group_count, .{ e, 1, 1 }, place_threads, a);

    a = .{};
    for ([_]u64{ pick, sc.counts, sc.ids, sc.count, sc.members }) |v| a.add(v);
    for ([_]usize{ p, slots, e, maxm }) |v| i32a(&a, v);
    ls[1] = promptLaunch(.group_place, .{ e, 1, 1 }, place_threads, a);

    a = .{};
    for ([_]u64{ sc.counts, sc.ids, sc.count }) |v| a.add(v);
    i32a(&a, maxu);
    a.add(sc.work_gu);
    i32a(&a, mma_rows_gu);
    i32a(&a, n_gu);
    a.add(sc.work_d);
    i32a(&a, down_rows);
    i32a(&a, n_d);
    a.add(@as(c_int, -1)); // skip_e: no shared-dense GEMM
    ls[2] = promptLaunch(.work_list, .{ 1, 1, 1 }, list_threads, a);

    a = .{};
    a.add(x);
    i32a(&a, x_stride);
    for ([_]u64{ pick, ex.suh_g, ex.suh_u, sc.xg, sc.xu }) |v| a.add(v);
    for ([_]usize{ d, slots, e }) |v| i32a(&a, v);
    ls[3] = promptLaunch(.rot_in, .{ p, d / 128, 2 }, 32, a);

    // gate and up: grid (1, N / 64, pairs x 2 matrices), the K splits summed in the program (FOLD)
    a = .{};
    for ([_]u64{ sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ d, i, p, cgu[2], cgu[1], maxm, slots, n_gu }) |v| i32a(&a, v);
    a.add(sc.work_gu);
    ls[4] = promptLaunch(.gate_up, .{ 1, i / mma_cols, n_gu * 2 }, 256, a);
    ls[4].range = range(ex.k2_gu[0], ex.k2_gu[1]);

    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| a.add(v);
    for ([_]usize{ p, i, 1, e }) |v| i32a(&a, v);
    a.add(limit);
    a.add(@as(c_int, act_f32));
    ls[5] = promptLaunch(.gateup_epilogue, .{ p, i / 128, 1 }, 32, a);

    // One-split down: MMA uses D/64 and 64-row work pairs; rows uses D/128
    // and two 16-row member tiles. MMA ABI additionally takes the warp count.
    a = .{};
    for ([_]u64{ sc.xd, sc.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ i, d, p, 1 }) |v| i32a(&a, v);
    if (mma_d) i32a(&a, cd[1]);
    for ([_]usize{ maxm, slots, n_d }) |v| i32a(&a, v);
    a.add(sc.work_d);
    ls[6] = if (mma_d) promptLaunch(.down_mma, .{ 1, d / mma_cols, n_d }, 256, a) else promptLaunch(.down_rows, .{ 1, d / 128, n_d }, 4 * 32, a);
    ls[6].range = range(ex.k2_d[0], ex.k2_d[1]);

    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_d, sc.no_y, wts, 0, out }) |v| a.add(v);
    for ([_]usize{ p, d, 1, e, slots, 0 }) |v| i32a(&a, v);
    ls[7] = promptLaunch(.down_combine, .{ r, d / 128, 1 }, 32 * slots, a);
    for (&ls) |*l| try l.cfg.validate();
    return ls;
}

/// experts_grouped.cuh's DecodeEpi, field for field (C layout, its defaults; the C padding as zero fields, the bytes
/// every recorded launch passed): grouped_cp_kernel's by-value epilogue. The gate / up launch fills svh_g .. act_mode,
/// the down launch svh_d .. store_y, both pick, E, cnt, the readiness (ready, ready_cnt, epoch) and discard.
pub const DecodeEpi = extern struct {
    pick: u64 = 0,
    E: c_int = 0,
    pad0: c_int = 0,
    svh_g: u64 = 0,
    svh_u: u64 = 0,
    suh_d: u64 = 0,
    xd: u64 = 0,
    limit: f32 = 0,
    act_mode: c_int = 1,
    svh_d: u64 = 0,
    y: u64 = 0,
    wts: u64 = 0,
    add: u64 = 0,
    out: u64 = 0,
    store_y: c_int = 1,
    pad1: c_int = 0,
    cnt: u64 = 0,
    ready: u64 = 0,
    ready_cnt: u64 = 0,
    epoch: u64 = 0,
    discard: c_int = 0,
    pad2: c_int = 0,
};

comptime {
    // the layout the kernels were compiled with (pointers 8 bytes, ints and floats 4, C padding)
    std.debug.assert(@sizeOf(DecodeEpi) == 144 and @offsetOf(DecodeEpi, "svh_g") == 16 and @offsetOf(DecodeEpi, "cnt") == 104 and @offsetOf(DecodeEpi, "discard") == 136);
}

/// A decode Scratch (experts.py Scratch(prompt=False); the served engine's holds max(n, 64) rows) for windows of up to
/// `rows` rows of `slots` slots: device addresses the caller allocated with these sizes and initial contents. A call
/// writes ids, count and members (decode_prep), the routed slots' xg and xu rows (decode_prep), z and xd (gate / up)
/// and y (down) before it reads them. epoch is the call's number (decode_prep adds one) and ready[u] the number expert
/// place u's rows were last published to down under: both carry over from call to call. The counters cnt_gu, cnt_d and
/// ready_cnt start at zero and each launch's last program leaves them zero. Scratch's counts, no_y and no_work reach
/// no decode launch (experts.cu passes a null pointer for each tensor a launch does not take).
pub const DecodeScratch = struct {
    rows: usize,
    slots: usize,
    xg: u64, // fp16 [rows * slots, D], zeros
    xu: u64, // fp16 [rows * slots, D], zeros
    xd: u64, // fp16 [rows * slots, I], zeros
    z: u64, // fp32 [max(2 * gate/up splits * I, down splits * D) * rows * slots], zeros
    y: u64, // fp32 [rows * slots, D], zeros
    cnt_gu: u64, // int32 [maxu * max(1, I / 128)], zeros
    cnt_d: u64, // int32 [rows * max(1, D / 128)], zeros
    epoch: u64, // int32 [1], zeros
    ready: u64, // int32 [max(1, maxu)], zeros
    ready_cnt: u64, // int32 [max(1, maxu)], zeros
    ids: u64, // int32 [maxu], zeros
    count: u64, // int32 [1], zeros
    members: u64, // int32 [maxu * ceil(rows / 128) * 128], -1

    /// Bytes of each buffer, in the field order above (xg .. members); maxu = min(rows * slots, E).
    pub fn sizes(rows: usize, slots: usize, d: usize, i: usize, e: usize) ![13]usize {
        const p = rows * slots;
        const maxu = @min(p, e);
        const one: usize = 1;
        const zrow = @max(2 * (try config(d, i, true))[2] * i, (try config(i, d, false))[2] * d);
        return .{ 2 * p * d, 2 * p * d, 2 * p * i, 4 * zrow * p, 4 * p * d, 4 * maxu * @max(one, i / 128), 4 * rows * @max(one, d / 128), 4, 4 * @max(one, maxu), 4 * @max(one, maxu), 4 * maxu, 4, 4 * maxu * ((rows + 127) / 128 * 128) };
    }
};

/// One of decode's launches: its kernel (decode_prep, or grouped_cp_kernel with the gate / up or the down epilogue at
/// K2 range `range`), configuration and arguments.
pub const DecodeLaunch = struct {
    kernel: enum { prep, gate_up, down },
    range: usize = 0,
    cfg: cuda.Config,
    args: cuda.Args = .{},
};

/// routed(x, pick, wts, ex, s, None, R, limit, ACT_F32, before_down=None, shared_slot=slots - 1) on a decode window
/// (R < 64), the served Model.moe's one call a layer: out fp32 [R, D] = each row's slots' expert outputs weighted by
/// wts, summed in slot order. x bf16 [R, D] (row stride x_stride elements), pick int32 and wts fp32 [R, slots]
/// contiguous, every pick an expert id (the shared expert, the last, in every row's last slot); out need not be
/// initialized. shared_slot changes nothing here (only a prompt chunk's SHARED_DENSE path reads it), and without
/// before_down nothing waits before down. Three launches on `s`: decode_prep plain, gate / up and down as programmatic
/// dependents.
pub fn decode(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: DecodeScratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, r: usize, limit: f32) !void {
    var ls = try decodeLaunches(ex, sc, x, x_stride, pick, wts, out, r, limit);
    var fs: [3]cuda.Function = undefined;
    for (&ls, &fs) |*l, *f| f.* = switch (l.kernel) {
        .prep => k.decode_prep,
        .gate_up => k.cp[l.range][0] orelse return error.MissingKernel,
        .down => k.cp[l.range][1] orelse return error.MissingKernel,
    };
    for (&ls, fs) |*l, f| try cuda.launch.launch(f, l.cfg, s, &l.args);
}

/// decode's three launches in order: experts.py _routed_fused, experts.cu exl3x_decode_prep_cuda and
/// exl3x_grouped_decode_cuda, experts_grouped.cuh grouped_cp_launch.
pub fn decodeLaunches(ex: weights.Experts, sc: DecodeScratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, r: usize, limit: f32) ![3]DecodeLaunch {
    const d: usize = ex.dims;
    const i: usize = ex.width;
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r == 0 or r >= exact_rows or r > sc.rows) return error.NotADecodeWindow;
    // routed()'s gate of the fused path: GLM's tiles for gate / up (8 n tiles, 4 warps) and down (one split), 128-column
    // blocks, at most 32 slots; and every trellis 16-byte aligned (weights packs each at a multiple of 16 bytes)
    const cgu = try config(d, i, true);
    const cd = try config(i, d, false);
    if (cgu[0] != 8 or cgu[1] != 4 or cd[0] != 8 or cd[1] != 4 or cd[2] != 1) return error.UnsupportedShape;
    if (i % 128 != 0 or d % 128 != 0 or slots > 32) return error.UnsupportedShape;
    const p = r * slots;
    const maxu = @min(p, e); // the window's expert places (Scratch.window): ids [maxu], members [maxu, R]
    const mt = (r + 15) / 16; // a place's 16-row member tiles

    // decode_prep: block 0 adds one to epoch, groups the picks (ids, count, members) and combines the rows that route
    // no slot; the other blocks rotate gate / up's input rows, a warp a (member, 128 columns, matrix)
    var prep: cuda.Args = .{};
    prep.add(x);
    i32a(&prep, x_stride);
    for ([_]u64{ pick, ex.suh_g, ex.suh_u, sc.xg, sc.xu }) |v| prep.add(v);
    for ([_]usize{ d, slots, e, r }) |v| i32a(&prep, v);
    for ([_]u64{ sc.ids, sc.count, sc.members }) |v| prep.add(v);
    i32a(&prep, r); // maxm: the member lists' stride
    for ([_]u64{ wts, sc.y, 0, out }) |v| prep.add(v); // no add
    for ([_]usize{ d, 1 }) |v| i32a(&prep, v); // store_y: a slot that is not routed adds the y it finds
    prep.add(sc.epoch);
    const prep_warps = prep_threads / 32;
    const blocks = 1 + (p * (d / 128) * 2 + prep_warps - 1) / prep_warps;

    // gate and up: grid (places, I / 128, 2 matrices x K splits x member tiles), 4 warps copying trellis words through
    // 3-stage rings, the last place first; the program completing a (place, 128 columns) sums the splits into the
    // gate / up epilogue (xd) and the place's last such program publishes ready[place] = epoch; with discard the
    // summed partials (8 x splits <= 32 lanes) and the place's rotated rows leave L2 unwritten
    var gu: cuda.Args = .{};
    for ([_]u64{ sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc.ids, sc.count, sc.members, sc.z }) |v| gu.add(v);
    for ([_]usize{ d, i, p, cgu[2], r, slots }) |v| i32a(&gu, v);
    gu.add(DecodeEpi{ .pick = pick, .E = @intCast(e), .svh_g = ex.svh_g, .svh_u = ex.svh_u, .suh_d = ex.suh_d, .xd = sc.xd, .limit = limit, .act_mode = act_f32, .cnt = sc.cnt_gu, .ready = sc.ready, .ready_cnt = sc.ready_cnt, .epoch = sc.epoch, .discard = @intFromBool(8 * cgu[2] <= 32) });

    // down: grid (places, D / 128, member tiles), one split; a program copies its first trellis steps, waits for its
    // place's ready == epoch, writes its rows' slot outputs to y, and the program completing a (row, 128 columns)
    // combines the row's slots with wts into out, then drops their y from L2 (4 x slots <= 32 lanes); limit and
    // act_mode keep DecodeEpi's defaults
    var dn: cuda.Args = .{};
    for ([_]u64{ sc.xd, sc.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc.ids, sc.count, sc.members, sc.z }) |v| dn.add(v);
    for ([_]usize{ i, d, p, 1, r, slots }) |v| i32a(&dn, v);
    dn.add(DecodeEpi{ .pick = pick, .E = @intCast(e), .svh_d = ex.svh_d, .y = sc.y, .wts = wts, .out = out, .cnt = sc.cnt_d, .ready = sc.ready, .ready_cnt = sc.ready_cnt, .epoch = sc.epoch, .discard = @intFromBool(4 * slots <= 32) });

    return .{
        .{ .kernel = .prep, .cfg = .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = prep_threads }, .shared = @intCast(2 * p * 4) }, .args = prep },
        .{ .kernel = .gate_up, .range = range(ex.k2_gu[0], ex.k2_gu[1]), .cfg = .{ .grid = .{ .x = @intCast(maxu), .y = @intCast(i / 128), .z = @intCast(2 * cgu[2] * mt) }, .block = .{ .x = 4 * 32 }, .pdl = true }, .args = gu },
        .{ .kernel = .down, .range = range(ex.k2_d[0], ex.k2_d[1]), .cfg = .{ .grid = .{ .x = @intCast(maxu), .y = @intCast(d / 128), .z = @intCast(mt) }, .block = .{ .x = 4 * 32 }, .pdl = true }, .args = dn },
    };
}

test "template arguments of the served experts kernels" {
    var t5: [5]i64 = undefined;
    try std.testing.expect(templateArgs("ILi2ELi2ELi10ELb1ELi4EEEvPK6__half", &t5));
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 10, 1, 4 }, &t5);
    var t8: [8]i64 = undefined;
    try std.testing.expect(templateArgs("ILi2ELi8ELi4ELi1ELi8ELi8ELi2ELb0EEEvPK6__half", &t8));
    try std.testing.expectEqualSlices(i64, &.{ 2, 8, 4, 1, 8, 8, 2, 0 }, &t8);
    try std.testing.expect(!templateArgs("ILi2ELi8EEEv", &t5));
    try std.testing.expect(!templateArgs("v", &t5));
}

test "a TP2 DeepSeek-V4.1 rank's prompt experts: settings, lists and ranges" {
    // D 5120, I 1152, 385 experts (the shared one last), 7 slots
    try std.testing.expectEqual([4]usize{ 8, 4, 4, 1 }, try config(5120, 1152, true));
    try std.testing.expectEqual([4]usize{ 8, 4, 1, 1 }, try config(1152, 5120, false));
    try std.testing.expectEqual(@as(usize, 2048), memberStride(2048));
    try std.testing.expectEqual(@as(usize, 192), memberStride(181));
    try std.testing.expectEqual(@as(usize, 64), memberStride(64));
    try std.testing.expectEqual(@as(usize, 32), memberStride(20));
    try std.testing.expectEqual(@as(usize, (14336 + 385 * 63) / 64), listLen(2048 * 7, 385, 64));
    try std.testing.expectEqual(@as(usize, 0), range(8, 8));
    try std.testing.expectEqual(@as(usize, 1), range(3, 10));
    try std.testing.expectEqual(@as(usize, 2), range(4, 12));
    try std.testing.expectEqual(@as(usize, 5120), Scratch.promptZ(5120, 1152));
}

test "a TP2 DeepSeek-V4.1 rank's decode scratch and windows" {
    // a decoder layer (D 5120, I 1152, 385 experts, 7 slots) and an MTP layer (129 experts, 4 slots), 64 rows
    try std.testing.expectEqual([13]usize{ 4587520, 4587520, 1032192, 16515072, 9175040, 13860, 10240, 4, 1540, 1540, 1540, 4, 197120 }, try DecodeScratch.sizes(64, 7, 5120, 1152, 385));
    try std.testing.expectEqual([13]usize{ 2621440, 2621440, 589824, 9437184, 5242880, 4644, 10240, 4, 516, 516, 516, 4, 66048 }, try DecodeScratch.sizes(64, 4, 5120, 1152, 129));
    // 1 to 63 rows (no more than the scratch holds); 64 and more are prompt chunks
    const ex: weights.Experts = .{ .trellis = 0, .gate_ptr = 0, .up_ptr = 0, .down_ptr = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .count = 385, .dims = 5120, .width = 1152, .down_k = 1152, .down_n = 5120, .k2_gu = .{ 6, 10 }, .k2_d = .{ 6, 10 } };
    const sc: DecodeScratch = .{ .rows = 64, .slots = 7, .xg = 0, .xu = 0, .xd = 0, .z = 0, .y = 0, .cnt_gu = 0, .cnt_d = 0, .epoch = 0, .ready = 0, .ready_cnt = 0, .ids = 0, .count = 0, .members = 0 };
    for ([_]usize{ 0, 64, 65 }) |r| try std.testing.expectError(error.NotADecodeWindow, decodeLaunches(ex, sc, 0, 5120, 0, 0, 0, r, 10));
    var small = sc;
    small.rows = 8;
    try std.testing.expectError(error.NotADecodeWindow, decodeLaunches(ex, small, 0, 5120, 0, 0, 0, 9, 10));
    _ = try decodeLaunches(ex, small, 0, 5120, 0, 0, 0, 8, 10);
}

test "TP3 prompt down follows upstream MMA ABI and TP2 retains rows ABI" {
    var ex: weights.Experts = .{ .trellis = 0, .gate_ptr = 101, .up_ptr = 102, .down_ptr = 103, .gate_k2 = 104, .up_k2 = 105, .down_k2 = 106, .suh_g = 107, .suh_u = 108, .svh_g = 109, .svh_u = 110, .suh_d = 111, .svh_d = 112, .count = 385, .dims = 5120, .width = 768, .down_k = 768, .down_n = 5120, .k2_gu = .{ 4, 10 }, .k2_d = .{ 2, 16 } };
    var sc: Scratch = .{ .rows = 4096, .slots = 7, .xg = 201, .xu = 202, .xd = 203, .z = 204, .no_y = 205, .ids = 206, .count = 207, .counts = 208, .members = 209, .work_gu = 210, .work_d = 211 };
    for ([_]usize{ 64, 65, 128, 4095, 4096 }) |r| {
        for ([_]u32{ 768, 1152 }) |width| {
            ex.width = width;
            ex.down_k = width;
            for ([_]usize{ 4, 7 }) |slots| {
                sc.slots = slots;
                ex.count = if (slots == 4) 129 else 385;
                const ls = try promptLaunches(ex, sc, 301, 5120, 302, 303, 304, r, 10);
                const mma = width == 768;
                const p = r * slots;
                const maxu = @min(p, ex.count);
                const pair_rows: usize = if (mma) 64 else 32;
                const n_down = (p + maxu * (pair_rows - 1)) / pair_rows;
                const expected_kernel: @FieldType(PromptLaunch, "kernel") = if (mma) .down_mma else .down_rows;
                try std.testing.expectEqual(expected_kernel, ls[6].kernel);
                try std.testing.expectEqual(@as(usize, 2), ls[6].range);
                try std.testing.expectEqual(@as(u32, if (mma) 256 else 128), ls[6].cfg.block.x);
                try std.testing.expectEqual(@as(u32, if (mma) 80 else 40), ls[6].cfg.grid.y);
                try std.testing.expectEqual(n_down, ls[6].cfg.grid.z);
                try std.testing.expectEqual(@as(u64, pair_rows), ls[2].args.integer(8).?);
                try std.testing.expectEqual(@as(u64, n_down), ls[2].args.integer(9).?);
                try std.testing.expectEqual(@as(usize, if (mma) 19 else 18), ls[6].args.count);
                const prefix = [_]u64{ 203, 203, 103, 103, 106, 106, 206, 207, 209, 204, width, 5120, p, 1 };
                for (prefix, 0..) |value, j| try std.testing.expectEqual(value, ls[6].args.integer(j).?);
                const shift: usize = @intFromBool(mma);
                if (mma) try std.testing.expectEqual(@as(u64, 4), ls[6].args.integer(14).?);
                try std.testing.expectEqual(@as(u64, (r + 31) / 32 * 32), ls[6].args.integer(14 + shift).?);
                try std.testing.expectEqual(@as(u64, slots), ls[6].args.integer(15 + shift).?);
                try std.testing.expectEqual(@as(u64, n_down), ls[6].args.integer(16 + shift).?);
                try std.testing.expectEqual(@as(u64, 211), ls[6].args.integer(17 + shift).?);
                // Existing 32-row allocation bounds the smaller 64-row list.
                const sizes = Scratch.sizes(sc.rows, slots, 5120, width, ex.count);
                try std.testing.expect(8 * n_down <= sizes[10]);
                try std.testing.expectEqual(@as(u64, 1), ls[7].args.integer(9).?); // down_combine SK
            }
        }
    }
    try std.testing.expectError(error.NotAPromptChunk, promptLaunches(ex, sc, 0, 5120, 0, 0, 0, 63, 10));
    try std.testing.expectError(error.NotAPromptChunk, promptLaunches(ex, sc, 0, 5120, 0, 0, 0, 4097, 10));
}

pub var prompt_shared_split = false;
const shared_split_image align(16) = @embedFile("fixtures/shared_split.cubin").*;

/// Bind persistent/rank identity to the exact embedded partition module. Off
/// returns the same borrowed bytes, with no allocation or fingerprint change.
pub fn sharedSplitBuildFlags(a: std.mem.Allocator, base: []const u8, enabled: bool) ![]const u8 {
    if (!enabled) return base;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&shared_split_image, &digest, .{});
    return std.fmt.allocPrint(a, "{s};prefill128_shared_split={x}", .{ base, &digest });
}
const SplitRanges = struct { routed: [2]usize, shared: [2]usize };

fn validK2(bounds: [2]u32) bool {
    return bounds[0] >= 2 and bounds[0] <= bounds[1] and bounds[1] <= 16;
}

/// Derived from each rank's actual loader metadata, never rank-0's histogram.
/// MTP (all K2=8), unsupported shapes and unavailable metadata stay unsplit.
fn sharedSplitRanges(enabled: bool, ex: weights.Experts, slots: usize) ?SplitRanges {
    if (!enabled or ex.count < 2 or slots < 2 or slots > 32 or ex.dims != 5120 or ex.width != 768 or ex.down_k != ex.width or ex.down_n != ex.dims) return null;
    const routed = [_][2]u32{ ex.routed_k2_gu, ex.routed_k2_d };
    const shared = [_][2]u32{ ex.shared_k2_gu, ex.shared_k2_d };
    var result: SplitRanges = undefined;
    for (0..2) |stage| {
        if (!validK2(routed[stage]) or !validK2(shared[stage])) return null;
        result.routed[stage] = range(routed[stage][0], routed[stage][1]);
        result.shared[stage] = range(shared[stage][0], shared[stage][1]);
        if (result.routed[stage] >= 2) return null;
    }
    if (range(ex.k2_gu[0], ex.k2_gu[1]) == result.routed[0] and range(ex.k2_d[0], ex.k2_d[1]) == result.routed[1]) return null;
    return result;
}

fn raggedModuleRangeAllowed(range16: bool, split: bool, index: usize) bool {
    return raggedRangeAllowed(range16 or split, index);
}

/// Only pair-list ownership and grid.z change. In particular P, SK, WK,
/// maxm, slots, Z's projection-plane stride and all rotations are untouched.
fn splitMmaLaunch(original: PromptLaunch, index: usize, pairs: usize, offset: usize, ragged: bool) PromptLaunch {
    var l = original;
    l.range = index;
    l.cfg.grid.z = @intCast(pairs * @as(usize, if (l.kernel == .gate_up) 2 else 1));
    const pairs_at = l.args.offsets[17];
    std.mem.writeInt(u32, l.args.storage[pairs_at..][0..4], @intCast(pairs), .little);
    const work = l.args.integer(18).? + offset * 2 * @sizeOf(i32);
    const work_at = l.args.offsets[18];
    std.mem.writeInt(u64, l.args.storage[work_at..][0..8], work, .little);
    if (ragged) l.cfg = raggedConfig(l.cfg, index);
    return l;
}

fn partitionLaunch(ex: weights.Experts, sc: Scratch, r: usize, ls: *const [8]PromptLaunch) PromptLaunch {
    var a: cuda.Args = .{};
    a.add(sc.work_gu);
    i32a(&a, ls[4].args.integer(17).?);
    a.add(sc.work_d);
    i32a(&a, ls[6].args.integer(17).?);
    for ([_]u64{ sc.ids, sc.count, sc.counts, sc.members }) |v| a.add(v);
    for ([_]usize{ @min(r * sc.slots, ex.count), memberStride(r), ex.count - 1, r, sc.slots }) |v| i32a(&a, v);
    // One 256-thread CTA owns each list. Dynamic storage is one bit per
    // pair, rounded to whole warp ballots, sized for the larger original list.
    const n = @max(ls[4].args.integer(17).?, ls[6].args.integer(17).?);
    var l = promptLaunch(.work_list, .{ 2, 1, 1 }, 256, a);
    l.cfg.shared = @intCast((n + 31) / 32 * @sizeOf(u32));
    return l;
}

const SplitKey = struct {
    routed: [2]usize,
    shared: [2]usize,
    original: [2]usize,

    // Two routed W ranges and three shared/reference ranges for each stage.
    const capacity = 2 * 2 * 3 * 3 * 3 * 3;

    fn init(ranges: SplitRanges, ls: *const [8]PromptLaunch) SplitKey {
        return .{ .routed = ranges.routed, .shared = ranges.shared, .original = .{ ls[4].range, ls[6].range } };
    }

    fn index(key: SplitKey) usize {
        var result: usize = 0;
        for (key.routed) |r| {
            std.debug.assert(r < 2);
            result = result * 2 + r;
        }
        for (key.shared ++ key.original) |r| {
            std.debug.assert(r < 3);
            result = result * 3 + r;
        }
        return result;
    }
};
// Qualification belongs to the loaded kernel/range tuple, not a layer or
// prompt shape. Candidate/module load and unload reset this bounded table.
var split_checked: [SplitKey.capacity]bool = @splat(false);

fn splitNeedsReference(checked: bool, capturing: bool) bool {
    return !checked and capturing;
}

fn splitPrompt(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: Scratch, r: usize, ls: *[8]PromptLaunch, fs: [8]cuda.Function, ranges: SplitRanges) !void {
    const key = SplitKey.init(ranges, ls);
    const checked = split_checked[key.index()];
    // Do not allocate, synchronize, poison, or partition an unchecked graph.
    // Such a graph stays the original eight launches until it is recreated.
    if (splitNeedsReference(checked, try cuda.graph.captureStatus(s) != .none)) {
        for (ls, fs) |*l, f| try cuda.launch.launch(f, l.cfg, s, &l.args);
        return;
    }
    const partition = k.shared_partition orelse return error.MissingSharedPartition;
    var routed: [2]PromptLaunch = undefined;
    var shared: [2]PromptLaunch = undefined;
    var routed_fn: [2]cuda.Function = undefined;
    var shared_fn: [2]cuda.Function = undefined;
    const ns = (r + mma_rows_gu - 1) / mma_rows_gu;
    // Resolve both stages before the first launch. The full worklist includes
    // inactive padding, so its routed prefix remains nonempty for this shape.
    for ([_]usize{ 4, 6 }, 0..) |at, stage| {
        std.debug.assert(ls[at].kernel == .gate_up or ls[at].kernel == .down_mma);
        const n: usize = @intCast(ls[at].args.integer(17).?);
        if (n <= ns) return error.BadSharedPartition;
        const nr = n - ns;
        routed[stage] = splitMmaLaunch(ls[at], ranges.routed[stage], nr, 0, true);
        shared[stage] = splitMmaLaunch(ls[at], ranges.shared[stage], ns, nr, false);
        routed_fn[stage] = (if (stage == 0) k.ragged_gu[ranges.routed[stage]] else k.ragged_down[ranges.routed[stage]]) orelse return error.MissingRaggedKernel;
        shared_fn[stage] = (if (stage == 0) k.mma[ranges.shared[stage]] else k.mma_down[ranges.shared[stage]]) orelse return error.MissingKernel;
    }
    for (0..4) |at| try cuda.launch.launch(fs[at], ls[at].cfg, s, &ls[at].args);
    var part = partitionLaunch(ex, sc, r, ls);
    try cuda.launch.launch(partition, part.cfg, s, &part.args);
    for ([_]usize{ 4, 6 }, 0..) |at, stage| {
        const n = if (stage == 0) 2 * @as(usize, ex.width) else @as(usize, ex.dims);
        try splitStage(s, &ls[at], fs[at], &routed[stage], routed_fn[stage], &shared[stage], shared_fn[stage], checked, r * sc.slots * n * 4, sc.z, stage);
        // Original epilogue once, original down_combine once, with every
        // argument, slot address, rounding point and sum order unchanged.
        try cuda.launch.launch(fs[at + 1], ls[at + 1].cfg, s, &ls[at + 1].args);
    }
    if (!checked) split_checked[key.index()] = true;
}

fn splitStage(s: cuda.Stream, original: *PromptLaunch, original_fn: cuda.Function, routed: *PromptLaunch, routed_fn: cuda.Function, shared: *PromptLaunch, shared_fn: cuda.Function, checked: bool, bytes: usize, z: u64, stage: usize) !void {
    const a = std.heap.page_allocator;
    var ref: []u8 = &.{};
    var actual: []u8 = &.{};
    defer a.free(ref);
    defer a.free(actual);
    if (!checked) {
        ref = try a.alloc(u8, bytes);
        actual = try a.alloc(u8, bytes);
        // Compare against the OLD all-expert template, not a narrowed control.
        // Partition only permutes independent per-slot work in this reference.
        try cuda.launch.launch(original_fn, original.cfg, s, &original.args);
        try s.synchronize();
        try s.d.check(s.d.api.cuMemcpyDtoH_v2(ref.ptr, z, bytes), "shared split original");
        raggedPoison(ref, actual);
        try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(z, actual.ptr, bytes, s.handle), "shared split poison");
        try s.synchronize(); // pageable poison upload must finish before reuse/free
    }
    try cuda.launch.launch(routed_fn, routed.cfg, s, &routed.args);
    try cuda.launch.launch(shared_fn, shared.cfg, s, &shared.args);
    if (!checked) {
        try s.synchronize();
        try s.d.check(s.d.api.cuMemcpyDtoH_v2(actual.ptr, z, bytes), "shared split candidate");
        if (!std.mem.eql(u8, ref, actual)) return error.RaggedMmaNumericalMismatch;
        std.debug.print("prefill_shared_split_check: rank={d} stage={d} routed_range={d} shared_range={d} bytes={d} exact_fp32=PASS\n", .{ prompt_ragged_rank, stage, routed.range, shared.range, bytes });
    }
}

pub var prompt_ragged_mma = false;
pub var prompt_ragged_range16 = false;
pub var prompt_ragged_rank: u32 = 0;
var ragged_checked: [2][3]bool = @splat(@splat(false));

fn raggedRangeAllowed(range16: bool, index: usize) bool {
    return index < 2 or (range16 and index == 2);
}

fn raggedNeedsReference(checked: bool, capturing: bool) bool {
    return !checked and capturing;
}

fn raggedPoison(ref: []const u8, actual: []u8) void {
    for (ref, actual) |byte, *out| out.* = ~byte;
}

pub fn raggedEligible(enabled: bool, r: usize, d: usize, i: usize, slots: usize) bool {
    return enabled and r >= 64 and d == 5120 and i == 768 and slots > 0 and slots <= 32;
}

fn raggedLaunch(s: cuda.Stream, l: *PromptLaunch, original: cuda.Function, candidate: cuda.Function, stage: usize, p: usize, n: usize, z: u64) !void {
    const cfg = raggedConfig(l.cfg, l.range);
    if (!ragged_checked[stage][l.range]) {
        // An unchecked capture records only the reference, for every range: the
        // check synchronizes, which invalidates a capture (CUDA 900). Decode
        // graphs reach this with >= 64 MTP rows at C16. Eager use qualifies first.
        if (raggedNeedsReference(false, try cuda.graph.captureStatus(s) != .none))
            return cuda.launch.launch(original, l.cfg, s, &l.args);
        const bytes = p * n * 4;
        const a = std.heap.page_allocator;
        const ref = try a.alloc(u8, bytes);
        defer a.free(ref);
        const actual = try a.alloc(u8, bytes);
        defer a.free(actual);
        try cuda.launch.launch(original, l.cfg, s, &l.args);
        try s.synchronize();
        try s.d.check(s.d.api.cuMemcpyDtoH_v2(ref.ptr, z, bytes), "ragged original");
        if (l.range == 2) {
            // A missing store must not inherit the reference and pass. Fence the
            // pageable host buffer before launching or freeing it on an error.
            raggedPoison(ref, actual);
            try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(z, actual.ptr, bytes, s.handle), "ragged poison");
            try s.synchronize();
        }
        try cuda.launch.launch(candidate, cfg, s, &l.args);
        try s.synchronize();
        try s.d.check(s.d.api.cuMemcpyDtoH_v2(actual.ptr, z, bytes), "ragged candidate");
        if (!std.mem.eql(u8, ref, actual)) return error.RaggedMmaNumericalMismatch;
        ragged_checked[stage][l.range] = true;
        std.debug.print("prefill_ragged_mma_check: rank={d} stage={d} range={d} bytes={d} exact_fp32=PASS\n", .{ prompt_ragged_rank, stage, l.range, bytes });
    } else try cuda.launch.launch(candidate, cfg, s, &l.args);
}

test "prefill128 W range16 is independently gated for both stages" {
    for (0..4) |index| {
        try std.testing.expectEqual(index < 2, raggedRangeAllowed(false, index));
        try std.testing.expectEqual(index < 3, raggedRangeAllowed(true, index));
    }
    // The live quant's shared expert is K2=12, not its routed experts' 6/8.
    try std.testing.expectEqual(@as(usize, 2), range(8, 12));
    try std.testing.expectEqual(@as(usize, 2), range(6, 12));
    try std.testing.expectEqual(@as(usize, 2), range(2, 16));
    const saved = ragged_checked;
    defer ragged_checked = saved;
    ragged_checked = @splat(@splat(false));
    for (0..2) |stage| {
        ragged_checked[stage][2] = true;
        try std.testing.expect(!ragged_checked[stage][0] and !ragged_checked[stage][1]);
    }
}

test "prefill128 W capture stays reference until eager qualification" {
    for ([_]bool{ false, true }) |checked| {
        for ([_]bool{ false, true }) |capturing| {
            try std.testing.expectEqual(!checked and capturing, raggedNeedsReference(checked, capturing));
        }
    }
}

test "prefill128 W range16 poison rejects every stale reference byte" {
    var ref: [256]u8 = undefined;
    var actual: [256]u8 = undefined;
    for (&ref, 0..) |*byte, i| byte.* = @intCast(i);
    raggedPoison(&ref, &actual);
    for (ref, actual) |expected, poisoned| try std.testing.expect(expected != poisoned);
}

test "prefill128 W ragged MoE scope preserves decode and other shapes" {
    try std.testing.expect(raggedEligible(true, 2048, 5120, 768, 7));
    try std.testing.expect(raggedEligible(true, 64, 5120, 768, 1));
    try std.testing.expect(!raggedEligible(false, 2048, 5120, 768, 7));
    try std.testing.expect(!raggedEligible(true, 63, 5120, 768, 7));
    try std.testing.expect(!raggedEligible(true, 2048, 5120, 1152, 7));
    try std.testing.expect(!raggedEligible(true, 2048, 4096, 768, 7));
    try std.testing.expect(!raggedEligible(true, 2048, 5120, 768, 0));
    try std.testing.expect(!raggedEligible(true, 2048, 5120, 768, 33));
}

// Only HI16's weight ring is dynamic: stages * KB * 8 warps * 4*HI words.
const ragged_range16_shared: u32 = 3 * 4 * 8 * (4 * 16) * @sizeOf(u32);

// W doubles the output tile width, without changing any kernel arguments.
fn raggedConfig(original: cuda.Config, index: usize) cuda.Config {
    var cfg = original;
    cfg.block.x = 256;
    cfg.grid.y /= 2;
    if (index == 2) cfg.shared = ragged_range16_shared;
    return cfg;
}

test "prefill128 W changes only the prompt MMA tile geometry" {
    const original: cuda.Config = .{ .grid = .{ .x = 1, .y = 80, .z = 19 }, .block = .{ .x = 256 } };
    const candidate = raggedConfig(original, 0);
    try std.testing.expectEqual(@as(u32, 40), candidate.grid.y);
    try std.testing.expectEqual(original.grid.x, candidate.grid.x);
    try std.testing.expectEqual(original.grid.z, candidate.grid.z);
    try std.testing.expectEqual(original.block, candidate.block);
    try candidate.validate();
    try std.testing.expectEqual(candidate, raggedConfig(original, 1));
    const hi16 = raggedConfig(original, 2);
    try std.testing.expectEqual(@as(u32, 24576), hi16.shared);
    try std.testing.expectEqual(@as(u32, 52480), hi16.shared + 3 * 64 * 72 * 2 + 64 * 4);
    var same = hi16;
    same.shared = candidate.shared;
    try std.testing.expectEqual(candidate, same);
    try hi16.validate();
}

test "prefill128 shared split embedded module and fingerprint are pinned" {
    var symbols = elf.Symbols.init(&shared_split_image).?;
    var found: usize = 0;
    while (symbols.next()) |name| {
        if (std.mem.eql(u8, name, "tf_shared_split_partition")) found += 1;
        try std.testing.expect(std.mem.indexOf(u8, name, "grouped_mma") == null);
    }
    try std.testing.expectEqual(@as(usize, 1), found);
    const base = "original;prefill128_w=unchanged";
    const off = try sharedSplitBuildFlags(std.testing.allocator, base, false);
    try std.testing.expectEqual(base.ptr, off.ptr);
    try std.testing.expectEqualStrings(base, off);
    const on = try sharedSplitBuildFlags(std.testing.allocator, base, true);
    defer std.testing.allocator.free(on);
    try std.testing.expectEqualStrings(base ++ ";prefill128_shared_split=abebbd98922287a63c2139d77fb9718ebd160402b95ebeacf468ccb09c348c28", on);
}

fn splitTestExperts() weights.Experts {
    return .{ .trellis = 100, .gate_ptr = 101, .up_ptr = 102, .down_ptr = 103, .gate_k2 = 104, .up_k2 = 105, .down_k2 = 106, .suh_g = 107, .suh_u = 108, .svh_g = 109, .svh_u = 110, .suh_d = 111, .svh_d = 112, .count = 385, .dims = 5120, .width = 768, .down_k = 768, .down_n = 5120, .k2_gu = .{ 6, 12 }, .k2_d = .{ 8, 12 }, .routed_k2_gu = .{ 6, 8 }, .routed_k2_d = .{ 8, 8 }, .shared_k2_gu = .{ 12, 12 }, .shared_k2_d = .{ 12, 12 } };
}

fn splitTestScratch() Scratch {
    return .{ .rows = 8192, .slots = 7, .xg = 201, .xu = 202, .xd = 203, .z = 204, .no_y = 205, .ids = 206, .count = 207, .counts = 208, .members = 209, .work_gu = 210, .work_d = 211 };
}

test "prefill128 shared split keeps full slot ABI and flag-off sequence" {
    const ex = splitTestExperts();
    const sc = splitTestScratch();
    const expected = [_]@FieldType(PromptLaunch, "kernel"){ .group_count, .group_place, .work_list, .rot_in, .gate_up, .gateup_epilogue, .down_mma, .down_combine };
    for ([_]usize{ 64, 65, 127, 128, 129, 1376, 2720, 2736, 8192 }) |r| {
        var ls = try promptLaunches(ex, sc, 301, 5120, 302, 303, 304, r, 10);
        // This is the exact selector prompt() calls before touching its plan.
        try std.testing.expectEqual(@as(?SplitRanges, null), sharedSplitRanges(false, ex, sc.slots));
        for (ls, expected) |l, kind| try std.testing.expectEqual(kind, l.kernel);
        const ranges = sharedSplitRanges(true, ex, sc.slots).?;
        try std.testing.expectEqual([2]usize{ 1, 0 }, ranges.routed);
        try std.testing.expectEqual([2]usize{ 2, 2 }, ranges.shared);
        const ns = (r + 63) / 64;
        for ([_]usize{ 4, 6 }, 0..) |at, stage| {
            const n: usize = @intCast(ls[at].args.integer(17).?);
            const nr = n - ns;
            const routed = splitMmaLaunch(ls[at], ranges.routed[stage], nr, 0, true);
            const shared = splitMmaLaunch(ls[at], ranges.shared[stage], ns, nr, false);
            for ([_]PromptLaunch{ routed, shared }, 0..) |candidate, which| {
                try candidate.cfg.validate();
                try std.testing.expectEqual(ls[at].args.count, candidate.args.count);
                for (0..17) |arg| {
                    try std.testing.expectEqual(ls[at].args.integer(arg), candidate.args.integer(arg));
                    try std.testing.expectEqual(ls[at].args.sizes[arg], candidate.args.sizes[arg]);
                }
                try std.testing.expectEqual(@as(u64, r * sc.slots), candidate.args.integer(12).?);
                try std.testing.expectEqual(@as(u64, sc.slots), candidate.args.integer(16).?);
                const pairs = if (which == 0) nr else ns;
                try std.testing.expectEqual(@as(u64, pairs), candidate.args.integer(17).?);
                try std.testing.expectEqual(ls[at].args.integer(18).? + @as(u64, if (which == 0) 0 else nr * 8), candidate.args.integer(18).?);
                try std.testing.expectEqual(pairs * @as(usize, if (stage == 0) 2 else 1), candidate.cfg.grid.z);
                try std.testing.expectEqual(ls[at].cfg.grid.y / @as(u32, if (which == 0) 2 else 1), candidate.cfg.grid.y);
                try std.testing.expectEqual(ls[at].cfg.shared, candidate.cfg.shared);
            }
        }
        // The partition sees the ORIGINAL full-list lengths and member stride.
        const partition = partitionLaunch(ex, sc, r, &ls);
        try partition.cfg.validate();
        try std.testing.expectEqual(@as(u32, 2), partition.cfg.grid.x);
        try std.testing.expectEqual(@as(u32, 256), partition.cfg.block.x);
        const max_pairs = @max(ls[4].args.integer(17).?, ls[6].args.integer(17).?);
        try std.testing.expectEqual((max_pairs + 31) / 32 * 4, partition.cfg.shared);
        try std.testing.expectEqual(ls[4].args.integer(17), partition.args.integer(1));
        try std.testing.expectEqual(ls[6].args.integer(17), partition.args.integer(3));
        try std.testing.expectEqual(@as(u64, memberStride(r)), partition.args.integer(9).?);
        try std.testing.expectEqual(@as(u64, ex.count - 1), partition.args.integer(10).?);
        try std.testing.expectEqual(@as(u64, r), partition.args.integer(11).?);
        // Rebuilding the legacy plan is byte-for-byte equal for every USED
        // argument (Args contains intentionally undefined padding/ptr arrays).
        const again = try promptLaunches(ex, sc, 301, 5120, 302, 303, 304, r, 10);
        for (&ls, again) |*before, after| {
            try std.testing.expectEqual(before.kernel, after.kernel);
            try std.testing.expectEqual(before.cfg, after.cfg);
            try std.testing.expectEqual(before.range, after.range);
            for (0..before.args.count) |arg| {
                const off = before.args.offsets[arg];
                const size = before.args.sizes[arg];
                try std.testing.expectEqualSlices(u8, before.args.storage[off..][0..size], after.args.storage[after.args.offsets[arg]..][0..size]);
            }
        }
    }
}

test "prefill128 shared split bitmap storage covers both lists and maximum slots" {
    const ex = splitTestExperts();
    var sc = splitTestScratch();
    sc.slots = 32;
    const ls = try promptLaunches(ex, sc, 301, 5120, 302, 303, 304, 8192, 10);
    try std.testing.expectEqual(@as(u64, 4474), ls[4].args.integer(17).?);
    const part = partitionLaunch(ex, sc, 8192, &ls);
    try std.testing.expectEqual(@as(u32, 560), part.cfg.shared);
    try part.cfg.validate();
    // Size for the larger list, even if a future caller supplies unequal lists.
    var unequal = ls;
    const offset = unequal[6].args.offsets[17];
    std.mem.writeInt(u32, unequal[6].args.storage[offset..][0..4], 8193, .little);
    try std.testing.expectEqual(@as(u32, 1028), partitionLaunch(ex, sc, 8192, &unequal).cfg.shared);
}

test "prefill128 shared split capture and qualification keys cover every candidate" {
    for ([_]bool{ false, true }) |checked| for ([_]bool{ false, true }) |capturing| {
        try std.testing.expectEqual(!checked and capturing, splitNeedsReference(checked, capturing));
    };
    var checked: [SplitKey.capacity]bool = @splat(false);
    // Exhaust every stage/range tuple: no candidate can alias another entry.
    for (0..2) |rg| for (0..2) |rd| {
        for (0..3) |sg| for (0..3) |sd| {
            for (0..3) |og| for (0..3) |od| {
                const key: SplitKey = .{ .routed = .{ rg, rd }, .shared = .{ sg, sd }, .original = .{ og, od } };
                try std.testing.expect(!checked[key.index()]);
                try std.testing.expect(splitNeedsReference(checked[key.index()], true));
                checked[key.index()] = true;
                try std.testing.expect(!splitNeedsReference(checked[key.index()], true));
            };
        };
    };
    for (checked) |qualified| try std.testing.expect(qualified);
    // A measured HI8/HI10 cubin OR one with extra HI16 templates is accepted
    // when split is on, without enabling unsplit range16 dispatch.
    try std.testing.expect(raggedModuleRangeAllowed(false, true, 0));
    try std.testing.expect(raggedModuleRangeAllowed(false, true, 1));
    try std.testing.expect(raggedModuleRangeAllowed(false, true, 2));
    try std.testing.expect(!raggedModuleRangeAllowed(false, false, 2));
    try std.testing.expect(!raggedRangeAllowed(false, 2));
    var ex = splitTestExperts();
    ex.routed_k2_gu = .{ 0, 0 };
    try std.testing.expect(sharedSplitRanges(true, ex, 7) == null);
    ex = splitTestExperts();
    ex.routed_k2_d = .{ 6, 12 };
    try std.testing.expect(sharedSplitRanges(true, ex, 7) == null);
    ex = splitTestExperts();
    ex.k2_gu = .{ 8, 8 };
    ex.k2_d = .{ 8, 8 };
    ex.routed_k2_gu = .{ 8, 8 };
    try std.testing.expect(sharedSplitRanges(true, ex, 7) == null);
}

test "prefill128 shared split reuses qualification across layers and shapes" {
    const first_ex = splitTestExperts();
    const first_sc = splitTestScratch();
    const first_ls = try promptLaunches(first_ex, first_sc, 301, 5120, 302, 303, 304, 64, 10);
    const first = SplitKey.init(sharedSplitRanges(true, first_ex, first_sc.slots).?, &first_ls);
    var checked: [SplitKey.capacity]bool = @splat(false);
    checked[first.index()] = true;

    // Change all formerly keyed layer pointers and shape fields. Only the
    // stage/range identities survive, so this shape needs no new host copies.
    var ex = first_ex;
    ex.trellis += 1000;
    ex.gate_ptr += 1000;
    ex.up_ptr += 1000;
    ex.down_ptr += 1000;
    ex.count = 193;
    var sc = first_sc;
    sc.rows = 4096;
    sc.slots = 4;
    const ls = try promptLaunches(ex, sc, 401, 5120, 402, 403, 404, 4096, 10);
    const second = SplitKey.init(sharedSplitRanges(true, ex, sc.slots).?, &ls);
    try std.testing.expectEqual(first, second);
    try std.testing.expect(checked[second.index()]);
    try std.testing.expect(!splitNeedsReference(checked[second.index()], true));

    // Each stage of every kernel family must still qualify independently.
    for (0..6) |change| {
        var other = second;
        switch (change) {
            0, 1 => other.routed[change] = 1 - other.routed[change],
            2, 3 => other.shared[change - 2] = (other.shared[change - 2] + 1) % 3,
            else => other.original[change - 4] = (other.original[change - 4] + 1) % 3,
        }
        try std.testing.expect(!checked[other.index()]);
        try std.testing.expect(splitNeedsReference(checked[other.index()], true));
    }
}

// Exact model of the kernel's ballot bitmap and serial swap owner. The
// legacy oracle below intentionally keeps the original pair-by-pair loop.
fn partitionPairs(pairs: [][2]i32, ids: []const u32, shared_id: u32, expected_shared: usize) !usize {
    const bits = try std.testing.allocator.alloc(u32, (pairs.len + 31) / 32);
    defer std.testing.allocator.free(bits);
    @memset(bits, 0);
    for (pairs, 0..) |pair, i| {
        if (pair[0] < 0 or (pair[0] < ids.len and pair[1] < 0)) return error.BadSharedPartition;
        const u: usize = @intCast(pair[0]);
        if (u < ids.len and ids[u] == shared_id) bits[i / 32] |= @as(u32, 1) << @as(u5, @intCast(i % 32));
    }
    var left: usize = 0;
    var right = pairs.len;
    while (left < right) {
        const word = bits[left / 32] & (@as(u32, std.math.maxInt(u32)) << @as(u5, @intCast(left % 32)));
        if (word == 0) {
            left = (left | 31) + 1;
            continue;
        }
        left = (left & ~@as(usize, 31)) + @ctz(word);
        if (left >= right) break;
        right -= 1;
        if (left == right) break;
        const right_shared = bits[right / 32] & (@as(u32, 1) << @as(u5, @intCast(right % 32))) != 0;
        std.mem.swap([2]i32, &pairs[left], &pairs[right]);
        if (!right_shared) bits[left / 32] &= ~(@as(u32, 1) << @as(u5, @intCast(left % 32)));
    }
    if (pairs.len - right != expected_shared) return error.BadSharedPartition;
    return right;
}

// All of these checks run for each list CTA on every call, not only first-use
// qualification. CPU errors model the device trap, not recovery/fallback.
fn partitionContract(ids: []const u32, used: usize, maxu: usize, counts: []const usize, members: []const i32, maxm: usize, shared_id: usize, rows: usize, slots: usize) !void {
    if (used == 0 or used > maxu or counts[shared_id] != rows) return error.BadSharedPartition;
    var shared_u: ?usize = null;
    for (ids[0..used], 0..) |id, u| if (id == shared_id) {
        if (shared_u != null) return error.BadSharedPartition;
        shared_u = u;
    };
    const u = shared_u orelse return error.BadSharedPartition;
    for (0..rows) |row| if (members[u * maxm + row] != row * 32 + slots - 1) return error.BadSharedPartition;
}

fn partitionPairsLegacy(pairs: [][2]i32, ids: []const u32, shared_id: u32, expected_shared: usize) !usize {
    for (pairs) |pair| if (pair[0] < 0 or (pair[0] < ids.len and pair[1] < 0)) return error.BadSharedPartition;
    var left: usize = 0;
    var right = pairs.len;
    while (left < right) {
        const u: usize = @intCast(pairs[left][0]);
        if (u >= ids.len or ids[u] != shared_id) {
            left += 1;
            continue;
        }
        right -= 1;
        if (left == right) break;
        std.mem.swap([2]i32, &pairs[left], &pairs[right]);
    }
    if (pairs.len - right != expected_shared) return error.BadSharedPartition;
    return right;
}

test "prefill128 shared split bitmap exactly matches every legacy permutation" {
    // Exhaust every classification through length 12, using distinct pair
    // payloads so equality proves order, not only classification or slot sets.
    var original: [12][2]i32 = undefined;
    var legacy: [12][2]i32 = undefined;
    var actual: [12][2]i32 = undefined;
    for (0..13) |n| for (0..(@as(usize, 1) << @as(u6, @intCast(n)))) |mask| {
        var shared_count: usize = 0;
        for (original[0..n], 0..) |*pair, i| {
            const shared = mask & (@as(usize, 1) << @as(u6, @intCast(i))) != 0;
            pair.* = .{ @intFromBool(shared), @intCast(i) };
            shared_count += @intFromBool(shared);
        }
        @memcpy(legacy[0..n], original[0..n]);
        @memcpy(actual[0..n], original[0..n]);
        const expected = try partitionPairsLegacy(legacy[0..n], &.{ 0, 1 }, 1, shared_count);
        try std.testing.expectEqual(expected, try partitionPairs(actual[0..n], &.{ 0, 1 }, 1, shared_count));
        try std.testing.expectEqualSlices([2]i32, legacy[0..n], actual[0..n]);
    };
    // Warp/CTA seams, partial last ballot, all-shared/routed, and the real
    // 34.3K-row shared count. Include padding with a negative ignored mb.
    var prng = std.Random.DefaultPrng.init(0x4249544d4150);
    const random = prng.random();
    for ([_]usize{ 31, 32, 33, 63, 64, 65, 255, 256, 257, 4130, 17529 }) |n| {
        const pairs = try std.testing.allocator.alloc([2]i32, n);
        defer std.testing.allocator.free(pairs);
        const reference = try std.testing.allocator.alloc([2]i32, n);
        defer std.testing.allocator.free(reference);
        for ([_]usize{ 0, 1, @min(n, 536), n }) |shared_count| {
            for (0..8) |_| {
                for (pairs, 0..) |*pair, i| pair.* = if (i < shared_count) .{ 1, @intCast(i) } else if (i % 3 == 0) .{ std.math.maxInt(i32), -1 } else .{ 0, @intCast(i) };
                random.shuffle([2]i32, pairs);
                @memcpy(reference, pairs);
                const expected = try partitionPairsLegacy(reference, &.{ 0, 1 }, 1, shared_count);
                try std.testing.expectEqual(expected, try partitionPairs(pairs, &.{ 0, 1 }, 1, shared_count));
                try std.testing.expectEqualSlices([2]i32, reference, pairs);
            }
        }
    }
    for ([_][2]i32{ .{ -1, 0 }, .{ 0, -1 }, .{ 1, -1 } }) |bad| {
        var pairs = [_][2]i32{ .{ 1, 0 }, .{ 0, 0 }, bad };
        try std.testing.expectError(error.BadSharedPartition, partitionPairs(&pairs, &.{ 0, 1 }, 1, 1));
    }
}

test "prefill128 shared split retains every-call router contract checks" {
    var members = [_]i32{ -1, -1, 6, 38 };
    try partitionContract(&.{ 0, 1 }, 2, 2, &.{ 0, 2 }, &members, 2, 1, 2, 7);
    try std.testing.expectError(error.BadSharedPartition, partitionContract(&.{ 0, 1 }, 0, 2, &.{ 0, 2 }, &members, 2, 1, 2, 7));
    try std.testing.expectError(error.BadSharedPartition, partitionContract(&.{ 0, 1 }, 2, 1, &.{ 0, 2 }, &members, 2, 1, 2, 7));
    try std.testing.expectError(error.BadSharedPartition, partitionContract(&.{ 0, 1 }, 2, 2, &.{ 0, 1 }, &members, 2, 1, 2, 7));
    try std.testing.expectError(error.BadSharedPartition, partitionContract(&.{ 0, 0 }, 2, 2, &.{ 0, 2 }, &members, 2, 1, 2, 7));
    try std.testing.expectError(error.BadSharedPartition, partitionContract(&.{ 1, 1 }, 2, 2, &.{ 0, 2 }, &members, 2, 1, 2, 7));
    for (0..2) |row| {
        members[2 + row] += 1;
        try std.testing.expectError(error.BadSharedPartition, partitionContract(&.{ 0, 1 }, 2, 2, &.{ 0, 2 }, &members, 2, 1, 2, 7));
        members[2 + row] -= 1;
    }
}

test "prefill128 shared split random routings cover every slot exactly once" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x53504c4954);
    const random = prng.random();
    for ([_]usize{ 64, 65, 127, 128, 129, 1376, 2720, 2736, 8192 }) |rows| {
        for ([_]usize{ 2, 4, 7, 32 }) |slots| {
            const e = 41; // includes empty experts at smaller routing counts
            const p = rows * slots;
            const maxm = memberStride(rows);
            var ids: [e]u32 = undefined;
            for (&ids, 0..) |*id, j| id.* = @intCast(j);
            random.shuffle(u32, &ids); // shared's place must not be assumed
            const members = try a.alloc(i32, e * maxm);
            defer a.free(members);
            @memset(members, -1);
            var counts: [e]usize = @splat(0);
            var choices: [e - 1]u32 = undefined;
            for (&choices, 0..) |*id, j| id.* = @intCast(j);
            for (0..rows) |row| {
                random.shuffle(u32, &choices);
                for (0..slots) |slot| {
                    // Keep at most one assignment per expert per row; the
                    // shared expert always occupies the final original slot.
                    const expert = if (slot == slots - 1) e - 1 else choices[slot];
                    const u = std.mem.indexOfScalar(u32, &ids, expert).?;
                    members[u * maxm + counts[u]] = @intCast(row * 32 + slot);
                    counts[u] += 1;
                }
            }
            const pairs = try a.alloc([2]i32, listLen(p, e, 64));
            defer a.free(pairs);
            @memset(pairs, .{ std.math.maxInt(i32), 0 });
            var used: usize = 0;
            for (counts, 0..) |count, u| for (0..((count + 63) / 64)) |mb| {
                pairs[used] = .{ @intCast(u), @intCast(mb) };
                used += 1;
            };
            random.shuffle([2]i32, pairs);
            var expert_counts: [e]usize = @splat(0);
            for (ids, counts) |expert, count| expert_counts[expert] = count;
            try partitionContract(&ids, e, e, &expert_counts, members, maxm, e - 1, rows, slots);
            const reference = try a.dupe([2]i32, pairs);
            defer a.free(reference);
            const expected_nr = try partitionPairsLegacy(reference, &ids, e - 1, (rows + 63) / 64);
            const nr = try partitionPairs(pairs, &ids, e - 1, (rows + 63) / 64);
            try std.testing.expectEqual(expected_nr, nr);
            try std.testing.expectEqualSlices([2]i32, reference, pairs);
            const seen = try a.alloc(u8, p);
            defer a.free(seen);
            @memset(seen, 0);
            for (pairs, 0..) |pair, pair_index| {
                const u: usize = @intCast(pair[0]);
                if (u >= e) {
                    try std.testing.expect(pair_index < nr);
                    continue;
                }
                const is_shared_slot = ids[u] == e - 1;
                try std.testing.expectEqual(is_shared_slot, pair_index >= nr);
                const start: usize = @as(usize, @intCast(pair[1])) * 64;
                for (start..@min(start + 64, maxm)) |m| {
                    const code = members[u * maxm + m];
                    if (code < 0) continue;
                    const slot: usize = @intCast(code & 31);
                    const flat: usize = @as(usize, @intCast(code >> 5)) * slots + slot;
                    try std.testing.expect(flat < p);
                    try std.testing.expectEqual(is_shared_slot, slot == slots - 1);
                    seen[flat] += 1;
                    // GU's two planes and down use the same full P, not a
                    // compacted routed stride or a separately added shared sum.
                    for (0..2) |mat| try std.testing.expect((mat * p + flat) * 768 < 2 * p * 768);
                }
            }
            for (seen) |count| try std.testing.expectEqual(@as(u8, 1), count);
        }
    }
    var malformed = [_][2]i32{ .{ -1, 0 }, .{ 0, 0 } };
    try std.testing.expectError(error.BadSharedPartition, partitionPairs(&malformed, &.{0}, 0, 1));
    malformed = .{ .{ 0, 0 }, .{ 0, 1 } };
    try std.testing.expectError(error.BadSharedPartition, partitionPairs(&malformed, &.{0}, 0, 1));
}
