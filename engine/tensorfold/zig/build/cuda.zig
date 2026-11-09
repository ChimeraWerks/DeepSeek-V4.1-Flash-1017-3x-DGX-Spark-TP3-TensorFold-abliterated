//! The CUDA half of the root build: kernel fatbins with each Python extension's nvcc flags, the runtime, Nemotron, the CLI.

const std = @import("std");

/// Each .cu in zig/kernels/cuda (`src`, else `name`) with the flags its Python extension passes in `extra_cuda_cflags`.
/// opt_in: built only with its dedicated -Dport_kernels, -Dsample_cand or -Dkeyed_draft build option; absent from default fatbins.
const Kernel = struct { name: []const u8, flags: []const []const u8, src: ?[]const u8 = null, arch_specific: bool = false, opt_in: bool = false };

/// The torch-op replacements' qualification flags (runs/006): no contraction, no flush to zero.
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

const kernels = [_]Kernel{
    .{ .name = "gdn", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "probe", .flags = &.{"-O3"} },
    .{ .name = "qmm_group", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "qmm_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts", .flags = &.{"-O3"} }, // cuda/experts.py, tensorfold_experts_v7
    .{ .name = "experts_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts_pack", .flags = &.{"-O3"} },
    .{ .name = "prefill_attention", .flags = &.{ "-O3", "--fmad=false" } }, // tensorfold_prefill_attention_v1
    .{ .name = "scan_rows", .flags = &.{ "-O3", "--fmad=false" } }, // nemotron_h/cuda/mamba.py
    .{ .name = "nemotron_ops", .flags = &.{"-O3"} }, // ours: Nemotron's layouts and the serial feed
    .{ .name = "lane_gemv", .flags = &.{"-O3"} }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    .{ .name = "sample", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the Metal engine's keyed draws
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
    .{ .name = "dsv41_ops", .flags = torch_ops }, // ours: the DeepSeek-V4.1 casts torch_ops/ lacks
    .{ .name = "dsv41_torch", .flags = torch_ops }, // ours: DeepSeek-V4.1 torch arithmetic at torch's rounding points
    .{ .name = "dsv41_shard", .flags = torch_ops }, // ours: TF_DS_KV_SHARD's fetch and select byte movers (GLM 0118 port)
    .{ .name = "dsv41_rdma", .src = "dsv41/rdma_gather", .flags = &.{"-O3"} }, // ours: the decode gathers' RDMA kernel pair
    .{ .name = "dsv41_port", .flags = torch_ops, .opt_in = true }, // ours: the GLM port's spill block staging (port_kernels.zig; fetch/select/union are dsv41_shard)
    .{ .name = "dsv41_sample_cand", .flags = torch_ops, .opt_in = true }, // sampled rows' bounded top candidates
    .{ .name = "dsv41_keyed", .flags = torch_ops, .opt_in = true }, // isolated keyed draft noise
};

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// core/stagger.zig, the segment schedule the Metal and CUDA runners share, as its own module (no imports).
fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images). An opt-in kernel's image
/// may be null (not built): `has_<name>` says whether it is embedded.
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    if (with) for (kernels, images) |k, i| {
        if (!k.opt_in and i == null) with = false;
    };
    options.addOption(bool, "with_kernels", with);
    for (kernels, 0..) |k, i| {
        if (k.opt_in) options.addOption(bool, b.fmt("has_{s}", .{k.name}), with and images[i] != null);
    }
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (with) for (kernels, images) |k, image| if (image) |file| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = file });
    return cuda;
}

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, tokenizer: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .tokenizer = tokenizer };
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (121; later 120,89)") orelse "121";
    const port_kernels = b.option(bool, "port_kernels", "also build and embed the GLM port's opt-in kernels (dsv41_port.cu: tf-dsv41-port, spill v2's one-launch staging)") orelse false;
    const sample_cand = b.option(bool, "sample_cand", "embed the opt-in dsv41_sample_cand kernel") orelse false;
    const keyed_draft = b.option(bool, "keyed_draft", "embed the opt-in dsv41_keyed proposal noise kernel") orelse false;
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and nvcc != null) blk: {
        const run = b.addSystemCommand(&.{ nvcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    for (kernels, &images) |k, *image| {
        if (k.opt_in and !(if (std.mem.eql(u8, k.name, "dsv41_sample_cand")) sample_cand else if (std.mem.eql(u8, k.name, "dsv41_keyed")) keyed_draft else port_kernels)) continue;
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    const cuda = runtime(b, target, optimize, if (nvcc != null or prebuilt != null) &images else &.{});
    const mods = family(b, target, optimize, cuda, draft_ids);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("cuda", cuda);
    runner.addImport("lanes", mods.lanes);
    b.installArtifact(b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner }));
    // DeepSeek-V4.1 (our port): the TP split plan against the lane's rank cache index, no GPU
    const ds_tools_step = b.step("dsv41-tools", "DeepSeek follower executable and CPU rank-cache checker");
    const ds_qualify_step = b.step("dsv41-qualify", "DeepSeek GPU load and TP communication qualification tools (build only)");
    const ds_check = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/check_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_check.addImport("dsv41", engines(b, target, optimize, cuda, mods.lanes, mods.nemotron, mods.core).dsv41);
    const ds_check_install = b.addInstallArtifact(b.addExecutable(.{ .name = "tf-dsv41-check", .root_module = ds_check }), .{});
    b.getInstallStep().dependOn(&ds_check_install.step);
    ds_tools_step.dependOn(&ds_check_install.step);
    const ds_load = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/load_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_load.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    ds_load.addImport("cuda", cuda);
    const ds_load_install = b.addInstallArtifact(b.addExecutable(.{ .name = "tf-dsv41-load", .root_module = ds_load }), .{});
    b.getInstallStep().dependOn(&ds_load_install.step);
    ds_qualify_step.dependOn(&ds_load_install.step);
    const ds_comm = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/comm_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_comm.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    ds_comm.addImport("cuda", cuda);
    const ds_comm_install = b.addInstallArtifact(b.addExecutable(.{ .name = "tf-dsv41-comm", .root_module = ds_comm }), .{});
    b.getInstallStep().dependOn(&ds_comm_install.step);
    ds_qualify_step.dependOn(&ds_comm_install.step);
    const ds_lifecycle = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/rdma_lifecycle_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_lifecycle.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    ds_lifecycle.addImport("cuda", cuda);
    const ds_lifecycle_install = b.addInstallArtifact(b.addExecutable(.{ .name = "tf-dsv41-rdma-lifecycle", .root_module = ds_lifecycle }), .{});
    b.step("dsv41-lifecycle", "Isolated RDMA/thread/Pillow lifecycle diagnostic, build only").dependOn(&ds_lifecycle_install.step);
    const ds_linear_probe = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/linear_probe_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_linear_probe.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    ds_linear_probe.addImport("cuda", cuda);
    const ds_linear_install = b.addInstallArtifact(b.addExecutable(.{ .name = "tf-dsv41-linear-probe", .root_module = ds_linear_probe }), .{});
    b.getInstallStep().dependOn(&ds_linear_install.step);
    ds_qualify_step.dependOn(&ds_linear_install.step);
    const ds_layer = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/layer_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_layer.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    ds_layer.addImport("cuda", cuda);
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-layer", .root_module = ds_layer }));
    // the RDMA ring across real nodes (rdma.zig): rendezvous, gathers at each row count, every float checked
    const ds_rdma = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/rdma_bench_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_rdma.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    ds_rdma.addImport("cuda", cuda);
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-rdma-bench", .root_module = ds_rdma }));
    // the Engram reader's buffered, O_DIRECT and AIO paths on the real tables, every byte compared
    const ds_engram = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/engram_check_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_engram.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-engram-check", .root_module = ds_engram }));
    // the vision tower's input of each image as the served lane makes it (picture.zig), by sha256 against the fixtures
    const ds_picture = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/picture_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_picture.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-picture", .root_module = ds_picture }));
    // the vision tower on recorded patches, each step's bytes against the served lane's (vit.zig)
    const ds_vit = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/vit_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_vit.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    ds_vit.addImport("cuda", cuda);
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-vit", .root_module = ds_vit }));
    // the 2D split's decode-size exchanges over the RDMA rings on four nodes, word for word against Comm2D's layout
    const ds_ring2d = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/ring2d_check_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_ring2d.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    ds_ring2d.addImport("cuda", cuda);
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-ring2d-check", .root_module = ds_ring2d }));
    // the lane gate: TensorFold 1.0s round loop serving recorded requests through the ports Backend on both ranks
    const ds_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/lanes_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_lanes.addImport("dsv41", ds_check.import_table.get("dsv41").?);
    ds_lanes.addImport("cuda", cuda);
    ds_lanes.addImport("lanes", mods.lanes);
    const ds_lanes_install = b.addInstallArtifact(b.addExecutable(.{ .name = "tf-dsv41-lanes", .root_module = ds_lanes }), .{});
    b.getInstallStep().dependOn(&ds_lanes_install.step);
    ds_tools_step.dependOn(&ds_lanes_install.step);
    // the GLM port's spill block staging kernel (dsv41_port.cu) against its CPU reference, one GPU
    const ds_port = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/port_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_port.addImport("cuda", cuda);
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-port", .root_module = ds_port }));
    _ = nativeServer(b, target, optimize, cuda, mods.lanes, mods.nemotron, mods.core, mods.tokenizer, build_options, true);
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, core: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module, dsv41: *std.Build.Module } {
    // DeepSeek-V4.1-Flash (our port): its module reads the runtime and the lane core, as Nemotron's does
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const dsv41 = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/dsv41.zig"), .target = target, .optimize = optimize, .link_libc = true });
    dsv41.addImport("cuda", cuda);
    dsv41.addImport("lanes", lanes);
    dsv41.addImport("core", core);
    // the prompt cache's Tier the spill tier implements (spill_tier.zig)
    dsv41.addImport("engine_api", api);
    // the RDMA ring's verbs (rdma.zig)
    dsv41.addImport("fabric", b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "nemotron", .module = nemotron }, .{ .name = "dsv41", .module = dsv41 } },
    });
    return .{ .api = api, .engines = mod, .dsv41 = dsv41 };
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, core: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options, install_native: bool) *std.Build.Step.Compile {
    const m = engines(b, target, optimize, cuda, lanes, nemotron, core);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = m.engines }} }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    if (!install_native) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin").dependOn(&install.step);
    return exe;
}

/// Host unit tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug, &.{});
    const mods = family(b, host, .debug, cuda, draft_ids);
    const eng = engines(b, host, .debug, cuda, mods.lanes, mods.nemotron, mods.core);
    b.step("test-kv-loopback", "Loopback fixture connection and worker lifetimes").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "loopback fixture", "over a loopback link:" } })).step);
    const draft_cal_tests = b.step("test-draft-calibration", "Draft survival, first depth, allocator and TP3 settings host regressions");
    draft_cal_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.lanes })).step);
    draft_cal_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{"draft settings"} })).step);
    const audit = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/aot_check_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    audit.addImport("cuda", cuda);
    audit.addImport("dsv41", eng.dsv41);
    const audit_run = b.addRunArtifact(b.addExecutable(.{ .name = "tf-dsv41-aot-check", .root_module = audit }));
    if (b.option([]const u8, "aot-manifest", "aot.json for the CPU-only launch audit")) |path| audit_run.addArg(path);
    const projection_shapes = b.option([]const u8, "projection-shapes", "native-plan projection shapes for the CPU launch audit");
    if (projection_shapes) |path| audit_run.addArg(path);
    b.step("tp3-aot-check", "CPU-only native launch matrix against aot.json").dependOn(&audit_run.step);
    const cuda_audit = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/cuda_check_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cuda_audit.addImport("cuda", cuda);
    cuda_audit.addImport("dsv41", eng.dsv41);
    const cuda_audit_run = b.addRunArtifact(b.addExecutable(.{ .name = "tf-dsv41-cuda-check", .root_module = cuda_audit }));
    if (b.option([]const u8, "audit-cubins", "cubin directory for CPU-only CUDA symbol audit")) |path| cuda_audit_run.addArg(path);
    if (projection_shapes) |path| cuda_audit_run.addArg(path);
    b.step("tp3-cuda-check", "CPU-only EXL3 loader and launch audit against actual cubins").dependOn(&cuda_audit_run.step);
    // SOURCE ONLY / UNRUN: exact affected cases, no inherited broad suites.
    const source_final = b.step("test-native-source-final", "Twenty-five final affected regressions; includes scratch-store fixture, no GPU or process fixtures");
    source_final.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.api, .filters = &.{
        "aggregate tier debt offers decode before restore admission poll and pump",
        "aggregate tier poll debits before another restore or pump quantum",
        "aggregate tier pending restores complete when all decoders are paused",
        "aggregate tier maintenance fatal fences the decode offer",
        "custody stage stepped warm foreground arrival cancels without generation",
        "cancellation drain retains keep and callback submissions without allocation",
        "cancellation drain cancels shared restore at last waiter without allocation",
        "fatal final restoreCancel fences cancellation and enclosing between decode",
        "final reported unpin retains poisoned job and healthy finish is once only",
        "ContextFull waits for a suspended fill without a live decoder",
        "FR01 terminal step failure drains release custody exactly once",
        "FR01 partial terminal open and released finished control drain once",
        "failAll clears suspended prefill gauges",
        "failAll drains admitted ownership with allocator failure after setup",
        "ContextFull retry resets destroyed attempt progress before readmission",
        "pinned and retired cache storage refuses an over-budget save without restores",
        "pinned invalid same-key cache state retires and permits a valid replacement",
        "cache save still evicts an unpinned victim within budget",
        "scoped resume pins survive real budget eviction and retired cancellation",
        "terminal native grid composes exact end with family and namespace eligibility",
        "cache keep callback retains decoded continuation tokens beyond request length",
    } })).step);
    source_final.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{
        "one rank: debt maintenance retires capture and restore events without launching copies",
        "outer follower decoder accepts fill_resume 13 and rejects reserved kinds",
        "idle relocation expired ready query is never admitted",
        "idle relocation expired entry never reaches side effects",
    } })).step);
    const final_safety = b.step("test-native-final-safety", "Final affected bind, device health and fatal policy tests");
    final_safety.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.engines, .filters = &.{"native bind"} })).step);
    final_safety.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.api, .filters = &.{ "device health", "peer-watch classifies" } })).step);
    const rank0_test_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/lanes_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    rank0_test_module.addImport("cuda", cuda);
    rank0_test_module.addImport("dsv41", eng.dsv41);
    rank0_test_module.addImport("lanes", mods.lanes);
    b.step("test-native-rank0-custody", "Actual rank zero teardown failure matrix").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = rank0_test_module, .filters = &.{"custody stage actual Rank0Custody"} })).step);
    b.step("test-native-family-custody", "Actual family state cleanup and spill admission host stages").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "custody stage family State", "custody stage openSpill" } })).step);
    const buffer_tests = b.step("test-native-buffer-custody", "Actual buffer convenience constructor host matrix");
    buffer_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cuda, .filters = &.{"custody stage convenience"} })).step);
    const nested_tests = b.step("test-native-nested-custody", "Weight and spill constructor custody host matrices");
    nested_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "custody stage weights", "custody stage spillEvents" } })).step);
    const owner_tests = b.step("test-native-owner-final", "Owner admission, staged custody and stepped warm regressions");
    for ([_]*std.Build.Module{ cuda, eng.api, eng.engines, eng.dsv41 }) |m| {
        owner_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{ "admission", "owner reference", "custody stage", "step-ahead: a reply", "Lt custody", "NCCL custody", "terminal native grid" } })).step);
    }
    const custody_tests = b.step("test-native-custody", "Scoped cache borrow and CUDA cleanup host regressions");
    for ([_]*std.Build.Module{ mods.core, cuda }) |m| {
        custody_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{ "scoped resume", "poison" } })).step);
    }
    const admission_tests = b.step("test-prefill-admission", "Bounded admission, restored images and whole-pass switch on the host");
    admission_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.api, .filters = &.{
        "late short admission", "partial admission", "a lane host resumes a conversation", "a turn's mark evicting",
        "image cache resumes when pixels match",
    } })).step);
    admission_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{
        "pictures in a resumed pass", "prefill admission switch", "outer follower decoder accepts fill_resume",
    } })).step);
    const stepped_tests = b.step("test-native-stepped", "Focused resumable prefill scheduler and exact-token tests");
    for ([_]*std.Build.Module{ eng.api, eng.engines, eng.dsv41 }) |m| {
        stepped_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{ "stepped prefill", "prefill interleave", "fatal restorePoll", "step-ahead", "native bind" } })).step);
    }
    b.step("test-native-relocation-whole", "Relocation spill staging races and terminal reserve integration").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "rows that change before they are staged", "guard without staging room", "guard stages the follower", "spill relocation" } })).step);
    const placement_tests = b.step("test-native-placement", "Affected replicated placement regressions");
    placement_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "placement keeps restore", "prefix growth relocates", "pool pressure", "soft reservations", "idle relocation receipt" } })).step);
    const ahead_tests = b.step("test-native-ahead", "Step-ahead lifecycle and exact-token regressions");
    ahead_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.api, .filters = &.{"step-ahead"} })).step);
    const integration_tests = b.step("test-native-integration", "Focused native safety and step-ahead host tests");
    for ([_]*std.Build.Module{ eng.api, eng.engines, eng.dsv41 }) |m| {
        integration_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{ "step-ahead", "fork", "poison", "soft reservations", "pool pressure", "native bind" } })).step);
    }
    const context_tests = b.step("test-context-capacity", "Focused spill, cache, KV ownership and rank-protocol host tests");
    for ([_]*std.Build.Module{ mods.core, eng.api, eng.dsv41 }) |m| {
        const tests = b.addTest(.{ .root_module = m, .filters = &.{ "spill", "cache", "shard", "parts", "rank_wire", "rank identity", "RowPairing", "ring", "continuation", "pictures", "TP3" } });
        context_tests.dependOn(&b.addRunArtifact(tests).step);
    }
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, eng.api, eng.engines, eng.dsv41, stagger(b, host, .debug) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    // Linux must exercise the HTTP/engine seam too; the original build only
    // attached these tests to Metal's nativeServer target.
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const server_tests = b.createModule(.{ .root_source_file = b.path("zig/src/server/root.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = eng.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template } } });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests })).step);
    const image_count_tests = b.step("test-image-count", "Image spans from headers and the count routes' exact image counts, CPU only");
    image_count_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "header size", "header sizes", "a header past" } })).step);
    image_count_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"a count route"} })).step);
    const room_tests = b.step("test-restore-room", "Bounded nonsharded restore-start wait, tier lifecycle, CLI and metrics");
    for ([_]*std.Build.Module{ eng.api, eng.dsv41, server_tests }) |m| {
        room_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{ "restore room", "parked", "restore", "aggregate tier", "spill flags", "spill CLI", "cache metrics" } })).step);
    }
    const structured_tests = b.step("test-structured-output", "Focused grammar, DSML and masked-row host tests");
    for ([_]*std.Build.Module{ mods.lanes, eng.api, eng.dsv41, server_tests }) |m| {
        structured_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{ "structured output", "grammar", "DSML" } })).step);
    }
    context_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{ "cache families", "kv shard families", "Responses token counting", "image span" } })).step);
    const graph_telemetry_tests = b.step("test-graph-telemetry", "Graph telemetry, TP3 cache/votes and capture-clean learning, CPU only");
    graph_telemetry_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "graph telemetry", "graph", "vote", "three ranks" } })).step);
    graph_telemetry_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mods.lanes, .filters = &.{ "graph telemetry" } })).step);
    b.step("test-keyed-draft", "Keyed draft sampler, TP3 frame, graph keys and launch regressions, CPU only").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "keyed", "Markov", "pre hook", "logits row", "prefix sampling", "candidate windows", "outer follower decoder", "rank zero drafter", "graph cache" } })).step);
    b.step("test-drafter-pairfault", "Rank-zero drafter pair-ending faults and follower-watch decisions, CPU only").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "rank zero drafter qualification faults", "follower watch" } })).step);
    const cache_batch_tests = b.step("test-cache-batch", "Bounded dead release, first continuation, TP3 receipts and key compatibility, CPU only");
    for ([_]*std.Build.Module{ mods.core, eng.api, eng.dsv41 }) |m| {
        cache_batch_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{ "401a", "cache batch" } })).step);
    }
    b.step("test-decode-greedy-head", "Greedy head policy, wire, readiness and custody CPU tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{"decode greedy head"} })).step);
    const slice_tests = b.step("test-prefill-slices", "Yielded layer slices, custody and TP3 frames, CPU only");
    for ([_]*std.Build.Module{ eng.api, eng.dsv41 }) |mod| slice_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod, .filters = &.{"prefill slice"} })).step);
    b.step("test-prefill128", "Laky 1.2.8 affected prefill checks, CPU only").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "prefill128" } })).step);
    b.step("test-prefill-owned", "TF_DS_PREFILL_OWNED's row thirds on the host").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "owned thirds", "the features" } })).step);
    b.step("test-engram-fault", "Engram read faults that end the pair, on the host").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "rankFault", "direct all-gather" } })).step);
    b.step("test-direct-gather", "TF_DS_DIRECT_GATHER's sends and receives on the host").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = eng.dsv41, .filters = &.{ "direct all-gather", "NCCL custody" } })).step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
    // port_kernels.zig's CPU references and launch geometry (the GLM port's kernels; their GPU check is tf-dsv41-port)
    const port = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/port_kernels.zig"), .target = host, .optimize = .debug, .link_libc = true });
    port.addImport("cuda", cuda);
    const port_run = b.addRunArtifact(b.addTest(.{ .root_module = port }));
    step.dependOn(&port_run.step);
    // tri_norm.zig alone (no dsv41 module): the kit-driven _rowmm_parts split for rounds past 16 rows, every recorded launch
    const norm = b.createModule(.{ .root_source_file = b.path("zig/src/families/dsv41/tri_norm.zig"), .target = host, .optimize = .debug, .link_libc = true });
    norm.addImport("cuda", cuda);
    const norm_run = b.addRunArtifact(b.addTest(.{ .root_module = norm }));
    const port_step = b.step("test-dsv41-port", "The GLM port kernels' CPU references and launch geometry, and tri_norm's gate split (no GPU)");
    port_step.dependOn(&port_run.step);
    port_step.dependOn(&norm_run.step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(k.flags);
    const a = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}

/// Cross-build a release server, embedding the same fatbins on either host CPU.
pub fn distServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        if (k.opt_in) continue; // no served path calls one
        image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
    };
    const cuda = runtime(b, target, .fast, if (prebuilt != null) &images else &.{});
    const mods = family(b, target, .fast, cuda, draft_ids);
    return nativeServer(b, target, .fast, cuda, mods.lanes, mods.nemotron, mods.core, mods.tokenizer, build_options, false);
}

/// Validate the complete named input set, including images unused by today's server.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const check = b.addSystemCommand(&.{ "sh", "-c", "for file do [ -f \"$file\" ] && [ -s \"$file\" ] || { echo \"missing or empty CUDA fatbin: $file\" >&2; exit 1; }; done", "check-fatbins" });
    for (kernels) |k| if (!k.opt_in) check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    return &check.step;
}
