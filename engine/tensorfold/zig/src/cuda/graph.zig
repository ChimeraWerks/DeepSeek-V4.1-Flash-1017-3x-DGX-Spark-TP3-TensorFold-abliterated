//! CUDA graphs: captured from a stream or built node by node, instantiated once, replayed and updated in place.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const Stream = @import("stream.zig").Stream;
const Function = @import("module.zig").Function;
const launch = @import("launch.zig");

pub const Node = abi.GraphNode;

/// Starts capturing `stream`; every launch on it is recorded until `endCapture`, nothing runs.
pub fn beginCapture(stream: Stream, mode: abi.CaptureMode) Error!void {
    try stream.owner.ensure();
    try stream.d.ensureHealthy();
    try stream.d.check(stream.d.invoke(.cuStreamBeginCapture_v2, .{stream.handle, mode}), "cuStreamBeginCapture");
}

/// Ends the capture; an invalidated capture returns an error and no graph.
pub fn endCapture(stream: Stream) Error!Graph {
    try stream.owner.ensure();
    try stream.d.ensureHealthy();
    var g: abi.Graph = null;
    try stream.d.check(stream.d.invoke(.cuStreamEndCapture, .{stream.handle, &g}), "cuStreamEndCapture");
    return .{ .d = stream.d, .owner = try stream.owner.retain(), .handle = g };
}

pub fn captureStatus(stream: Stream) Error!abi.CaptureStatus {
    try stream.owner.ensure();
    try stream.d.ensureHealthy();
    var s: abi.CaptureStatus = .none;
    try stream.d.check(stream.d.invoke(.cuStreamIsCapturing, .{stream.handle, &s}), "cuStreamIsCapturing");
    return s;
}

fn nodeParams(f: Function, cfg: launch.Config, args: *launch.Args) Error!abi.KernelNodeParams {
    try cfg.validate();
    if (cfg.cluster != null or cfg.pdl or cfg.cooperative) return error.Invalid;
    return .{
        .func = f.handle,
        .grid_x = cfg.grid.x,
        .grid_y = cfg.grid.y,
        .grid_z = cfg.grid.z,
        .block_x = cfg.block.x,
        .block_y = cfg.block.y,
        .block_z = cfg.block.z,
        .shared_bytes = cfg.shared,
        .params = args.pointers(),
        .extra = null,
    };
}

pub const Graph = struct {
    d: *const Driver,
    owner: @import("context_health.zig").Owner = .{},
    handle: abi.Graph,

    pub fn init(d: *const Driver) Error!Graph {
        try d.ensureHealthy();
        var g: abi.Graph = null;
        try d.check(d.invoke(.cuGraphCreate, .{&g, 0}), "cuGraphCreate");
        return .{ .d = d, .owner = @import("context_health.zig").Owner.capture(), .handle = g };
    }

    pub fn deinit(self: *Graph) void {
        self.owner.ensure() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.ensureHealthy() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.result(self.d.invoke(.cuGraphDestroy, .{self.handle})) catch { @import("context_health.zig").cleanupRetained(); return; };
        self.owner.release();
        self.* = undefined;
    }

    /// A kernel node after `deps`; the driver copies the argument values, so `args` may change afterwards.
    pub fn addKernel(self: Graph, deps: []const Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!Node {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        const p = try nodeParams(f, cfg, args);
        var n: Node = null;
        try self.d.check(self.d.invoke(.cuGraphAddKernelNode_v2, .{&n, self.handle, if (deps.len > 0) deps.ptr else null, deps.len, &p}), "cuGraphAddKernelNode");
        return n;
    }

    /// New arguments or geometry for a node of this (template) graph, before it is instantiated again or updated from.
    pub fn setKernel(self: Graph, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        const p = try nodeParams(f, cfg, args);
        try self.d.check(self.d.invoke(.cuGraphKernelNodeSetParams_v2, .{node, &p}), "cuGraphKernelNodeSetParams");
    }

    pub fn depend(self: Graph, from: Node, to: Node) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        const a = [1]Node{from};
        const b = [1]Node{to};
        try self.d.check(self.d.invoke(.cuGraphAddDependencies, .{self.handle, &a, &b, 1}), "cuGraphAddDependencies");
    }

    pub fn nodeCount(self: Graph) Error!usize {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        var n: usize = 0;
        try self.d.check(self.d.invoke(.cuGraphGetNodes, .{self.handle, null, &n}), "cuGraphGetNodes");
        return n;
    }

    /// Fills `out` with the graph's nodes (capture order for captured graphs) and returns them.
    pub fn nodes(self: Graph, out: []Node) Error![]Node {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        var n: usize = out.len;
        try self.d.check(self.d.invoke(.cuGraphGetNodes, .{self.handle, out.ptr, &n}), "cuGraphGetNodes");
        return out[0..@min(n, out.len)];
    }

    pub fn instantiate(self: Graph) Error!Exec {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        var e: abi.GraphExec = null;
        try self.d.check(self.d.invoke(.cuGraphInstantiateWithFlags, .{&e, self.handle, 0}), "cuGraphInstantiateWithFlags");
        return .{ .d = self.d, .owner = try self.owner.retain(), .handle = e };
    }
};

pub const Exec = struct {
    d: *const Driver,
    owner: @import("context_health.zig").Owner = .{},
    handle: abi.GraphExec,

    pub fn deinit(self: *Exec) void {
        self.owner.ensure() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.ensureHealthy() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.result(self.d.invoke(.cuGraphExecDestroy, .{self.handle})) catch { @import("context_health.zig").cleanupRetained(); return; };
        self.owner.release();
        self.* = undefined;
    }

    /// Moves the graph's work to the device ahead of the first launch, so that launch pays no setup.
    pub fn upload(self: Exec, stream: Stream) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        try self.d.check(self.d.invoke(.cuGraphUpload, .{self.handle, stream.handle}), "cuGraphUpload");
    }

    pub fn launchOn(self: Exec, stream: Stream) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        try self.d.check(self.d.invoke(.cuGraphLaunch, .{self.handle, stream.handle}), "cuGraphLaunch");
    }

    /// Changes one kernel node's arguments or geometry in the executable graph without rebuilding it.
    pub fn setKernel(self: Exec, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        const p = try nodeParams(f, cfg, args);
        try self.d.check(self.d.invoke(.cuGraphExecKernelNodeSetParams_v2, .{self.handle, node, &p}), "cuGraphExecKernelNodeSetParams");
    }

    /// Takes every node's parameters from `g`, which must have the same topology; returns the driver's verdict.
    pub fn update(self: Exec, g: Graph) Error!abi.ExecUpdateResult {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        var info: abi.ExecUpdateResultInfo = .{ .result = .success, .error_node = null, .error_from_node = null };
        const res = self.d.invoke(.cuGraphExecUpdate_v2, .{self.handle, g.handle, &info});
        if (res == abi.success) return info.result;
        if (@import("context_health.zig").sticky(res)) try self.d.result(res);
        if (info.result != .success) return info.result;
        try self.d.check(res, "cuGraphExecUpdate");
        unreachable;
    }
};


test "poison graph sticky result precedes topology verdict" {
    const health = @import("context_health.zig");
    const Mock = struct {
        fn update(_: abi.GraphExec, _: abi.Graph, info: *abi.ExecUpdateResultInfo) callconv(.c) abi.Result {
            info.result = @enumFromInt(1);
            return 700;
        }
    };
    var d: Driver = undefined;
    d.api.cuGraphExecUpdate_v2 = Mock.update;
    const h: abi.Context = @ptrFromInt(0xa991);
    try health.retain(@intFromPtr(&d), h);
    health.setCurrent(@intFromPtr(&d), h);
    defer health.clearCurrent(@intFromPtr(&d), h);
    const exec: Exec = .{ .d = &d, .owner = health.Owner.capture(), .handle = null };
    const graph: Graph = .{ .d = &d, .owner = health.Owner.capture(), .handle = null };
    try std.testing.expectError(error.CudaContextLost, exec.update(graph));
    try std.testing.expect(!health.boundHealthy());
}
