//! CPU-only proof of the real process exit code; never opens a model or GPU.
const std = @import("std");
const watch = @import("peer_watch");
pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return 2;
    const Context = struct {
        dead: bool,
        fn failed(ptr: *anyopaque) bool {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            return self.dead;
        }
    };
    var context: Context = .{ .dead = std.mem.eql(u8, args[1], "peer") };
    var m: watch.Monitor = .{ .io = init.io, .probe = .{ .ctx = &context, .failed = Context.failed }, .policy = .{ .stall_ms = 10, .grace_ms = 1, .poll_ms = 1 } };
    if (std.mem.eql(u8, args[1], "stall")) m.submitted();
    if (std.mem.eql(u8, args[1], "fatal")) m.signal(error.ShardDiverged);
    try m.start();
    defer m.stop();
    try std.Io.sleep(init.io, .fromMilliseconds(50), .awake);
    if (std.mem.eql(u8, args[1], "idle")) return if (m.tripped.load(.acquire)) 3 else 0;
    return 99; // every non-idle mode must have exited 75 before reaching this line
}
