//! Isolated communication diagnostic, no model loading or quality qualification.
//! Checked payloads before/after real Pillow startup and on an HTTP-like worker.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const pil = dsv41.pil;

const Work = struct {
    ctx: *const cuda.Context,
    ring: *dsv41.rdma.Ring,
    stream: cuda.Stream,
    send: cuda.DeviceBuffer,
    recv: cuda.DeviceBuffer,
    host: []f32,
    rank: u32,
    world: u32,
    ok: bool = false,
    failure: ?anyerror = null,

    fn run(w: *Work) void {
        w.check() catch |err| { w.failure = err; };
    }

    fn check(w: *Work) !void {
        try w.ctx.makeCurrent();
        w.ok = true;
        for ([_]usize{ 5120, 43136 }) |cols| for ([_]usize{ 1, 6, 13, 16 }) |rows| {
            const n = rows * cols;
            for (0..3) |pass| {
                for (w.host[0..n], 0..) |*x, j| x.* = value(w.rank, j, pass, rows);
                try w.send.upload(0, std.mem.sliceAsBytes(w.host[0..n]));
                try w.ring.gather(w.stream, w.send.ptr, @intCast(n / 4), w.recv.ptr);
                try w.stream.synchronize();
                try w.recv.download(0, std.mem.sliceAsBytes(w.host[0..w.world*n]));
                for (0..w.world) |r| for (0..n) |j| {
                    if (w.host[r*n+j] != value(@intCast(r), j, pass, rows)) w.ok = false;
                };
                var dsts: [dsv41.rdma.max_ranks]dsv41.rdma.Ring.Dst = undefined;
                for (0..w.world) |r| dsts[r] = .{ .ptr = w.recv.ptr + r*cols*4, .row4 = @intCast(cols/4), .stride4 = w.world*cols/4 };
                try w.ring.gatherInto(w.stream,w.send.ptr,@intCast(rows),@intCast(cols/4),dsts[0..w.world]);
                try w.stream.synchronize();
                try w.recv.download(0,std.mem.sliceAsBytes(w.host[0..w.world*n]));
                for (0..rows) |row| for (0..w.world) |r| for (0..cols) |col| {
                    if (w.host[row*w.world*cols+r*cols+col] != value(@intCast(r),row*cols+col,pass,rows)) w.ok = false;
                };
            }
        };
        if (w.ring.failure()!=null) return error.RdmaFailed;
    }
};

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len!=6) {
        std.debug.print("usage: tf-dsv41-rdma-lifecycle RANK WORLD MASTER PORT DEVICES\n",.{});
        return 2;
    }
    const rank=try std.fmt.parseInt(u32,args[1],10);
    const world=try std.fmt.parseInt(u32,args[2],10);
    if (world!=3 or rank>=world) return error.BadWorld;
    const ip=try dsv41.link.parseIp(args[3]);
    const port=try std.fmt.parseInt(u16,args[4],10);
    const a=init.arena.allocator();
    var devices:std.ArrayList([]const u8)=.empty;
    var it=std.mem.splitScalar(u8,args[5],',');
    while (it.next()) |device| {if (device.len==0) return error.BadDevice;try devices.append(a,device);}
    var driver=try cuda.Driver.open();defer driver.close();
    var ctx=try cuda.Context.init(&driver,0);defer ctx.deinit();
    var fan=try dsv41.prompt2d.Fan.openTimed(init.io,ip,port,rank,world,120);defer fan.close();
    var comm=try fan.comm(rank,world);defer comm.deinit();
    var stream=try cuda.Stream.init(&driver,true);defer stream.deinit();
    var module=try cuda.Module.load(&driver,cuda.kernels.dsv41_rdma);defer module.unload();
    const most=16*43136;
    var send=try cuda.DeviceBuffer.alloc(&driver,most*4);defer send.free();
    var recv=try cuda.DeviceBuffer.alloc(&driver,world*most*4);defer recv.free();
    const host=try a.alloc(f32,world*most);
    const ring=try dsv41.model.Model.openRingSized(init.gpa,&driver,try dsv41.rdma.Kernels.load(module),devices.items,rank,&comm,stream,dsv41.model.ring_bytes);
    defer ring.destroy(init.gpa);
    std.debug.print("{{\"rank\":{d},\"send_addr\":{d},\"send_bytes\":{d},\"recv_addr\":{d},\"recv_bytes\":{d},\"meta_addr\":{d},\"meta_bytes\":{d}}}\n",.{rank,@intFromPtr(ring.send.mem.ptr),ring.send.mem.len,@intFromPtr(ring.recv.mem.ptr),ring.recv.mem.len,@intFromPtr(ring.meta.mem.ptr),ring.meta.mem.len});
    var pillow:?*pil.Pil=null;defer if (pillow) |p| p.stop(init.gpa);
    for ([_][]const u8{"baseline","thread","pillow","pillow-thread"},0..) |stage,index| {
        if (index==2 and rank==0) {
            pillow=try pil.Pil.start(init.gpa,init.io);
            std.debug.print("{{\"rank\":0,\"pillow\":\"{s}\"}}\n",.{pillow.?.versionText()});
        }
        try ctx.makeCurrent();
        try comm.allGather(send.ptr,recv.ptr,1,.u8,stream);try stream.synchronize();
        var work:Work=.{.ctx=&ctx,.ring=ring,.stream=stream,.send=send,.recv=recv,.host=host,.rank=rank,.world=world};
        if (rank==0 and (index==1 or index==3)) {
            const thread=try std.Thread.spawn(.{.stack_size=16<<20},Work.run,.{&work});thread.join();
        } else work.run();
        std.debug.print("{{\"rank\":{d},\"stage\":\"{s}\",\"ok\":{},\"error\":\"{s}\",\"model_quality_qualified\":false}}\n",.{rank,stage,work.ok and work.failure==null,if (work.failure) |e| @errorName(e) else ""});
        if (work.failure!=null or !work.ok) return 1;
    }
    return 0;
}
fn value(rank:u32,index:usize,pass:usize,rows:usize) f32 {
    return @floatFromInt((rank+1)*1000003+@as(u32,@intCast(index%65521+pass*70001+rows*997)));
}
