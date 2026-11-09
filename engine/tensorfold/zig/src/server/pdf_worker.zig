//! The PDF render worker: the server's own executable, started fresh per document, renders under limits over pipes.
//! Ported from TensorFold upstream (joernchen, PR #1820; ashhart, PR #2393); the reply carries PNG pages.
const std = @import("std");
const builtin = @import("builtin");
const errors = @import("errors.zig");
const log = @import("log.zig");
const pdf = @import("pdf.zig");
const png = @import("png.zig");
const Cx = errors.Cx;
const Budget = pdf.Budget;
const Allocator = std.mem.Allocator;

/// The hidden argument that makes tensorfold-native run only the PDF render worker (main.zig).
pub const arg = "--pdf-render-worker";
/// The binary started as the worker when set: tests name the server binary, since a test runner cannot be the worker.
pub var server_exe: ?[]const u8 = null;
const failed = "PDF rendering exceeded resource limits or failed";

/// The parent's request on the worker's stdin, ahead of the PDF bytes; both ends are the same build.
const Request = extern struct { remaining_ns: u64, pixels_used: u64, pdf_len: u64, max_pages: u32, pdf_pages_used: u32, mode: u32, _pad: u32 = 0 };

/// The most a reply can hold: every page within the request's pixel limit as a PNG stored uncompressed, and its framing.
const max_reply_bytes: usize = 4 * pdf.max_request_pixels + (1 << 20);

/// This executable and the hidden argument; Linux's /proc/self/exe still names the running file once it is replaced.
pub fn argv(a: Allocator) ![]const []const u8 {
    if (builtin.is_test and server_exe == null) return error.NoWorker; // a test runner started again is no worker
    const exe = server_exe orelse if (comptime builtin.os.tag == .linux) "/proc/self/exe" else try std.process.executablePathAlloc(log.io(), a);
    return a.dupe([]const u8, &.{ exe, arg });
}

/// Renders (or measures) `pdf_bytes` in a fresh worker started from `cmd`, killed at `deadline_ns`, and checks its reply.
pub fn render(cx: *Cx, cmd: []const []const u8, pdf_bytes: []const u8, mode: pdf.Mode, max_pages: u32, deadline_ns: i96, budget: *Budget, out: *std.ArrayList([]const u8)) errors.Refused!void {
    const io = log.io();
    var child = std.process.spawn(io, .{ .argv = cmd, .stdin = .pipe, .stdout = .pipe }) catch return cx.refuse("PDF rendering worker process could not be started");
    defer if (child.id) |pid| {
        std.posix.kill(pid, .KILL) catch {};
        _ = child.wait(io) catch {};
    };
    const left = deadline_ns - std.Io.Clock.awake.now(io).toNanoseconds();
    const req: Request = .{ .remaining_ns = @intCast(@max(0, left)), .pixels_used = budget.pixels_used, .pdf_len = pdf_bytes.len, .max_pages = max_pages, .pdf_pages_used = budget.pdf_pages_used, .mode = @backingInt(mode) };
    const in = child.stdin.?.handle;
    const nonblock: u32 = @bitCast(std.c.O{ .NONBLOCK = true });
    _ = std.c.fcntl(in, std.c.F.SETFL, std.c.fcntl(in, std.c.F.GETFL) | @as(c_int, @bitCast(nonblock)));
    _ = sendAll(in, std.mem.asBytes(&req), deadline_ns) and sendAll(in, pdf_bytes, deadline_ns);
    child.stdin.?.close(io);
    child.stdin = null;
    const frame = readChildPipe(cx.a, child.stdout.?.handle, deadline_ns, max_reply_bytes) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Timeout => return cx.refuse("PDF rendering timed out"),
        error.Overflow => return cx.refuse(failed),
    };
    const term = child.wait(io) catch return cx.refuse(failed);
    return reply(cx, term, frame, mode, max_pages, budget, out);
}

/// A worker the kernel stopped at a limit keeps the limit message; any other signal is named as a crash.
fn reply(cx: *Cx, term: std.process.Child.Term, frame: []const u8, mode: pdf.Mode, max_pages: u32, budget: *Budget, out: *std.ArrayList([]const u8)) errors.Refused!void {
    if (term == .signal) switch (term.signal) {
        .KILL, .XCPU, .XFSZ => return cx.refuse(failed),
        else => |sig| if (std.enums.tagName(std.posix.SIG, sig)) |name|
            return cx.fail(.request, "the PDF render worker crashed (SIG{s})", .{name})
        else
            return cx.fail(.request, "the PDF render worker crashed (signal {d})", .{@backingInt(sig)}),
    };
    return parseChildReply(cx, frame, mode, max_pages, budget, out);
}

fn ready(fd: c_int, events: i16, deadline_ns: i96) bool {
    const now = std.Io.Clock.awake.now(log.io()).toNanoseconds();
    if (now >= deadline_ns) return false;
    const rem_ms: i32 = @intCast(@max(@as(i96, 1), @min(@as(i96, 10_000), @divTrunc(deadline_ns - now + 999_999, 1_000_000))));
    var pfd = [1]std.posix.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
    return (std.posix.poll(&pfd, rem_ms) catch return false) > 0;
}

fn sendAll(fd: c_int, bytes: []const u8, deadline_ns: i96) bool {
    var off: usize = 0;
    while (off < bytes.len) {
        if (!ready(fd, std.posix.POLL.OUT, deadline_ns)) return false;
        const n = std.c.write(fd, bytes[off..].ptr, bytes.len - off);
        if (n < 0 and std.posix.errno(n) == .AGAIN) continue;
        if (n <= 0) return false;
        off += @intCast(n);
    }
    return true;
}

fn readChildPipe(a: Allocator, fd: c_int, deadline_ns: i96, max_bytes: usize) error{ OutOfMemory, Timeout, Overflow }![]u8 {
    var got: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        if (!ready(fd, std.posix.POLL.IN, deadline_ns)) return error.Timeout;
        const n = std.c.read(fd, &buf, buf.len);
        if (n == 0) break;
        if (n < 0) return error.Timeout;
        if (got.items.len + @as(usize, @intCast(n)) > max_bytes) return error.Overflow;
        try got.appendSlice(a, buf[0..@intCast(n)]);
    }
    return got.items;
}

/// The worker's reply, checked as untrusted: each page a PNG of the size its header declares (only that header when
/// measuring), no side past pdf.max_page_side, the page and pixel budgets recomputed here from those headers.
pub fn parseChildReply(cx: *Cx, frame: []const u8, mode: pdf.Mode, max_pages: u32, budget: *Budget, out: *std.ArrayList([]const u8)) errors.Refused!void {
    if (frame.len == 0) return cx.refuse(failed);
    switch (frame[0]) {
        0 => {
            if (frame.len < 5) return cx.refuse(failed);
            const count = std.mem.readInt(u32, frame[1..5], .little);
            if (count == 0) return cx.refuse("PDF document contains no pages");
            const limit: u32 = @max(@as(u32, 1), max_pages);
            if (budget.pdf_pages_used +| count > limit) return pdf.pageLimitRefusal(cx, limit, budget.pdf_pages_used +| count);
            var next = budget.*;
            const first = out.items.len;
            errdefer out.shrinkRetainingCapacity(first);
            var pos: usize = 5;
            for (0..count) |_| {
                if (pos + 4 > frame.len) return cx.refuse(failed);
                const len: usize = std.mem.readInt(u32, frame[pos..][0..4], .little);
                pos += 4;
                if (len == 0 or pos + len > frame.len) return cx.refuse(failed);
                const page = frame[pos .. pos + len];
                const s = png.size(page) orelse return cx.refuse(failed);
                // a side past the worker's clamp is refused before any size math (png.bound, the pixel count)
                if (s.w > pdf.max_page_side or s.h > pdf.max_page_side) return cx.refuse(failed);
                if (if (mode == .measure) len != png.header_len else len <= png.header_len or len > png.bound(s.w, s.h)) return cx.refuse(failed);
                const px = @as(u64, s.w) * s.h;
                if (next.pixels_used + px > pdf.max_request_pixels) return cx.refuse(pdf.pixel_refusal);
                next.pdf_pages_used += 1;
                next.pixels_used += px;
                try out.append(cx.a, page);
                pos += len;
            }
            if (pos != frame.len) return cx.refuse(failed);
            budget.* = next;
        },
        1 => {
            if (frame.len < 6) return cx.refuse("PDF rendering failed");
            cx.kind = std.enums.fromInt(errors.Kind, frame[1]) orelse .request;
            const len: usize = std.mem.readInt(u32, frame[2..6], .little);
            if (6 + len != frame.len) return cx.refuse("PDF rendering failed");
            cx.message = frame[6 .. 6 + len];
            return error.Refused;
        },
        2 => return error.OutOfMemory,
        else => return cx.refuse(failed),
    }
}

/// The worker process: reads its request, applies the limits, opens PDFium, writes one reply frame and exits.
pub fn serve() noreturn {
    closeRange(3, ~@as(c_uint, 0));
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    var cx: Cx = .{ .a = arena.allocator() };
    var pages: std.ArrayList([]const u8) = .empty;
    if (work(&cx, &pages)) |_| {
        _ = writeAll(1, &.{0});
        _ = writeInt(u32, 1, @intCast(pages.items.len));
        for (pages.items) |page| {
            _ = writeInt(u32, 1, @intCast(page.len));
            _ = writeAll(1, page);
        }
    } else |err| switch (err) {
        error.OutOfMemory => _ = writeAll(1, &.{2}),
        error.Refused => {
            _ = writeAll(1, &.{ 1, @backingInt(cx.kind) });
            _ = writeInt(u32, 1, @intCast(cx.message.len));
            _ = writeAll(1, cx.message);
        },
    }
    std.c._exit(0);
}

fn work(cx: *Cx, pages: *std.ArrayList([]const u8)) errors.Refused!void {
    var req: Request = undefined;
    if (!readAll(0, std.mem.asBytes(&req)) or req.pdf_len > pdf.max_pdf_bytes) return cx.refuse(failed);
    const mode = std.enums.fromInt(pdf.Mode, req.mode) orelse return cx.refuse(failed);
    const deadline_ns = std.Io.Clock.awake.now(log.io()).toNanoseconds() + req.remaining_ns;
    const pdf_bytes = try cx.a.alloc(u8, @intCast(req.pdf_len));
    if (!readAll(0, pdf_bytes)) return cx.refuse(failed);
    applyChildLimits();
    const f = pdf.loadPdfium(cx.a, log.io()) orelse return cx.fail(.unimplemented, "{s}", .{pdf.missing_message});
    f.FPDF_InitLibrary();
    var budget: Budget = .{ .pdf_pages_used = req.pdf_pages_used, .pixels_used = req.pixels_used };
    return pdf.renderDocument(cx, f, pdf_bytes, mode, req.max_pages, deadline_ns, &budget, pages);
}

fn applyChildLimits() void {
    std.posix.setrlimit(.CPU, .{ .cur = 10, .max = 10 }) catch {};
    std.posix.setrlimit(.CORE, .{ .cur = 0, .max = 0 }) catch {};
    std.posix.setrlimit(.FSIZE, .{ .cur = 0, .max = 0 }) catch {};
    if (comptime builtin.os.tag == .linux) {
        const fd = std.c.open("/proc/self/statm", .{}, @as(c_uint, 0));
        if (fd >= 0) {
            defer _ = std.c.close(fd);
            var buf: [64]u8 = undefined;
            const n = std.c.read(fd, &buf, buf.len);
            if (n > 0) {
                const text = buf[0..@intCast(n)];
                const end = std.mem.indexOfAny(u8, text, " \t\n") orelse text.len;
                if (std.fmt.parseInt(u64, text[0..end], 10) catch null) |vm_pages| {
                    // statm counts the kernel's pages: 64 KiB on some aarch64 kernels, so the size is queried
                    const cap = vm_pages * std.heap.pageSize() + pdf.max_child_heap_bytes;
                    std.posix.setrlimit(.AS, .{ .cur = cap, .max = cap }) catch {};
                }
            }
        }
    }
}

fn closeRange(first: c_uint, last: c_uint) void {
    if (first > last) return;
    if (comptime builtin.os.tag == .linux) if (std.os.linux.syscall3(.close_range, first, last, 0) == 0) return;
    const limit = std.posix.getrlimit(.NOFILE) catch std.posix.rlimit{ .cur = 1024, .max = 1024 };
    const top: c_uint = @intCast(@min(@as(u64, last), @min(@as(u64, limit.cur), 65536)));
    var d = first;
    while (d <= top) : (d += 1) _ = std.c.close(@intCast(d));
}

fn readAll(fd: c_int, buf: []u8) bool {
    var off: usize = 0;
    while (off < buf.len) {
        const n = std.c.read(fd, buf[off..].ptr, buf.len - off);
        if (n <= 0) return false;
        off += @intCast(n);
    }
    return true;
}

fn writeAll(fd: c_int, bytes: []const u8) bool {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(fd, bytes[off..].ptr, bytes.len - off);
        if (n <= 0) return false;
        off += @intCast(n);
    }
    return true;
}

fn writeInt(comptime T: type, fd: c_int, val: T) bool {
    var b: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &b, val, .little);
    return writeAll(fd, &b);
}

fn fakeWorker(a: Allocator, script: []const u8, ms: i96) ![]const u8 {
    log.init(std.testing.io, false);
    var cx: Cx = .{ .a = a };
    var budget: Budget = .{};
    var pages: std.ArrayList([]const u8) = .empty;
    const deadline_ns = std.Io.Clock.awake.now(log.io()).toNanoseconds() + ms * std.time.ns_per_ms;
    try std.testing.expectError(error.Refused, render(&cx, &.{ "/bin/sh", "-c", script }, pdf.minimal_two_page_pdf, .render, 4, deadline_ns, &budget, &pages));
    try std.testing.expectEqual(Budget{}, budget);
    return cx.message;
}

test "PDF render worker: a worker killed by a signal is named as a crash; a limit or a missing reply keeps today's message" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest; // the worker is POSIX only
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("the PDF render worker crashed (SIGSEGV)", try fakeWorker(a, "cat >/dev/null; kill -SEGV $$", 10_000));
    try std.testing.expectEqualStrings(failed, try fakeWorker(a, "cat >/dev/null; ulimit -t 1; exec /bin/sh -c 'while :; do :; done'", 10_000));
    try std.testing.expectEqualStrings(failed, try fakeWorker(a, "cat >/dev/null; exit 0", 10_000));
    try std.testing.expectEqualStrings(failed, try fakeWorker(a, "cat >/dev/null; exit 3", 10_000));
    for ([_]std.process.Child.Term{ .{ .signal = .KILL }, .{ .signal = .XCPU }, .{ .signal = .XFSZ }, .{ .signal = .ABRT } }) |term| {
        var cx: Cx = .{ .a = a };
        var budget: Budget = .{};
        var pages: std.ArrayList([]const u8) = .empty;
        try std.testing.expectError(error.Refused, reply(&cx, term, &.{ 0, 0, 0, 0, 0 }, .render, 4, &budget, &pages));
        try std.testing.expectEqualStrings(if (term.signal == .ABRT) "the PDF render worker crashed (SIGABRT)" else failed, cx.message);
    }
}

test "PDF render worker: a worker past its deadline is killed and reported as timed out" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest; // the worker is POSIX only
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const started = std.Io.Clock.awake.now(log.io()).toNanoseconds();
    try std.testing.expectEqualStrings("PDF rendering timed out", try fakeWorker(arena.allocator(), "exec sleep 30", 300));
    try std.testing.expect(std.Io.Clock.awake.now(log.io()).toNanoseconds() - started < 10 * std.time.ns_per_s);
}

/// A reply frame of `pages` as the worker writes it.
fn frameOf(a: Allocator, pages: []const []const u8) ![]u8 {
    var f: std.ArrayList(u8) = .empty;
    try f.append(a, 0);
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, @intCast(pages.len), .little);
    try f.appendSlice(a, &b);
    for (pages) |p| {
        std.mem.writeInt(u32, &b, @intCast(p.len), .little);
        try f.appendSlice(a, &b);
        try f.appendSlice(a, p);
    }
    return f.items;
}

test "PDF render worker: a reply's pages are checked and the budget recomputed from their PNG headers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const head = png.header(400, 300);
    // measure: headers only, the pixels counted from them
    var budget: Budget = .{};
    var pages: std.ArrayList([]const u8) = .empty;
    var cx: Cx = .{ .a = a };
    try parseChildReply(&cx, try frameOf(a, &.{ &head, &head }), .measure, 4, &budget, &pages);
    try std.testing.expectEqual(Budget{ .pdf_pages_used = 2, .pixels_used = 2 * 400 * 300 }, budget);
    try std.testing.expectEqual(@as(usize, 2), pages.items.len);
    // a third and fourth fit four pages; a fifth does not, and nothing of that reply is kept
    try std.testing.expectError(error.Refused, parseChildReply(&cx, try frameOf(a, &.{ &head, &head, &head }), .measure, 4, &budget, &pages));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "at most 4 pages") != null);
    try std.testing.expectEqual(@as(usize, 2), pages.items.len);
    // a measured page longer than its header, a rendered page that is only one, or not a PNG at all: refused
    const tail = try std.mem.concat(a, u8, &.{ &head, "IDATxxxx" });
    for ([_]struct { []const u8, pdf.Mode }{ .{ tail, .measure }, .{ &head, .render }, .{ "not a png at all, not at all, no", .render } }) |case| {
        try std.testing.expectError(error.Refused, parseChildReply(&cx, try frameOf(a, &.{case[0]}), case[1], 16, &budget, &pages));
        try std.testing.expectEqualStrings(failed, cx.message);
    }
    // a header naming a side past the worker's clamp (0xFFFFFFFF square, or one side over): refused before its size
    // is used, rendered or measured, and nothing kept
    for ([_][2]u32{ .{ 0xFFFF_FFFF, 0xFFFF_FFFF }, .{ pdf.max_page_side + 1, 28 }, .{ 28, pdf.max_page_side + 1 } }) |side| {
        const huge = png.header(side[0], side[1]);
        const huge_tail = try std.mem.concat(a, u8, &.{ &huge, "IDATxxxx" });
        for ([_]struct { []const u8, pdf.Mode }{ .{ huge_tail, .render }, .{ &huge, .measure } }) |case| {
            var clean: Budget = .{};
            try std.testing.expectError(error.Refused, parseChildReply(&cx, try frameOf(a, &.{case[0]}), case[1], 16, &clean, &pages));
            try std.testing.expectEqualStrings(failed, cx.message);
            try std.testing.expectEqual(Budget{}, clean);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), pages.items.len);
    // pixels past the request's limit: refused before the page is kept
    var full: Budget = .{ .pixels_used = pdf.max_request_pixels - 1000 };
    try std.testing.expectError(error.Refused, parseChildReply(&cx, try frameOf(a, &.{&head}), .measure, 16, &full, &pages));
    try std.testing.expectEqualStrings(pdf.pixel_refusal, cx.message);
    try std.testing.expectEqual(@as(usize, 2), pages.items.len);
}
