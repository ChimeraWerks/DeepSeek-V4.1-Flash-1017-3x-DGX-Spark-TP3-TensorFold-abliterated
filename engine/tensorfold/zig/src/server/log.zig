//! Server output: ``[tensorfold]`` lines on stdout, and the live status line they clear first.
const std = @import("std");
const Conn = @import("http_conn.zig").Conn;
const afterChars = @import("reply_text.zig").afterChars;

var global_io: ?std.Io = null;
var mutex: std.Io.Mutex = .init;
var shown = false; // the live line is on screen
var line_start = true; // the newest write ended its line, so a redraw cannot split it
var quiet = false;

pub fn init(io_: std.Io, silent: bool) void {
    global_io = io_;
    quiet = silent;
}

pub fn io() std.Io {
    return global_io orelse std.Io.Threaded.global_single_threaded.io();
}

const clear = "\r\x1b[2K";

fn writeOut(bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const rc = std.posix.system.write(1, bytes[off..].ptr, bytes.len - off);
        if (std.posix.errno(rc) != .SUCCESS or rc <= 0) return;
        off += @intCast(rc);
    }
}

/// One ``[tensorfold] ...`` line; its control bytes are written as ``\xNN``, as Python's ``log_message`` writes them.
pub fn line(comptime fmt: []const u8, args: anytype) void {
    if (quiet or @import("builtin").is_test) return; // a test's stdout is the build runner's channel
    var escaped: Escaped = undefined;
    raw(render(&escaped, fmt, args));
}

const line_max = 8192;
/// The most a line can grow to: every byte escaped, and its newline.
const Escaped = [4 * line_max + 1]u8;

/// The bytes ``line`` writes.
fn render(out: *Escaped, comptime fmt: []const u8, args: anytype) []const u8 {
    var buf: [line_max]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.print("[tensorfold] " ++ fmt, args) catch {}; // a longer line ends where the buffer does
    return controlSafe(out, w.buffered());
}

/// ``text`` and its newline with every C0, DEL and C1 control as ``\xNN``: a request line cannot drive the terminal.
fn controlSafe(out: []u8, text: []const u8) []const u8 {
    var w: std.Io.Writer = .fixed(out);
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const b = text[i];
        // a request line is latin-1 re-encoded as UTF-8 (http_conn.latin1), so its C1 control is 0xc2 then the byte
        const c1 = b == 0xc2 and i + 1 < text.len and text[i + 1] >= 0x80 and text[i + 1] <= 0x9f;
        if (c1) i += 1;
        if (c1 or b < 0x20 or b == 0x7f) {
            w.print("\\x{x:0>2}", .{text[i]}) catch break;
        } else w.writeByte(b) catch break;
    }
    w.writeByte('\n') catch unreachable; // out holds four bytes a character of text and the newline
    return w.buffered();
}

/// Text as given, the live line cleared first.
pub fn raw(text: []const u8) void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (shown) {
        writeOut(clear);
        shown = false;
    }
    writeOut(text);
    if (text.len > 0) line_start = text[text.len - 1] == '\n';
}

/// Redraws the status line unless a log line is half written.
pub fn status(text: []const u8, width: usize) void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (!line_start) return;
    writeOut(clear);
    writeOut(text[0..@min(text.len, @max(20, width) - 1)]);
    shown = true;
}

pub fn clearStatus() void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (shown) writeOut(clear);
    shown = false;
}

pub fn keySuffix(c: *const Conn) []const u8 {
    if (!c.auth_enabled) return "";
    return if (c.key_label) |label| label_suffix(label) else " key=unauthenticated";
}

threadlocal var suffix_buf: [80]u8 = undefined;

fn label_suffix(label: []const u8) []const u8 {
    return std.fmt.bufPrint(&suffix_buf, " key={s}", .{label}) catch " key=?";
}

/// The access line ``send_response`` prints: ``ip "GET / HTTP/1.1" 200 -``.
pub fn request(c: *const Conn, code: u16) void {
    line(request_fmt, .{ c.peer, c.requestline, code, keySuffix(c) });
}

const request_fmt = "{s} \"{s}\" {d} -{s}";

/// ``refused <id>: <message>``, the message cut to 300 characters: why a request got an error reply.
pub fn refused(buf: []u8, id: []const u8, message: []const u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("refused {s}: {s}", .{ id, message[0 .. message.len - afterChars(message, 300).len] }) catch {};
    return w.buffered();
}

/// ``ended <id> reason=...``: a request that ends without its reply, because its client left (``why`` null) or it failed.
pub fn ended(buf: []u8, id: []const u8, why: ?[]const u8, prompt: usize, tokens: usize, after: f64) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("ended {s} reason=", .{id}) catch {};
    if (why) |e| w.print("error ({s})", .{e}) catch {} else w.writeAll("client-left") catch {};
    w.print(" prompt={d} tokens={d} after={d:.2}s", .{ prompt, tokens, after }) catch {};
    return w.buffered();
}

test "a request's refused and ended lines" {
    var buf: [1400]u8 = undefined;
    const accents: [301][2]u8 = @splat(.{ 0xc3, 0xa9 }); // two bytes a character, so a byte cut would split one
    const cut = refused(&buf, "cmpl-1", std.mem.sliceAsBytes(&accents));
    try std.testing.expectEqualStrings("refused cmpl-1: ", cut[0..16]);
    try std.testing.expectEqualStrings(std.mem.sliceAsBytes(accents[0..300]), cut[16..]);
    try std.testing.expectEqualStrings("refused cmpl-1: short", refused(&buf, "cmpl-1", "short"));
    try std.testing.expectEqualStrings("ended chatcmpl-1 reason=client-left prompt=12 tokens=3 after=0.50s", ended(&buf, "chatcmpl-1", null, 12, 3, 0.5));
    try std.testing.expectEqualStrings("ended chatcmpl-1 reason=error (OutOfMemory) prompt=12 tokens=0 after=1.25s", ended(&buf, "chatcmpl-1", "OutOfMemory", 12, 0, 1.25));
}

test "security: a request line's control bytes reach the log as \\xNN, never as terminal controls" {
    var out: Escaped = undefined;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var conn = try Conn.init(std.testing.allocator, 0, "127.0.0.1");
    defer conn.deinit();
    // a method of escape sequences (clear the screen, set the title), DEL, and an 8-bit CSI as http_conn reads it
    const raw_line = "\x1b[2J\x1b]0;owned\x07\x7f\x9bGET /health HTTP/1.1";
    conn.requestline = try @import("http_conn.zig").latin1(arena.allocator(), raw_line);
    const access = render(&out, request_fmt, .{ conn.peer, conn.requestline, 501, keySuffix(&conn) });
    const want = "[tensorfold] 127.0.0.1 \"\\x1b[2J\\x1b]0;owned\\x07\\x7f\\x9bGET /health HTTP/1.1\" 501 -\n";
    try std.testing.expectEqualStrings(want, access);
    for (access[0 .. access.len - 1]) |b| try std.testing.expect(b >= 0x20 and b != 0x7f);
    // the ordinary line is as before; its own newline is the only control byte
    try std.testing.expectEqualStrings("[tensorfold] serving x\n", render(&out, "serving {s}", .{"x"}));
    // a carriage return mid-line cannot overwrite the line on screen
    try std.testing.expectEqualStrings("[tensorfold] a\\x0db\n", render(&out, "{s}", .{"a\rb"}));
    // UTF-8 text, including a 0xc2 lead byte that is not a C1 control, passes through
    try std.testing.expectEqualStrings("[tensorfold] café © 日本語\n", render(&out, "{s}", .{"café © 日本語"}));
    // a line past the buffer still ends with one newline
    const long: [9000]u8 = @splat('y');
    const cut = render(&out, "{s}", .{&long});
    try std.testing.expectEqual(@as(usize, line_max + 1), cut.len);
    try std.testing.expectEqual(@as(u8, '\n'), cut[cut.len - 1]);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, cut, "\n"));
}
