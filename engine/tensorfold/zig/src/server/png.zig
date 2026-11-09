//! A rendered PDF page as an 8-bit RGB PNG, which the vision family decodes itself (dsv41 picture.decodePng), so PDF
//! input needs no Pillow. `header` is the encoded page's first bytes: the count routes span a page from it alone.
const std = @import("std");
const Allocator = std.mem.Allocator;
const flate = std.compress.flate;

pub const signature = "\x89PNG\r\n\x1a\n";

/// The signature and the IHDR chunk: 8 + 4 (length) + 4 (type) + 13 (body) + 4 (CRC).
pub const header_len = 33;

pub const Size = struct { w: u32, h: u32 };

/// An encoded page's first `header_len` bytes: an 8-bit RGB image (colour type 2) of w x h, not interlaced.
pub fn header(w: u32, h: u32) [header_len]u8 {
    var out: [header_len]u8 = undefined;
    @memcpy(out[0..8], signature);
    var body: [13]u8 = undefined;
    std.mem.writeInt(u32, body[0..4], w, .big);
    std.mem.writeInt(u32, body[4..8], h, .big);
    @memcpy(body[8..], &[_]u8{ 8, 2, 0, 0, 0 }); // depth 8, RGB, deflate, adaptive filters, no interlace
    std.mem.writeInt(u32, out[8..12], 13, .big);
    @memcpy(out[12..16], "IHDR");
    @memcpy(out[16..29], &body);
    std.mem.writeInt(u32, out[29..33], crc("IHDR", &body), .big);
    return out;
}

/// The size a page's bytes declare when they start with `header`'s IHDR (its CRC checked), else null.
pub fn size(bytes: []const u8) ?Size {
    if (bytes.len < header_len or !std.mem.eql(u8, bytes[0..8], signature)) return null;
    if (std.mem.readInt(u32, bytes[8..12], .big) != 13 or !std.mem.eql(u8, bytes[12..16], "IHDR")) return null;
    const body = bytes[16..29];
    if (std.mem.readInt(u32, bytes[29..33], .big) != crc("IHDR", body)) return null;
    if (!std.mem.eql(u8, body[8..13], &[_]u8{ 8, 2, 0, 0, 0 })) return null;
    const s: Size = .{ .w = std.mem.readInt(u32, body[0..4], .big), .h = std.mem.readInt(u32, body[4..8], .big) };
    return if (s.w == 0 or s.h == 0) null else s;
}

/// The most bytes `encodeBgrx` writes for w x h: the filtered rows stored uncompressed, with every block, chunk and
/// trailer around them, and slack. Saturating, so a hostile header's size cannot overflow it.
pub fn bound(w: u32, h: u32) usize {
    const raw = @as(usize, h) *| (1 +| 3 *| @as(usize, w));
    return header_len +| raw +| raw / 256 +| 4096;
}

fn crc(kind: *const [4]u8, body: []const u8) u32 {
    var c = std.hash.Crc32.init();
    c.update(kind);
    c.update(body);
    return c.final();
}

fn chunk(a: Allocator, out: *std.ArrayList(u8), kind: *const [4]u8, body: []const u8) Allocator.Error!void {
    var be: [4]u8 = undefined;
    std.mem.writeInt(u32, &be, @intCast(body.len), .big);
    try out.appendSlice(a, &be);
    try out.appendSlice(a, kind);
    try out.appendSlice(a, body);
    std.mem.writeInt(u32, &be, crc(kind, body), .big);
    try out.appendSlice(a, &be);
}

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p: i16 = @as(i16, a) + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

/// One filtered row: libpng's heuristic, the filter whose bytes, read as signed, sum smallest.
fn filterRow(row: []const u8, prev: []const u8, out: []u8, tries: *[5][]u8) void {
    var best: u8 = 0;
    var best_sum: u64 = std.math.maxInt(u64);
    for (0..5) |f| {
        const t = tries[f];
        var sum: u64 = 0;
        for (row, 0..) |x, i| {
            const left: u8 = if (i >= 3) row[i - 3] else 0;
            const up = prev[i];
            const ul: u8 = if (i >= 3) prev[i - 3] else 0;
            t[i] = switch (f) {
                0 => x,
                1 => x -% left,
                2 => x -% up,
                3 => x -% @as(u8, @intCast((@as(u16, left) + up) / 2)),
                else => x -% paeth(left, up, ul),
            };
            const s: i8 = @bitCast(t[i]);
            sum += @abs(s);
        }
        if (sum < best_sum) {
            best_sum = sum;
            best = @intCast(f);
        }
    }
    out[0] = best;
    @memcpy(out[1..], tries[best]);
}

/// A PDFium BGRx bitmap (FPDFBitmap_BGRx: blue, green, red, unused) of w x h, rows `stride` bytes apart, as a PNG.
pub fn encodeBgrx(a: Allocator, bgrx: []const u8, w: u32, h: u32, stride: usize) error{ OutOfMemory, EncodeFailed }![]u8 {
    if (w == 0 or h == 0 or stride < @as(usize, w) * 4 or bgrx.len < stride * (h - 1) + @as(usize, w) * 4) return error.EncodeFailed;
    const row_len = @as(usize, w) * 3;
    const rows = try a.alloc(u8, 2 * row_len + (1 + row_len) + 5 * row_len);
    defer a.free(rows);
    var prev = rows[0..row_len];
    var cur = rows[row_len .. 2 * row_len];
    const line = rows[2 * row_len ..][0 .. 1 + row_len];
    var tries: [5][]u8 = undefined;
    for (&tries, 0..) |*t, i| t.* = rows[3 * row_len + 1 + i * row_len ..][0..row_len];
    @memset(prev, 0);

    var z: std.Io.Writer.Allocating = try .initCapacity(a, @max(row_len * h / 4, 64 << 10));
    defer z.deinit();
    const window = try a.alloc(u8, flate.max_window_len);
    defer a.free(window);
    const c = try a.create(flate.Compress);
    defer a.destroy(c);
    c.* = flate.Compress.init(&z.writer, window, .zlib, .level_4) catch return error.EncodeFailed;
    for (0..h) |y| {
        const src = bgrx[y * stride ..][0 .. @as(usize, w) * 4];
        for (0..w) |x| {
            cur[3 * x] = src[4 * x + 2];
            cur[3 * x + 1] = src[4 * x + 1];
            cur[3 * x + 2] = src[4 * x];
        }
        filterRow(cur, prev, line, &tries);
        c.writer.writeAll(line) catch return error.EncodeFailed;
        std.mem.swap([]u8, &prev, &cur);
    }
    c.finish() catch return error.EncodeFailed;
    const zdata = z.written();

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.ensureTotalCapacity(a, header_len + zdata.len + zdata.len / 65536 * 12 + 64);
    try out.appendSlice(a, &header(w, h));
    var at: usize = 0;
    while (at < zdata.len) {
        const n = @min(zdata.len - at, 1 << 16);
        try chunk(a, &out, "IDAT", zdata[at..][0..n]);
        at += n;
    }
    try chunk(a, &out, "IEND", "");
    return out.toOwnedSlice(a);
}

/// The RGB rows of an encodeBgrx PNG, inflated and unfiltered (for tests: the vision family's decoder is the served one).
fn decodeForTest(a: Allocator, bytes: []const u8) ![]u8 {
    const s = size(bytes) orelse return error.NotPng;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(a);
    var at: usize = header_len;
    while (at + 12 <= bytes.len) {
        const len = std.mem.readInt(u32, bytes[at..][0..4], .big);
        const kind = bytes[at + 4 .. at + 8];
        const body = bytes[at + 8 .. at + 8 + len];
        if (std.mem.readInt(u32, bytes[at + 8 + len ..][0..4], .big) != crc(kind[0..4], body)) return error.BadCrc;
        if (std.mem.eql(u8, kind, "IDAT")) try idat.appendSlice(a, body);
        at += 12 + len;
    }
    var raw: std.Io.Writer.Allocating = .init(a);
    defer raw.deinit();
    var in: std.Io.Reader = .fixed(idat.items);
    var d: flate.Decompress = .init(&in, .zlib, &.{});
    _ = try d.reader.streamRemaining(&raw.writer);
    const z = raw.written();
    const row_len = @as(usize, s.w) * 3;
    if (z.len != @as(usize, s.h) * (1 + row_len)) return error.BadLength;
    const px = try a.alloc(u8, row_len * s.h);
    errdefer a.free(px);
    for (0..s.h) |y| {
        const f = z[y * (1 + row_len)];
        const line = z[y * (1 + row_len) + 1 ..][0..row_len];
        const out = px[y * row_len ..][0..row_len];
        for (0..row_len) |i| {
            const left: u8 = if (i >= 3) out[i - 3] else 0;
            const up: u8 = if (y > 0) px[(y - 1) * row_len + i] else 0;
            const ul: u8 = if (y > 0 and i >= 3) px[(y - 1) * row_len + i - 3] else 0;
            out[i] = switch (f) {
                0 => line[i],
                1 => line[i] +% left,
                2 => line[i] +% up,
                3 => line[i] +% @as(u8, @intCast((@as(u16, left) + up) / 2)),
                4 => line[i] +% paeth(left, up, ul),
                else => return error.BadFilter,
            };
        }
    }
    return px;
}

test "a page's PNG starts with its header, and its pixels are the bitmap's in RGB order" {
    const a = std.testing.allocator;
    const w: u32 = 37;
    const h: u32 = 23;
    const stride: usize = w * 4 + 12; // PDFium pads rows
    const bgrx = try a.alloc(u8, stride * h);
    defer a.free(bgrx);
    var prng: std.Random.DefaultPrng = .init(7);
    for (bgrx, 0..) |*b, i| b.* = if ((i / 97) % 3 == 0) prng.random().int(u8) else @truncate(i / 5);
    const bytes = try encodeBgrx(a, bgrx, w, h, stride);
    defer a.free(bytes);
    try std.testing.expectEqualSlices(u8, &header(w, h), bytes[0..header_len]);
    try std.testing.expectEqual(Size{ .w = w, .h = h }, size(bytes).?);
    try std.testing.expectEqual(Size{ .w = w, .h = h }, size(&header(w, h)).?);
    try std.testing.expect(bytes.len <= bound(w, h));
    const rgb = try decodeForTest(a, bytes);
    defer a.free(rgb);
    for (0..h) |y| for (0..w) |x| {
        const src = bgrx[y * stride + 4 * x ..][0..3];
        try std.testing.expectEqualSlices(u8, &.{ src[2], src[1], src[0] }, rgb[(y * w + x) * 3 ..][0..3]);
    };
    // a white page compresses to a small fraction of its pixels
    @memset(bgrx, 0xff);
    const white = try encodeBgrx(a, bgrx, w, h, stride);
    defer a.free(white);
    try std.testing.expect(white.len < 200);
}

test "size reads only the header encodeBgrx writes" {
    var h = header(640, 480);
    try std.testing.expectEqual(Size{ .w = 640, .h = 480 }, size(&h).?);
    try std.testing.expectEqual(@as(?Size, null), size(h[0..32]));
    h[20] ^= 1; // the width changed under its CRC
    try std.testing.expectEqual(@as(?Size, null), size(&h));
    try std.testing.expectEqual(@as(?Size, null), size(&header(0, 480)));
    // the largest size a header can name saturates the bound rather than overflowing it
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), bound(0xFFFF_FFFF, 0xFFFF_FFFF));
}
