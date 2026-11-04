//! Incremental decoding of a custom tool's leading JSON input string.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Parser = struct {
    state: enum { prefix, value, passthrough, done } = .prefix,
    prefix: std.ArrayList(u8) = .empty,
    part: usize = 0,
    offset: usize = 0,
    escape: std.ArrayList(u8) = .empty,

    const parts = [_][]const u8{ "{", "\"input\"", ":", "\"" };

    pub fn feed(p: *Parser, a: Allocator, chunk: []const u8) Allocator.Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        for (chunk) |c| switch (p.state) {
            .prefix => {
                try p.prefix.append(a, c);
                if (c == parts[p.part][p.offset]) {
                    p.offset += 1;
                    if (p.offset == parts[p.part].len) {
                        p.part += 1;
                        p.offset = 0;
                        if (p.part == parts.len) {
                            p.prefix.clearRetainingCapacity();
                            p.state = .value;
                        }
                    }
                } else if (p.offset != 0 or !std.ascii.isWhitespace(c)) {
                    try out.appendSlice(a, p.prefix.items);
                    p.prefix.clearRetainingCapacity();
                    p.state = .passthrough;
                }
            },
            .value => {
                if (p.escape.items.len == 0) {
                    switch (c) {
                        '\\' => try p.escape.append(a, c),
                        '"' => p.state = .done,
                        else => try out.append(a, c),
                    }
                } else {
                    try p.escape.append(a, c);
                    try p.decode(a, &out);
                }
            },
            .passthrough => try out.append(a, c),
            .done => break,
        };
        return out.items;
    }

    pub fn finish(p: *Parser, a: Allocator) Allocator.Error![]const u8 {
        p.state = .done;
        const out = try std.mem.concat(a, u8, &.{ p.prefix.items, p.escape.items });
        p.prefix.clearRetainingCapacity();
        p.escape.clearRetainingCapacity();
        return out;
    }

    fn decode(p: *Parser, a: Allocator, out: *std.ArrayList(u8)) Allocator.Error!void {
        const raw = p.escape.items;
        if (raw.len == 2) {
            if (raw[1] == 'u') return;
            const c: ?u8 = switch (raw[1]) {
                '"', '\\', '/' => raw[1],
                'b' => 8,
                'f' => 12,
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                else => null,
            };
            if (c) |byte| try out.append(a, byte) else try out.appendSlice(a, raw);
        } else if (raw.len <= 6 and partial(raw)) {
            if (raw.len < 6) return;
            const code = unit(raw);
            if (code >= 0xd800 and code <= 0xdbff) return;
            if (code >= 0xdc00 and code <= 0xdfff) {
                try out.appendSlice(a, raw);
            } else try appendCode(a, out, code);
        } else if (raw.len > 6 and raw.len < 12 and partial(raw[6..])) {
            return;
        } else if (raw.len == 12 and partial(raw[6..]) and unit(raw[6..]) >= 0xdc00 and unit(raw[6..]) <= 0xdfff) {
            const code: u21 = 0x10000 + (@as(u21, unit(raw[0..6])) - 0xd800) * 0x400 + unit(raw[6..]) - 0xdc00;
            try appendCode(a, out, code);
        } else try out.appendSlice(a, raw);
        p.escape.clearRetainingCapacity();
    }

    fn partial(raw: []const u8) bool {
        if (raw.len > 6) return false;
        if (raw.len <= 2) return std.mem.startsWith(u8, "\\u", raw);
        if (!std.mem.startsWith(u8, raw, "\\u")) return false;
        for (raw[2..]) |c| if (!std.ascii.isHex(c)) return false;
        return true;
    }

    fn unit(raw: []const u8) u16 {
        return std.fmt.parseInt(u16, raw[2..], 16) catch unreachable;
    }

    fn appendCode(a: Allocator, out: *std.ArrayList(u8), code: u21) Allocator.Error!void {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(code, &buf) catch unreachable;
        try out.appendSlice(a, buf[0..n]);
    }
};

test "custom input preserves chunk boundaries, escapes and fallback" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ " { \"input\" : \"a\\nb\\u00e9\\uD83D\\uDE00\\\"\\\\\\/\\t\\r\\b\\f\" ,\"ignored\":1}", "a\nbé😀\"\\/\t\r\x08\x0c" },
        .{ "{\"other\":1}", "{\"other\":1}" },
        .{ "{\"inp", "{\"inp" },
        .{ "{\"input\":\"\\u12\"x\"}", "\\u12\"x" },
        .{ "{\"input\":\"\\uD83Dz\"}", "\\uD83Dz" },
        .{ "{\"input\":\"\\uDC00\\q\"}", "\\uDC00\\q" },
        .{ "{\"input\":\"tail\\uD83D", "tail\\uD83D" },
        .{ "{\"input\":\"x\\", "x\\" },
        .{ "{\"input\":\"raw é 😀\"}", "raw é 😀" },
    };
    for (cases) |c| {
        for (0..c[0].len + 1) |split| {
            var p: Parser = .{};
            const first = try p.feed(a, c[0][0..split]);
            const second = try p.feed(a, c[0][split..]);
            const last = try p.finish(a);
            try std.testing.expectEqualStrings(c[1], try std.mem.concat(a, u8, &.{ first, second, last }));
            try std.testing.expectEqualStrings("", try p.finish(a));
            try std.testing.expectEqualStrings("", try p.feed(a, "ignored"));
        }
        var p: Parser = .{};
        var one: std.ArrayList(u8) = .empty;
        for (0..c[0].len) |i| try one.appendSlice(a, try p.feed(a, c[0][i .. i + 1]));
        try one.appendSlice(a, try p.finish(a));
        try std.testing.expectEqualStrings(c[1], one.items);
    }
}

test "custom input emits only complete escapes incrementally" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: Parser = .{};
    try std.testing.expectEqualStrings("", try p.feed(a, "{\"inp"));
    try std.testing.expectEqualStrings("a", try p.feed(a, "ut\":\"a\\"));
    try std.testing.expectEqualStrings("\n", try p.feed(a, "n\\uD83D"));
    try std.testing.expectEqualStrings("", try p.feed(a, "\\uDE0"));
    try std.testing.expectEqualStrings("😀!", try p.feed(a, "0!\"}"));
    try std.testing.expectEqualStrings("", try p.finish(a));
}
