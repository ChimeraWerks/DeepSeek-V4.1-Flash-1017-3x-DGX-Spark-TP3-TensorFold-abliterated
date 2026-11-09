//! TENSORFOLD_REQUEST_LOG: each chat and completion body appended as one JSON line, images redacted, for replays.
const std = @import("std");
const json = @import("json.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Where a `file` or `input_file` part can carry a PDF's bytes (pdf.extractPartBytes reads each).
const file_fields = [_][]const u8{ "file_data", "data", "url", "file_url" };

/// The body with every image part's URL and every file part's data replaced (a PDF's bytes too), the rest kept.
fn redact(a: Allocator, v: Value) Allocator.Error!Value {
    switch (v) {
        .array => |items| {
            const out = try a.alloc(Value, items.len);
            for (items, out) |item, *slot| slot.* = try redact(a, item);
            return .{ .array = out };
        },
        .object => |o| {
            if (v.typeIs("image_url")) {
                const copy = try json.copyObject(a, o);
                const url = try json.newObject(a);
                try url.put(a, "url", .{ .string = "<redacted>" });
                try copy.put(a, "image_url", .{ .object = url });
                return .{ .object = copy };
            }
            if (v.typeIs("file") or v.typeIs("input_file")) {
                // the data in a nested file object (Chat) or on the part itself (input_file, also sent to Chat)
                const copy = try json.copyObject(a, o);
                for (file_fields) |k| if (copy.contains(k)) try copy.put(a, k, .{ .string = "<redacted>" });
                if (v.typeIs("file") or v.get("file") != null) {
                    const file = try json.newObject(a);
                    if (v.get("file")) |f| if (f == .object) if (f.get("filename")) |name| try file.put(a, "filename", name);
                    try file.put(a, "file_data", .{ .string = "<redacted>" });
                    try copy.put(a, "file", .{ .object = file });
                }
                return .{ .object = copy };
            }
            const out = try json.newObject(a);
            for (o.keys(), o.values()) |k, item| try out.put(a, k, try redact(a, item));
            return .{ .object = out };
        },
        else => return v,
    }
}

/// Appends ``body`` to ``path`` unless it is background work (batch jobs are not client traffic).
pub fn append(a: Allocator, path: []const u8, body: Value) void {
    if (body.strField("priority")) |p| if (std.mem.eql(u8, p, "background")) return;
    const line = json.stringify(a, redact(a, body) catch return, .{}) catch return;
    const pathz = a.dupeSentinel(u8, path, 0) catch return;
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, pathz, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true }, 0o644) catch return;
    defer _ = std.posix.system.close(fd);
    const framed = std.mem.concat(a, u8, &.{ line, "\n" }) catch return;
    _ = std.posix.system.write(fd, framed.ptr, framed.len);
}

test "the request log keeps image URLs and PDF bytes out" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // every place a Chat body can carry a PDF: a file object, a file part's own fields, input_file's fields
    const body = (try json.parse(a,
        \\{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,c2VjcmV0"}},
        \\ {"type": "file", "file": {"filename": "a.pdf", "file_data": "JVBERi1zZWNyZXQ=", "url": "data:application/pdf;base64,JVBERi1vbmU="}},
        \\ {"type": "file", "file_data": "JVBERi10d28="},
        \\ {"type": "input_file", "filename": "b.pdf", "file_data": "JVBERi10aHJlZQ=="},
        \\ {"type": "input_file", "file_url": "data:application/pdf;base64,JVBERi1mb3Vy"},
        \\ {"type": "input_file", "data": "JVBERi1maXZl", "url": "data:application/pdf;base64,JVBERi1zaXg="},
        \\ {"type": "text", "text": "keep me"}]}]}
    )).ok;
    const text = try json.stringify(a, try redact(a, body), .{});
    for ([_][]const u8{ "c2VjcmV0", "JVBERi1zZWNyZXQ=", "JVBERi1vbmU=", "JVBERi10d28=", "JVBERi10aHJlZQ==", "JVBERi1mb3Vy", "JVBERi1maXZl", "JVBERi1zaXg=" }) |secret|
        try std.testing.expect(std.mem.indexOf(u8, text, secret) == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "a.pdf") != null and std.mem.indexOf(u8, text, "b.pdf") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "keep me") != null and std.mem.indexOf(u8, text, "input_file") != null);
}
