//! The PDF render worker spawned from the built tensorfold-native, as the server starts it, with PDFium from
//! TENSORFOLD_LIBPDFIUM (zig build -Dpdfium=FILE sets it); a library that does not load fails the test. Its pages
//! decode with the vision family's own PNG decoder, and a count route spans each page as generation holds it.
const std = @import("std");
const errors = @import("errors.zig");
const json = @import("json.zig");
const log = @import("log.zig");
const messages = @import("messages.zig");
const pdf = @import("pdf.zig");
const png = @import("png.zig");
const prompt = @import("prompt.zig");
const worker = @import("pdf_worker.zig");
const api = @import("engine_api");
const picture = @import("dsv41_picture");
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

fn useServer() void {
    log.init(std.testing.io, false);
    worker.server_exe = @import("pdf_worker_exe").path;
}

/// A refusal fails the test, a missing PDFium too: -Dpdfium named the library these tests must render with.
fn refused(cx: Cx) !void {
    if (cx.kind == .unimplemented and std.mem.indexOf(u8, cx.message, pdf.library) != null)
        std.debug.print("PDFium did not load from -Dpdfium (TENSORFOLD_LIBPDFIUM): {s}\n", .{cx.message})
    else
        std.debug.print("unexpected refusal: {s}\n", .{cx.message});
    return error.Refused;
}

const Page = struct { w: f32, h: f32, rotate: u32 = 0, content: []const u8 = "" };

/// A PDF of `pages` with a correct cross-reference table; Helvetica is the font of any text drawn.
fn buildPdf(a: Allocator, pages: []const Page) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var offsets: std.ArrayList(usize) = .empty;
    const n_objects = 3 + 2 * pages.len; // catalog, pages, font, then a page and its contents each
    try out.appendSlice(a, "%PDF-1.4\n");
    try offsets.append(a, out.items.len);
    try out.appendSlice(a, "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n");
    try offsets.append(a, out.items.len);
    try out.appendSlice(a, "2 0 obj << /Type /Pages /Kids [");
    for (0..pages.len) |i| try out.print(a, "{d} 0 R ", .{4 + 2 * i});
    try out.print(a, "] /Count {d} >> endobj\n", .{pages.len});
    try offsets.append(a, out.items.len);
    try out.appendSlice(a, "3 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n");
    for (pages, 0..) |p, i| {
        try offsets.append(a, out.items.len);
        try out.print(a, "{d} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 {d} {d}] /Rotate {d} /Resources << /Font << /F1 3 0 R >> >> /Contents {d} 0 R >> endobj\n", .{ 4 + 2 * i, p.w, p.h, p.rotate, 5 + 2 * i });
        try offsets.append(a, out.items.len);
        try out.print(a, "{d} 0 obj << /Length {d} >>\nstream\n{s}\nendstream endobj\n", .{ 5 + 2 * i, p.content.len, p.content });
    }
    const xref = out.items.len;
    try out.print(a, "xref\n0 {d}\n0000000000 65535 f \n", .{n_objects + 1});
    for (offsets.items) |off| try out.print(a, "{d:0>10} 00000 n \n", .{off});
    try out.print(a, "trailer << /Size {d} /Root 1 0 R >>\nstartxref\n{d}\n%%EOF\n", .{ n_objects + 1, xref });
    return out.items;
}

test "tensorfold-native PDF render worker: renders text and colour in its spawned worker, as the vision family's decoder reads it" {
    useServer();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // a red square (20..80 points from the bottom left) and black text along y = 150
    const doc = try buildPdf(a, &.{.{ .w = 200, .h = 200, .content = "1 0 0 rg 20 20 60 60 re f 0 g BT /F1 24 Tf 20 140 Td (TensorFold) Tj ET" }});
    var cx: Cx = .{ .a = a };
    var budget: pdf.Budget = .{};
    var pages: std.ArrayList([]const u8) = .empty;
    const deadline_ns = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + pdf.render_timeout_ns;
    worker.render(&cx, try worker.argv(a), doc, .render, 4, deadline_ns, &budget, &pages) catch |err| {
        try std.testing.expectEqual(error.Refused, err);
        return refused(cx);
    };
    try std.testing.expectEqual(@as(usize, 1), pages.items.len);
    try std.testing.expectEqual(pdf.Budget{ .pdf_pages_used = 1, .pixels_used = 400 * 400 }, budget);
    try std.testing.expectEqual(png.Size{ .w = 400, .h = 400 }, png.size(pages.items[0]).?);
    var rgb = try picture.decodePng(a, pages.items[0]);
    defer rgb.deinit(a);
    try std.testing.expectEqual(@as(usize, 400), rgb.w);
    const at = struct {
        fn f(r: picture.Rgb, x: usize, y: usize) [3]u8 {
            return r.px[(y * r.w + x) * 3 ..][0..3].*;
        }
    }.f;
    try std.testing.expectEqual([3]u8{ 255, 255, 255 }, at(rgb, 5, 5)); // the page's white
    try std.testing.expectEqual([3]u8{ 255, 0, 0 }, at(rgb, 100, 300)); // red, not blue: BGRx read in order
    var dark: usize = 0; // the text's band, rows 60 to 120 from the top
    for (60..120) |y| for (0..400) |x| {
        const p = at(rgb, x, y);
        if (@as(u32, p[0]) + p[1] + p[2] < 200) dark += 1;
    };
    try std.testing.expect(dark > 200);
}

test "tensorfold-native PDF render worker: libpdfium renders in the spawned worker and enforces request-wide page and pixel caps" {
    useServer();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var pages: std.ArrayList([]const u8) = .empty;
    var state: pdf.Pages = .{ .max_pages = 4 };
    pdf.expandBytes(&cx, pdf.minimal_two_page_pdf, &state, &pages) catch return refused(cx);
    try std.testing.expectEqual(@as(usize, 2), pages.items.len);
    try std.testing.expectEqual(pdf.Budget{ .pdf_pages_used = 2, .pixels_used = 2 * 400 * 400 }, state.budget);
    for (pages.items) |p| {
        var rgb = try picture.decodePng(a, p);
        defer rgb.deinit(a);
        try std.testing.expectEqual(@as(usize, 400), rgb.h);
    }
    // a second PDF in the same request counts on from the first: 2 + 2 pages past a limit of 3
    var second: Cx = .{ .a = a };
    state.max_pages = 3;
    try std.testing.expectError(error.Refused, pdf.expandBytes(&second, pdf.minimal_two_page_pdf, &state, &pages));
    try std.testing.expect(std.mem.indexOf(u8, second.message, "at most 3 pages") != null);
    try std.testing.expectEqual(@as(usize, 2), pages.items.len);
    // seventeen A4 pages pass the page limit but not the request's 32 megapixels, measured or drawn
    const a4: [17]Page = @splat(.{ .w = 595, .h = 842 });
    const many = try buildPdf(a, &a4);
    for ([_]pdf.Mode{ .measure, .render }) |mode| {
        var big: pdf.Pages = .{ .mode = mode, .max_pages = 64 };
        var big_cx: Cx = .{ .a = a };
        var out: std.ArrayList([]const u8) = .empty;
        try std.testing.expectError(error.Refused, pdf.expandBytes(&big_cx, many, &big, &out));
        try std.testing.expectEqualStrings(pdf.pixel_refusal, big_cx.message);
    }
    // a corrupt document is refused in the worker's words
    var corrupt: Cx = .{ .a = a };
    var none: pdf.Pages = .{};
    try std.testing.expectError(error.Refused, pdf.expandBytes(&corrupt, "%PDF-1.4\nnot really a document", &none, &pages));
    try std.testing.expect(std.mem.indexOf(u8, corrupt.message, "could not be opened") != null);
}

/// The vision family as the served lane plans a picture: generation by decoding it (dsv41 vision.prepareFn's decode,
/// plan, pad and patches), the count routes by its header (vision.countFn's picture.spanFromHeader).
const FamilyVision = struct {
    var ctx: u8 = 0;
    fn prepare(_: *anyopaque, a: Allocator, bytes: []const u8, problem: *[]const u8) api.Vision.PrepareError!api.Vision.Prepared {
        var rgb = picture.decodePng(a, bytes) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            problem.* = "image input: not a PNG";
            return error.BadImage;
        };
        defer rgb.deinit(a);
        var pic = picture.fromRgb(a, rgb, .{}) catch return error.OutOfMemory;
        defer pic.deinit(a);
        return .{ .tokens = @intCast(pic.grid.spanTokens()), .payload = &.{} };
    }
    fn count(_: *anyopaque, _: Allocator, bytes: []const u8, problem: *[]const u8) api.Vision.PrepareError!u32 {
        if (picture.spanFromHeader(bytes, .{})) |n| return @intCast(n);
        problem.* = "image input: no header size";
        return error.BadImage;
    }
    fn hook(with_count: bool) api.Vision {
        return .{ .ctx = &ctx, .placeholder = "<img>", .token = 9, .prepare = prepare, .count = if (with_count) count else null };
    }
};

test "tensorfold-native PDF render worker: a count route spans each PDF page as generation holds it" {
    useServer();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // A4 both ways, a strip, a rotated letter page, pages past the side and area caps, and a small one scaled up
    const doc = try buildPdf(a, &.{
        .{ .w = 595, .h = 842, .content = "BT /F1 18 Tf 72 720 Td (page one) Tj ET" },
        .{ .w = 842, .h = 595 },
        .{ .w = 100, .h = 50 },
        .{ .w = 612, .h = 792, .rotate = 90, .content = "0 0 1 rg 100 100 200 200 re f" },
        .{ .w = 3000, .h = 3000 },
        .{ .w = 200, .h = 2000 },
        .{ .w = 37.3, .h = 1234.5 },
    });
    const photo = try png.encodeBgrx(a, &(@as([64 * 48 * 4]u8, @splat(0x80))), 64, 48, 64 * 4);
    const request = try std.fmt.allocPrint(a,
        \\[{{"role": "user", "content": [{{"type": "text", "text": "Compare"}}, {{"type": "file", "file": {{"filename": "d.pdf", "file_data": "data:application/pdf;base64,{s}"}}}}, {{"type": "image_url", "image_url": {{"url": "data:image/png;base64,{s}"}}}}]}},
        \\ {{"role": "assistant", "content": "", "tool_calls": [{{"id": "t1", "type": "function", "function": {{"name": "read", "arguments": "{{}}"}}}}]}},
        \\ {{"role": "tool", "tool_call_id": "t1", "content": [{{"type": "file", "file": {{"file_data": "{s}"}}}}]}}]
    , .{ try pdf.base64Of(a, doc), try pdf.base64Of(a, photo), try pdf.base64Of(a, pdf.minimal_two_page_pdf) });
    const msgs = (try json.parse(a, request)).ok;
    var cx: Cx = .{ .a = a };
    var drawn: messages.Images = .{ .placeholder = "<img>", .pdf = .{ .mode = .render } };
    const generated = messages.normalizeImages(&cx, msgs, "system", false, &drawn) catch return refused(cx);
    pdf.page_cache.clear(log.io()); // the worker measures, not the page cache from the pages just drawn
    var measured: messages.Images = .{ .placeholder = "<img>", .pdf = .{ .mode = .measure } };
    const counted = try messages.normalizeImages(&cx, msgs, "system", false, &measured);
    // the same text, its placeholders in the same places: seven pages, the image, then two pages
    try std.testing.expectEqualStrings(try json.stringify(a, generated, .{}), try json.stringify(a, counted, .{}));
    try std.testing.expectEqual(@as(usize, 10), drawn.list.items.len);
    try std.testing.expectEqual(drawn.list.items.len, measured.list.items.len);
    try std.testing.expectEqual(drawn.pdf.?.budget, measured.pdf.?.budget);
    try std.testing.expectEqualStrings("Compare\n\n<img>\n\n<img>\n\n<img>\n\n<img>\n\n<img>\n\n<img>\n\n<img>\n\n<img>", generated.array[0].get("content").?.string);
    for (drawn.list.items, measured.list.items, 0..) |page, head, i| {
        if (i == 7) continue; // the image part itself, the same bytes both times
        try std.testing.expectEqual(png.header_len, head.len);
        try std.testing.expectEqualSlices(u8, head, page[0..png.header_len]); // a measured page is its drawn page's header
    }
    // the rendered prompt's placeholders, spanned by decoding (generation) and by header (the count routes)
    var ids: std.ArrayList(u32) = .empty;
    for (0..drawn.list.items.len) |_| try ids.appendSlice(a, &.{ 5, 9 });
    try ids.append(a, 6);
    const by_decode = try prompt.countedIds(&cx, std.testing.allocator, FamilyVision.hook(false), ids.items, drawn.list.items);
    const by_header = try prompt.countedIds(&cx, std.testing.allocator, FamilyVision.hook(true), ids.items, measured.list.items);
    try std.testing.expectEqualSlices(u32, by_decode.ids, by_header.ids);
    try std.testing.expectEqual(by_decode.image_tokens, by_header.image_tokens);
    try std.testing.expect(by_decode.image_tokens > 9 * 100);
}

test "tensorfold-native PDF render worker: a document sent again comes from the page cache, byte for byte, with no worker" {
    useServer();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    pdf.page_cache.clear(log.io());
    var a4: [6]Page = undefined;
    for (&a4, 0..) |*p, i| p.* = .{ .w = 595, .h = 842, .content = try std.fmt.allocPrint(a, "BT /F1 18 Tf 72 720 Td (page {d}) Tj ET 0 0 1 rg 72 {d} 200 100 re f", .{ i + 1, 100 + 50 * i }) };
    const doc = try buildPdf(a, &a4);
    // a fresh render, straight from a worker
    var cx: Cx = .{ .a = a };
    var fresh_budget: pdf.Budget = .{};
    var fresh: std.ArrayList([]const u8) = .empty;
    const deadline_ns = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + pdf.render_timeout_ns;
    worker.render(&cx, try worker.argv(a), doc, .render, 16, deadline_ns, &fresh_budget, &fresh) catch return refused(cx);
    // an agent client's turns: the first renders, the next two are hits that start no worker
    const before = worker.spawned.load(.monotonic);
    const hits0 = pdf.page_cache.hits; // clear() keeps the counters, so count from here
    for (0..3) |turn| {
        var state: pdf.Pages = .{ .max_pages = 16 };
        var pages: std.ArrayList([]const u8) = .empty;
        pdf.expandBytes(&cx, doc, &state, &pages) catch return refused(cx);
        try std.testing.expectEqual(before + 1, worker.spawned.load(.monotonic));
        try std.testing.expectEqual(fresh_budget, state.budget);
        try std.testing.expectEqual(@as(usize, 6), pages.items.len);
        for (fresh.items, pages.items) |f, p| try std.testing.expectEqualSlices(u8, f, p);
        if (turn == 0) try std.testing.expectEqual(hits0, pdf.page_cache.hits);
    }
    try std.testing.expectEqual(hits0 + 2, pdf.page_cache.hits);
    // the count route on a hit: each page's header, the same budget, no worker
    var counted: pdf.Pages = .{ .mode = .measure, .max_pages = 16 };
    var heads: std.ArrayList([]const u8) = .empty;
    pdf.expandBytes(&cx, doc, &counted, &heads) catch return refused(cx);
    try std.testing.expectEqual(before + 1, worker.spawned.load(.monotonic));
    try std.testing.expectEqual(fresh_budget, counted.budget);
    for (fresh.items, heads.items) |f, h| try std.testing.expectEqualSlices(u8, f[0..png.header_len], h);
}
