//! PDF document input (on unless --no-vision-pdf): each page rendered by PDFium in a bounded worker process
//! (pdf_worker.zig) into a PNG the vision family decodes itself, so no Pillow is needed. The count routes ask the
//! same worker for each page's size at the render scale (no pixels drawn) and span the page from its PNG header,
//! which is the rendered page's own.
//! Ported from TensorFold upstream (joernchen, PR #1820; ashhart, PRs #2387 and #2393), JPEG pages made PNG.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const png = @import("png.zig");
const worker = @import("pdf_worker.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const env = "TENSORFOLD_LIBPDFIUM";
pub const pages_env = "TENSORFOLD_MAX_PDF_PAGES";
pub const library = if (builtin.os.tag == .macos) "libpdfium.dylib" else "libpdfium.so";
pub const max_pdf_bytes: usize = 32 << 20;
pub const max_request_pixels: u64 = 32 << 20;
pub const default_max_pages: u32 = 16;
/// TENSORFOLD_MAX_PDF_PAGES at most (upstream's --vision-max-images ceiling; this server has no image cap of its own).
pub const max_pages_cap: u32 = 64;
pub const render_timeout_ns: i96 = 10 * std.time.ns_per_s;
pub const max_child_heap_bytes: u64 = 1 << 30;
/// The most pixels a page side can have (pageDimensions' clamp): a worker's reply naming more is refused unread.
pub const max_page_side: u32 = 4096;
pub const off_message = "PDF inputs are off on this server; start the server without --no-vision-pdf";
pub const missing_message = library ++ " is required for PDF inputs; install PDFium (chromium/7725) in lib/ beside the server or set " ++ env ++ " to its file";
pub const url_refusal = "URLs are not accepted for PDF inputs; send inline base64 or a data:application/pdf;base64,... URL";

/// A request's PDF pages so far, across all its PDFs: the page and pixel limits are request-wide.
pub const Budget = struct { pdf_pages_used: u32 = 0, pixels_used: u64 = 0 };

/// What the worker gives for each page: its PNG (generation), or only that PNG's header, the page's size at the render
/// scale with no pixels drawn (the count routes).
pub const Mode = enum(u32) { render = 0, measure = 1 };

/// A request's PDF settings (messages.Images.pdf; null: PDFs are off) and its running budget.
pub const Pages = struct { mode: Mode = .render, max_pages: u32 = default_max_pages, budget: Budget = .{} };

pub const IfsdkPause = extern struct { version: c_int = 1, NeedToPauseNow: *const fn (*IfsdkPause) callconv(.c) c_int, user: ?*anyopaque = null };

pub const PdfiumFns = struct {
    FPDF_InitLibrary: *const fn () callconv(.c) void,
    FPDF_LoadMemDocument: *const fn ([*]const u8, c_int, ?[*:0]const u8) callconv(.c) ?*anyopaque,
    FPDF_CloseDocument: *const fn (?*anyopaque) callconv(.c) void,
    FPDF_GetPageCount: *const fn (?*anyopaque) callconv(.c) c_int,
    FPDF_LoadPage: *const fn (?*anyopaque, c_int) callconv(.c) ?*anyopaque,
    FPDF_ClosePage: *const fn (?*anyopaque) callconv(.c) void,
    FPDF_GetPageWidthF: *const fn (?*anyopaque) callconv(.c) f32,
    FPDF_GetPageHeightF: *const fn (?*anyopaque) callconv(.c) f32,
    FPDFBitmap_Create: *const fn (c_int, c_int, c_int) callconv(.c) ?*anyopaque,
    FPDFBitmap_Destroy: *const fn (?*anyopaque) callconv(.c) void,
    FPDFBitmap_GetBuffer: *const fn (?*anyopaque) callconv(.c) ?*anyopaque,
    FPDFBitmap_GetStride: *const fn (?*anyopaque) callconv(.c) c_int,
    FPDF_RenderPageBitmap: *const fn (?*anyopaque, ?*anyopaque, c_int, c_int, c_int, c_int, c_int, c_int) callconv(.c) void,
    FPDF_RenderPageBitmap_Start: ?*const fn (?*anyopaque, ?*anyopaque, c_int, c_int, c_int, c_int, c_int, c_int, *IfsdkPause) callconv(.c) c_int = null,
    FPDF_RenderPage_Continue: ?*const fn (?*anyopaque, *IfsdkPause) callconv(.c) c_int = null,
    FPDF_RenderPage_Close: ?*const fn (?*anyopaque) callconv(.c) void = null,
};

/// The PDFium file: TENSORFOLD_LIBPDFIUM when set (set but empty: none), else lib/ beside the server (its directory's
/// ../lib, then the directory itself). The loader's own search path is never used. Null when no such file exists.
pub fn libraryPath(a: Allocator, io: std.Io) ?[]const u8 {
    if (std.c.getenv(env)) |p| {
        const v = std.mem.span(p);
        if (v.len == 0 or !readable(a, v)) return null;
        return a.dupe(u8, v) catch null;
    }
    const exe = std.process.executableDirPathAlloc(io, a) catch return null;
    for ([_][]const []const u8{ &.{ exe, "..", "lib", library }, &.{ exe, library } }) |parts| {
        const path = std.fs.path.join(a, parts) catch continue;
        if (readable(a, path)) return path;
    }
    return null;
}

fn readable(a: Allocator, path: []const u8) bool {
    const z = a.dupeSentinel(u8, path, 0) catch return false;
    defer a.free(z);
    return std.c.access(z.ptr, std.c.R_OK) == 0;
}

/// PDFium's functions from the file libraryPath names (the worker's, after its limits), or null.
pub fn loadPdfium(a: Allocator, io: std.Io) ?PdfiumFns {
    const path = libraryPath(a, io) orelse return null;
    var dl = std.DynLib.open(path) catch return null;
    var f: PdfiumFns = undefined;
    const info = @typeInfo(PdfiumFns).@"struct";
    inline for (info.field_names, info.field_types) |name, FT| {
        if (@typeInfo(FT) == .optional) {
            @field(f, name) = dl.lookup(@typeInfo(FT).optional.child, name);
        } else {
            @field(f, name) = dl.lookup(FT, name) orelse {
                dl.close();
                return null;
            };
        }
    }
    return f; // kept loaded until the worker exits
}

/// Logs the PDFium file and its SHA-256 at startup while PDF input is on, or that it is missing. The server only
/// reads the file: PDFium is loaded by each render worker, never by the server process.
pub fn logStartup(a: Allocator, io: std.Io) void {
    const path = libraryPath(a, io) orelse return log.line("PDF inputs: {s}", .{missing_message});
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(128 << 20)) catch return log.line("PDF inputs: {s}", .{path});
    defer a.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    log.line("PDF inputs: {s} (sha256 {s}); at most {d} pages a request", .{ path, &hex, maxPagesLimit() });
}

/// The page limit across a request's PDFs: 16, or TENSORFOLD_MAX_PDF_PAGES from 1 up to max_pages_cap.
pub fn maxPagesLimit() u32 {
    if (std.c.getenv(pages_env)) |raw| if (std.fmt.parseInt(u32, std.mem.trim(u8, std.mem.span(raw), " "), 10) catch null) |n| if (n > 0) return @min(n, max_pages_cap);
    return default_max_pages;
}

fn imageUrl(part: Value) ?[]const u8 {
    const iu = part.get("image_url") orelse return null;
    if (iu == .string) return iu.string;
    if (iu == .object) return iu.strField("url");
    return null;
}

/// True for a PDF content part: `file` (Chat, and Messages' documents as translated), `input_file` (Responses), or an
/// `image_url` part whose URL is a data:application/pdf URL.
pub fn isPdfPart(part: Value) bool {
    if (part != .object) return false;
    const t = part.get("type") orelse return false;
    if (t != .string) return false;
    if (std.mem.eql(u8, t.string, "file") or std.mem.eql(u8, t.string, "input_file")) return true;
    if (!std.mem.eql(u8, t.string, "image_url")) return false;
    const url = imageUrl(part) orelse return false;
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, url, " \t\r\n"), "data:application/pdf");
}

/// A PDF part's pages appended to `out` (PNGs, or PNG headers in measure mode), within the request's budget.
pub fn expandPart(cx: *Cx, part: Value, pages: *Pages, out: *std.ArrayList([]const u8)) errors.Refused!void {
    try expandBytes(cx, try extractPartBytes(cx, part), pages, out);
}

/// The document bytes of a PDF part, from inline data only: a URL or a file id is refused.
pub fn extractPartBytes(cx: *Cx, part: Value) errors.Refused![]const u8 {
    if (part.get("image_url") != null) {
        const url = imageUrl(part) orelse return cx.refuse("image_url must be a url string or an object with one");
        return decodeFileData(cx, url);
    }
    const container: Value = if (part.get("file")) |f| (if (f == .object) f else return cx.refuse("file content part must contain a file object")) else part;
    if (container.get("file_id") != null and container.get("file_data") == null and container.get("data") == null and container.get("url") == null and container.get("file_url") == null)
        return cx.refuse("file parts require inline file_data (base64 or data:application/pdf;base64,...); file_id is not supported");
    if (container.strField("file_url") orelse container.strField("url")) |u| {
        const trimmed = std.mem.trim(u8, u, " \t\r\n");
        if (trimmed.len > 0 and !std.ascii.startsWithIgnoreCase(trimmed, "data:")) return cx.refuse(url_refusal);
    }
    const raw = container.strField("file_data") orelse container.strField("data") orelse container.strField("url") orelse container.strField("file_url") orelse
        return cx.refuse("file content part must contain file_data");
    if (raw.len == 0) return cx.refuse("file_data must be a non-empty base64 string or data URL");
    return decodeFileData(cx, raw);
}

/// Base64 or a data:application/pdf;base64 URL, decoded and checked to be a PDF of at most 32 MiB.
pub fn decodeFileData(cx: *Cx, raw: []const u8) errors.Refused![]const u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const is_data = std.ascii.startsWithIgnoreCase(trimmed, "data:");
    if (!is_data and std.mem.indexOf(u8, trimmed, "://") != null) return cx.refuse(url_refusal);
    var payload = trimmed;
    if (is_data) {
        const comma = std.mem.indexOfScalar(u8, trimmed, ',') orelse return cx.refuse("invalid PDF data URL");
        var it = std.mem.splitScalar(u8, trimmed[5..comma], ';');
        const mime = it.next() orelse "";
        if (mime.len > 0 and !std.ascii.eqlIgnoreCase(mime, "application/pdf") and !std.ascii.eqlIgnoreCase(mime, "application/octet-stream") and !std.ascii.eqlIgnoreCase(mime, "application/x-pdf"))
            return cx.refuse("file parts support application/pdf only");
        const base64 = while (it.next()) |p| {
            if (std.ascii.eqlIgnoreCase(p, "base64")) break true;
        } else false;
        if (!base64) return cx.refuse("PDF data URLs must be base64");
        payload = trimmed[comma + 1 ..];
    }
    const max_b64_len = 4 * ((max_pdf_bytes + 2) / 3);
    if (payload.len == 0) return cx.refuse("PDF is empty or exceeds the encoded byte limit");
    if (payload.len > max_b64_len + (max_b64_len / 16)) return cx.refuse("PDF exceeds the encoded byte limit");
    var clean: std.ArrayList(u8) = .empty;
    for (payload) |ch| if (!std.ascii.isWhitespace(ch)) {
        if (clean.items.len >= max_b64_len) return cx.refuse("PDF exceeds the encoded byte limit");
        try clean.append(cx.a, ch);
    };
    if (clean.items.len == 0) return cx.refuse("PDF is empty or exceeds the encoded byte limit");
    const d = std.base64.standard.Decoder;
    const n = d.calcSizeForSlice(clean.items) catch return cx.refuse("invalid PDF base64 encoding");
    if (n == 0 or n > max_pdf_bytes) return cx.refuse("PDF is empty or exceeds the encoded byte limit");
    const bytes = try cx.a.alloc(u8, n);
    d.decode(bytes, clean.items) catch return cx.refuse("invalid PDF base64 encoding");
    if (std.mem.indexOf(u8, bytes[0..@min(bytes.len, 1024)], "%PDF-") == null) return cx.refuse("file_data is not a valid PDF document");
    return bytes;
}

/// One PDF renders (or is measured) at a time: each worker is a whole process with its own address-space limit.
var render_mutex: std.Io.Mutex = .init;

/// `pdf_bytes`'s pages appended to `out` by a fresh worker, within the request-wide `pages.budget`.
pub fn expandBytes(cx: *Cx, pdf_bytes: []const u8, pages: *Pages, out: *std.ArrayList([]const u8)) errors.Refused!void {
    if (comptime builtin.os.tag == .windows) return cx.fail(.unimplemented, "{s}", .{"PDF inputs need the POSIX render worker"});
    const io = log.io();
    _ = libraryPath(cx.a, io) orelse return cx.fail(.unimplemented, "{s}", .{missing_message});
    render_mutex.lockUncancelable(io);
    defer render_mutex.unlock(io);
    const deadline_ns = std.Io.Clock.awake.now(io).toNanoseconds() + render_timeout_ns;
    const cmd = worker.argv(cx.a) catch return cx.refuse("PDF rendering worker process could not be started");
    try worker.render(cx, cmd, pdf_bytes, pages.mode, pages.max_pages, deadline_ns, &pages.budget, out);
}

const PauseState = struct {
    pause: IfsdkPause,
    deadline_ns: i96,
    timed_out: bool = false,
    fn check(p: *IfsdkPause) callconv(.c) c_int {
        const self: *PauseState = @alignCast(@fieldParentPtr("pause", p));
        if (std.Io.Clock.awake.now(log.io()).toNanoseconds() < self.deadline_ns) return 0;
        self.timed_out = true;
        return 1;
    }
};

/// FPDF_ANNOT; 0x10 is FPDF_REVERSE_BYTE_ORDER, which would hand the PNG encoder RGBx where it reads BGRx (#2386).
pub const render_flags: c_int = 0x01;

fn renderPageWithDeadline(cx: *Cx, f: PdfiumFns, bmp: ?*anyopaque, page: ?*anyopaque, w: c_int, h: c_int, idx: usize, deadline_ns: i96) errors.Refused!void {
    if (f.FPDF_RenderPageBitmap_Start != null and f.FPDF_RenderPage_Continue != null and f.FPDF_RenderPage_Close != null) {
        var state: PauseState = .{ .pause = .{ .version = 1, .NeedToPauseNow = PauseState.check }, .deadline_ns = deadline_ns };
        var status = f.FPDF_RenderPageBitmap_Start.?(bmp, page, 0, 0, w, h, 0, render_flags, &state.pause);
        while (status == 1) {
            if (state.timed_out or std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) {
                f.FPDF_RenderPage_Close.?(page);
                return cx.refuse("PDF rendering timed out");
            }
            status = f.FPDF_RenderPage_Continue.?(page, &state.pause);
        }
        f.FPDF_RenderPage_Close.?(page);
        if (state.timed_out or std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) return cx.refuse("PDF rendering timed out");
        if (status == 3) return cx.fail(.request, "PDF page {d} could not be rendered", .{idx + 1});
        return;
    }
    f.FPDF_RenderPageBitmap(bmp, page, 0, 0, w, h, 0, render_flags);
    if (std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) return cx.refuse("PDF rendering timed out");
}

pub fn pageLimitRefusal(cx: *Cx, limit: u32, total: u32) errors.Refused {
    return cx.fail(.request, "a PDF may have at most {d} pages ({s}); this request has {d}", .{ limit, pages_env, total });
}

pub const pixel_refusal = "the request's PDF pages exceed the 32-megapixel limit";

/// The worker's work: each page of the document in order, a PNG (render) or its header (measure), the pages and
/// pixels checked against the request-wide budget before the page that would cross a limit is drawn.
pub fn renderDocument(cx: *Cx, f: PdfiumFns, pdf_bytes: []const u8, mode: Mode, max_pages: u32, deadline_ns: i96, budget: *Budget, out: *std.ArrayList([]const u8)) errors.Refused!void {
    if (pdf_bytes.len > std.math.maxInt(c_int)) return cx.refuse("PDF exceeds the encoded byte limit");
    if (std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) return cx.refuse("PDF rendering timed out");
    const doc = f.FPDF_LoadMemDocument(pdf_bytes.ptr, @intCast(pdf_bytes.len), null) orelse
        return cx.refuse("PDF document could not be opened (corrupt or password-protected)");
    defer f.FPDF_CloseDocument(doc);
    const total_pages = f.FPDF_GetPageCount(doc);
    if (total_pages <= 0) return cx.refuse("PDF document contains no pages");
    const pages: u32 = @intCast(total_pages);
    const limit: u32 = @max(@as(u32, 1), max_pages);
    if (budget.pdf_pages_used +| pages > limit) return pageLimitRefusal(cx, limit, budget.pdf_pages_used +| pages);
    for (0..pages) |idx| {
        if (std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) return cx.refuse("PDF rendering timed out");
        const page = f.FPDF_LoadPage(doc, @intCast(idx)) orelse return cx.fail(.request, "PDF page {d} could not be rendered", .{idx + 1});
        defer f.FPDF_ClosePage(page);
        const dims = pageDimensions(f.FPDF_GetPageWidthF(page), f.FPDF_GetPageHeightF(page));
        const w: u32 = @intCast(dims[0]);
        const h: u32 = @intCast(dims[1]);
        const page_pixels = @as(u64, w) * h;
        if (budget.pixels_used + page_pixels > max_request_pixels) return cx.refuse(pixel_refusal);
        const bytes: []const u8 = switch (mode) {
            .measure => try cx.a.dupe(u8, &png.header(w, h)),
            .render => blk: {
                const bmp = f.FPDFBitmap_Create(dims[0], dims[1], 0) orelse return error.OutOfMemory; // 0: FPDFBitmap_BGRx
                defer f.FPDFBitmap_Destroy(bmp);
                const stride = f.FPDFBitmap_GetStride(bmp);
                const buf_ptr = f.FPDFBitmap_GetBuffer(bmp) orelse return error.OutOfMemory;
                if (stride < dims[0] * 4) return cx.refuse("PDF page bitmap has invalid stride");
                const bgrx: [*]u8 = @ptrCast(buf_ptr);
                const total_bytes = @as(usize, @intCast(stride)) * h;
                @memset(bgrx[0..total_bytes], 0xFF);
                try renderPageWithDeadline(cx, f, bmp, page, dims[0], dims[1], idx, deadline_ns);
                break :blk png.encodeBgrx(cx.a, bgrx[0..total_bytes], w, h, @intCast(stride)) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.EncodeFailed => return cx.refuse("rendered PDF page could not be encoded as PNG"),
                };
            },
        };
        budget.pdf_pages_used += 1;
        budget.pixels_used += page_pixels;
        try out.append(cx.a, bytes);
    }
}

/// A page's pixel size: 144 DPI (2 pixels a point), fitted within 2,048 pixels a side and 3.6 megapixels. The render
/// and measure modes both take it from the loaded page, so a count spans exactly the page generation draws.
pub fn pageDimensions(pw_raw: f32, ph_raw: f32) [2]c_int {
    const pw: f32 = if (std.math.isFinite(pw_raw) and pw_raw > 1.0) pw_raw else 595.0;
    const ph: f32 = if (std.math.isFinite(ph_raw) and ph_raw > 1.0) ph_raw else 842.0;
    var scale: f32 = 2.0; // 144 DPI baseline
    const max_side: f32 = 2048.0;
    const max_area: f32 = 3_600_000.0;
    if (pw * scale > max_side) scale = max_side / pw;
    if (ph * scale > max_side) scale = max_side / ph;
    if ((pw * scale) * (ph * scale) > max_area) scale = @sqrt(max_area / (pw * ph));
    const w: c_int = @max(28, @min(@as(c_int, max_page_side), @as(c_int, @intFromFloat(@round(pw * scale)))));
    const h: c_int = @max(28, @min(@as(c_int, max_page_side), @as(c_int, @intFromFloat(@round(ph * scale)))));
    return .{ w, h };
}

pub const minimal_two_page_pdf = "%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj\n4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000121 00000 n \n0000000190 00000 n \ntrailer << /Size 5 /Root 1 0 R >>\nstartxref\n259\n%%EOF\n";

pub fn base64Of(a: Allocator, bytes: []const u8) Allocator.Error![]u8 {
    const enc = std.base64.standard.Encoder;
    const out = try a.alloc(u8, enc.calcSize(bytes.len));
    _ = enc.encode(out, bytes);
    return out;
}

test "decodeFileData and parseChildReply validate untrusted input and recompute parent budgets" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const data_url = try std.fmt.allocPrint(a, "data:application/pdf;base64,{s}", .{try base64Of(a, minimal_two_page_pdf)});
    try std.testing.expectEqualSlices(u8, minimal_two_page_pdf, try decodeFileData(&cx, data_url));
    try std.testing.expectEqualSlices(u8, minimal_two_page_pdf, try decodeFileData(&cx, try base64Of(a, minimal_two_page_pdf)));
    for ([_][2][]const u8{
        .{ "https://example.com/doc.pdf", "URLs are not accepted" },
        .{ "data:image/png;base64,iVBORw0KGgo=", "application/pdf only" },
        .{ "data:application/pdf,%PDF-1.4", "must be base64" },
        .{ "aGVsbG8gd29ybGQ=", "not a valid PDF" },
        .{ "%%%", "invalid PDF base64" },
    }) |case| {
        var bad: Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, decodeFileData(&bad, case[0]));
        try std.testing.expect(std.mem.indexOf(u8, bad.message, case[1]) != null);
    }
    var bad_reply_cx: Cx = .{ .a = a };
    var budget: Budget = .{};
    var pages: std.ArrayList([]const u8) = .empty;
    try std.testing.expectError(error.Refused, worker.parseChildReply(&bad_reply_cx, &.{ 1, 0xFF, 3, 0, 0, 0, 'b', 'a', 'd' }, .render, 4, &budget, &pages));
    try std.testing.expectEqual(errors.Kind.request, bad_reply_cx.kind);
    try std.testing.expectEqualStrings("bad", bad_reply_cx.message);
}

test "PDF parts: file, input_file and data:application/pdf image_url parts, inline data only" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const b64 = try base64Of(a, minimal_two_page_pdf);
    const parts = (try json.parse(a, try std.fmt.allocPrint(a,
        \\[{{"type": "file", "file": {{"filename": "a.pdf", "file_data": "data:application/pdf;base64,{s}"}}}},
        \\ {{"type": "input_file", "file_data": "{s}"}},
        \\ {{"type": "image_url", "image_url": {{"url": "data:application/pdf;base64,{s}"}}}},
        \\ {{"type": "image_url", "image_url": {{"url": "data:image/png;base64,iVBORw0KGgo="}}}},
        \\ {{"type": "text", "text": "file"}}]
    , .{ b64, b64, b64 }))).ok.array;
    for (parts[0..3]) |p| {
        try std.testing.expect(isPdfPart(p));
        var cx: Cx = .{ .a = a };
        try std.testing.expectEqualSlices(u8, minimal_two_page_pdf, try extractPartBytes(&cx, p));
    }
    try std.testing.expect(!isPdfPart(parts[3]) and !isPdfPart(parts[4]));
    for ([_][]const u8{
        \\{"type": "file", "file": {"file_id": "file-abc"}}
        ,
        \\{"type": "input_file", "file_url": "https://example.com/doc.pdf"}
        ,
        \\{"type": "file", "file": "inline"}
    }) |raw| {
        var cx: Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, extractPartBytes(&cx, (try json.parse(a, raw)).ok));
    }
}

test "page sizes: 144 DPI within 2,048 pixels a side and 3.6 megapixels" {
    try std.testing.expectEqual([2]c_int{ 400, 400 }, pageDimensions(200, 200));
    try std.testing.expectEqual([2]c_int{ 1190, 1684 }, pageDimensions(595, 842)); // A4
    const big = pageDimensions(2000, 1000);
    try std.testing.expect(big[0] <= 2048 and big[1] <= 2048);
    try std.testing.expectEqual([2]c_int{ 1190, 1684 }, pageDimensions(std.math.nan(f32), 0));
    try std.testing.expect(@as(u64, @intCast(pageDimensions(1200, 1200)[0])) * @as(u64, @intCast(pageDimensions(1200, 1200)[1])) <= 3_600_000 + 4096);
}
