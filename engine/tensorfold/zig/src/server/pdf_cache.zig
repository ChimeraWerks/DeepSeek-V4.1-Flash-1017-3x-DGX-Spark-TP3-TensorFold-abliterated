//! Rendered PDF pages kept across requests: an agent client resends the same PDF with every turn, and the count route
//! measures it again before each generation. A document's pages are keyed by the SHA-256 of its bytes and everything
//! else its pages depend on, and held in a least-recently-used cache bounded in bytes (TENSORFOLD_PDF_CACHE_MIB,
//! 256 MiB by default, 0 off). Only pages that passed the parent's checks of the worker's reply are stored, and a hit
//! is admitted against the request's page and pixel budget exactly as a worker's reply is, without the render lock.
//! Refusals are never stored: they can depend on the request's budget, and a timeout can be transient.
const std = @import("std");
const errors = @import("errors.zig");
const log = @import("log.zig");
const pdf = @import("pdf.zig");
const png = @import("png.zig");
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const env = "TENSORFOLD_PDF_CACHE_MIB";
pub const default_mib: u64 = 256;

/// The PDFium file a worker loads, as far as a stat tells: a replaced library makes a new key.
pub const Library = struct { path: u64, inode: u64, size: u64, mtime_ns: i64 };

/// What a document's pages depend on: its bytes, what the worker gives (pages or headers), the page limit, the
/// render settings and limits built into this binary (pdf.render_fingerprint), and the library.
pub const Key = struct {
    doc: [32]u8,
    mode: pdf.Mode,
    max_pages: u32,
    render: u64 = pdf.render_fingerprint,
    library: Library,

    pub fn init(pdf_bytes: []const u8, mode: pdf.Mode, max_pages: u32, library: Library) Key {
        var k: Key = .{ .doc = undefined, .mode = mode, .max_pages = max_pages, .library = library };
        std.crypto.hash.sha2.Sha256.hash(pdf_bytes, &k.doc, .{});
        return k;
    }

    /// The key of `pdf_bytes` with the library at `lib_path`; null when that file cannot be stat'ed (nothing cached).
    pub fn of(io: std.Io, pdf_bytes: []const u8, mode: pdf.Mode, max_pages: u32, lib_path: []const u8) ?Key {
        const st = std.Io.Dir.cwd().statFile(io, lib_path, .{}) catch return null;
        const ns = st.mtime.nanoseconds;
        return init(pdf_bytes, mode, max_pages, .{
            .path = std.hash.Wyhash.hash(0, lib_path),
            .inode = @intCast(st.inode),
            .size = st.size,
            .mtime_ns = @intCast(std.math.clamp(ns, std.math.minInt(i64), std.math.maxInt(i64))),
        });
    }
};

const Entry = struct {
    key: Key,
    node: std.DoublyLinkedList.Node = .{},
    /// Each page a slice of `data`, in document order.
    pages: [][]const u8,
    data: []u8,
    /// What this entry holds in memory: its pages, their slices, itself, and an allocation header for each of the three.
    bytes: u64,
};

/// The map's slot for an entry (key, pointer, metadata byte), counted for every slot its capacity holds.
const slot_bytes = @sizeOf(Key) + @sizeOf(*Entry) + 1;
const alloc_header = 16;

/// The bytes an entry of `pages` holds (Entry.bytes).
pub fn entryBytes(pages: []const []const u8) u64 {
    var data: u64 = 0;
    for (pages) |p| data += p.len;
    return data + pages.len * @sizeOf([]const u8) + @sizeOf(Entry) + 3 * alloc_header;
}

/// The cap TENSORFOLD_PDF_CACHE_MIB names: 256 MiB when unset or not a whole number, 0 (no cache) when 0.
pub fn capFromEnv() u64 {
    if (std.c.getenv(env)) |raw| if (std.fmt.parseInt(u64, std.mem.trim(u8, std.mem.span(raw), " "), 10) catch null) |mib| return mib *| (1 << 20);
    return default_mib << 20;
}

pub const Cache = struct {
    gpa: Allocator,
    /// The bound on bytes(); null until first use, then capFromEnv().
    cap_bytes: ?u64 = null,
    mutex: std.Io.Mutex = .init,
    map: std.AutoHashMapUnmanaged(Key, *Entry) = .empty,
    /// Most recently used first.
    lru: std.DoublyLinkedList = .{},
    /// The entries' bytes (Entry.bytes), without the map.
    entries_bytes: u64 = 0,
    hits: u64 = 0,
    fills: u64 = 0,
    evictions: u64 = 0,

    pub fn deinit(c: *Cache) void {
        while (c.lru.popFirst()) |node| c.free(@fieldParentPtr("node", node));
        c.map.deinit(c.gpa);
        c.* = .{ .gpa = c.gpa, .cap_bytes = c.cap_bytes };
    }

    /// Every entry dropped; the map keeps its capacity.
    pub fn clear(c: *Cache, io: std.Io) void {
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        while (c.lru.popFirst()) |node| c.free(@fieldParentPtr("node", node));
        c.map.clearRetainingCapacity();
        c.entries_bytes = 0;
    }

    fn capLocked(c: *Cache) u64 {
        if (c.cap_bytes == null) c.cap_bytes = capFromEnv();
        return c.cap_bytes.?;
    }

    /// The cap in bytes; 0 when the cache is off.
    pub fn cap(c: *Cache, io: std.Io) u64 {
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        return c.capLocked();
    }

    /// What the cache holds in memory: its entries and the map's slots. Never past the cap after an insert.
    pub fn bytes(c: *const Cache) u64 {
        return c.entries_bytes + @as(u64, c.map.capacity()) * slot_bytes;
    }

    /// The pages of `key` admitted into the request as a worker's reply is (worker.parseChildReply): the page limit
    /// across the request, then each page's pixels, nothing kept on a refusal, each page copied into the request's
    /// allocator. A measure is also answered from the same document's rendered pages, by their PNG headers. False when
    /// neither is cached.
    pub fn admit(c: *Cache, cx: *Cx, key: Key, budget: *pdf.Budget, out: *std.ArrayList([]const u8)) errors.Refused!bool {
        const io = log.io();
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        var headers_only = false;
        const e = c.map.get(key) orelse blk: {
            if (key.mode != .measure) return false;
            var drawn = key;
            drawn.mode = .render;
            headers_only = true;
            break :blk c.map.get(drawn) orelse return false;
        };
        c.lru.remove(&e.node);
        c.lru.prepend(&e.node);
        c.hits += 1;
        try admitPages(cx, e.pages, headers_only, key.max_pages, budget, out);
        return true;
    }

    /// Keeps a copy of `pages` (already checked: a worker's reply that passed parseChildReply) under `key`, evicting
    /// the least recently used entries until it fits. Best effort: a document larger than the cap, or memory that
    /// cannot be had, leaves it out.
    pub fn put(c: *Cache, key: Key, pages: []const []const u8) void {
        const io = log.io();
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        const limit = c.capLocked();
        if (limit == 0 or pages.len == 0) return;
        if (c.map.get(key)) |e| {
            c.lru.remove(&e.node);
            c.lru.prepend(&e.node);
            return;
        }
        const size = entryBytes(pages);
        if (size > limit) return;
        c.map.ensureUnusedCapacity(c.gpa, 1) catch return;
        if (@as(u64, c.map.capacity()) * slot_bytes + size > limit) return;
        const e = c.make(key, pages, size) catch return;
        while (c.bytes() + size > limit) {
            const node = c.lru.popLast() orelse break;
            const old: *Entry = @fieldParentPtr("node", node);
            _ = c.map.remove(old.key);
            c.free(old);
            c.evictions += 1;
        }
        c.map.putAssumeCapacityNoClobber(key, e);
        c.lru.prepend(&e.node);
        c.entries_bytes += size;
        c.fills += 1;
    }

    fn make(c: *Cache, key: Key, pages: []const []const u8, size: u64) Allocator.Error!*Entry {
        var total: usize = 0;
        for (pages) |p| total += p.len;
        const data = try c.gpa.alloc(u8, total);
        errdefer c.gpa.free(data);
        const slices = try c.gpa.alloc([]const u8, pages.len);
        errdefer c.gpa.free(slices);
        const e = try c.gpa.create(Entry);
        var at: usize = 0;
        for (pages, slices) |p, *s| {
            @memcpy(data[at..][0..p.len], p);
            s.* = data[at..][0..p.len];
            at += p.len;
        }
        e.* = .{ .key = key, .pages = slices, .data = data, .bytes = size };
        return e;
    }

    /// An entry's memory, once it is out of the map and the list.
    fn free(c: *Cache, e: *Entry) void {
        c.entries_bytes -= e.bytes;
        c.gpa.free(e.pages);
        c.gpa.free(e.data);
        c.gpa.destroy(e);
    }
};

/// Checked pages admitted into a request: the request-wide page limit, then each page's pixels from its PNG header,
/// with the worker's own refusals; on a refusal neither `budget` nor `out` changes.
fn admitPages(cx: *Cx, pages: []const []const u8, headers_only: bool, max_pages: u32, budget: *pdf.Budget, out: *std.ArrayList([]const u8)) errors.Refused!void {
    const count: u32 = @intCast(pages.len);
    const limit: u32 = @max(@as(u32, 1), max_pages);
    if (budget.pdf_pages_used +| count > limit) return pdf.pageLimitRefusal(cx, limit, budget.pdf_pages_used +| count);
    var next = budget.*;
    const first = out.items.len;
    errdefer out.shrinkRetainingCapacity(first);
    for (pages) |page| {
        const s = png.size(page) orelse return cx.refuse("cached PDF page is not a PNG"); // checked before it was stored
        const px = @as(u64, s.w) * s.h;
        if (next.pixels_used + px > pdf.max_request_pixels) return cx.refuse(pdf.pixel_refusal);
        next.pdf_pages_used += 1;
        next.pixels_used += px;
        try out.append(cx.a, try cx.a.dupe(u8, if (headers_only) page[0..png.header_len] else page));
    }
    budget.* = next;
}

// ---- tests: pdf.expandCached with a fake worker, whose replies pass the parent's own checks

const worker = @import("pdf_worker.zig");

const Fake = struct {
    var spawns: std.atomic.Value(u32) = .init(0);
    var refuse: bool = false;

    /// A fake document "%PDF-<n><tag>": n pages of (40 + i) x 30 pixels filled from the whole document's bytes, sent
    /// through worker.parseChildReply as a worker's reply frame is.
    fn spawn(cx: *Cx, pdf_bytes: []const u8, mode: pdf.Mode, max_pages: u32, _: i96, budget: *pdf.Budget, out: *std.ArrayList([]const u8)) errors.Refused!void {
        _ = spawns.fetchAdd(1, .monotonic);
        if (refuse) return cx.refuse("PDF document could not be opened (corrupt or password-protected)");
        const n = pdf_bytes[5] - '0';
        const fill: u8 = @truncate(std.hash.Wyhash.hash(0, pdf_bytes));
        var pages: std.ArrayList([]const u8) = .empty;
        for (0..n) |i| {
            const w: u32 = 40 + @as(u32, @intCast(i));
            const px = try cx.a.alloc(u8, w * 30 * 4);
            @memset(px, fill +% @as(u8, @intCast(i)));
            const page = png.encodeBgrx(cx.a, px, w, 30, w * 4) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.EncodeFailed => return cx.refuse("encode"),
            };
            try pages.append(cx.a, if (mode == .measure) page[0..png.header_len] else page);
        }
        return worker.parseChildReply(cx, try worker.frameOf(cx.a, pages.items), mode, max_pages, budget, out);
    }

    fn spawned() u32 {
        return spawns.load(.monotonic);
    }
};

const test_lib: Library = .{ .path = 1, .inode = 2, .size = 3, .mtime_ns = 4 };

/// One request's pages of `doc` through `c` (null: no cache), its budget and output fresh.
fn expand(a: Allocator, c: ?*Cache, doc: []const u8, mode: pdf.Mode, max_pages: u32) !struct { pages: []const []const u8, budget: pdf.Budget } {
    var cx: Cx = .{ .a = a };
    var state: pdf.Pages = .{ .mode = mode, .max_pages = max_pages };
    var out: std.ArrayList([]const u8) = .empty;
    const cached: ?pdf.Cached = if (c) |cache| .{ .cache = cache, .key = Key.init(doc, mode, max_pages, test_lib) } else null;
    try pdf.expandCached(&cx, cached, doc, &state, &out, Fake.spawn);
    return .{ .pages = out.items, .budget = state.budget };
}

test "PDF page cache: a hit gives the pages a fresh render gives, byte for byte, and spawns no worker" {
    log.init(std.testing.io, false);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Cache = .{ .gpa = std.testing.allocator, .cap_bytes = 1 << 20 };
    defer c.deinit();
    const doc = "%PDF-5 five pages";
    const fresh = try expand(a, null, doc, .render, 16);
    const before = Fake.spawned();
    const miss = try expand(a, &c, doc, .render, 16);
    try std.testing.expectEqual(before + 1, Fake.spawned());
    const hit = try expand(a, &c, doc, .render, 16);
    try std.testing.expectEqual(before + 1, Fake.spawned()); // the hit spawned nothing
    try std.testing.expectEqual(@as(u64, 1), c.hits);
    try std.testing.expectEqual(@as(usize, 5), hit.pages.len);
    for (fresh.pages, miss.pages, hit.pages) |f, m, h| {
        try std.testing.expectEqualSlices(u8, f, m);
        try std.testing.expectEqualSlices(u8, f, h);
        try std.testing.expect(h.ptr != m.ptr); // the request's own copy, not the cache's memory
    }
    try std.testing.expectEqual(fresh.budget, hit.budget);
    try std.testing.expectEqual(entryBytes(miss.pages) + @as(u64, c.map.capacity()) * slot_bytes, c.bytes());
}

test "PDF page cache: a hit counts against the request's page and pixel budget as a worker's reply does" {
    log.init(std.testing.io, false);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Cache = .{ .gpa = std.testing.allocator, .cap_bytes = 1 << 20 };
    defer c.deinit();
    const doc = "%PDF-3 three";
    _ = try expand(a, &c, doc, .render, 4);
    const spawned = Fake.spawned();
    // two pages already in the request: three more pass a limit of 4 nowhere, in the worker's words, keeping nothing
    for ([_]bool{ false, true }) |cached| {
        var cx: Cx = .{ .a = a };
        var state: pdf.Pages = .{ .max_pages = 4, .budget = .{ .pdf_pages_used = 2, .pixels_used = 7 } };
        var out: std.ArrayList([]const u8) = .empty;
        const with: ?pdf.Cached = if (cached) .{ .cache = &c, .key = Key.init(doc, .render, 4, test_lib) } else null;
        try std.testing.expectError(error.Refused, pdf.expandCached(&cx, with, doc, &state, &out, Fake.spawn));
        try std.testing.expectEqualStrings("a PDF may have at most 4 pages (TENSORFOLD_MAX_PDF_PAGES); this request has 5", cx.message);
        try std.testing.expectEqual(pdf.Budget{ .pdf_pages_used = 2, .pixels_used = 7 }, state.budget);
        try std.testing.expectEqual(@as(usize, 0), out.items.len);
    }
    try std.testing.expectEqual(spawned + 1, Fake.spawned()); // only the uncached request spawned
    // the pixels: a request all but full takes no cached page
    var cx: Cx = .{ .a = a };
    var state: pdf.Pages = .{ .max_pages = 4, .budget = .{ .pixels_used = pdf.max_request_pixels - 100 } };
    var out: std.ArrayList([]const u8) = .empty;
    try std.testing.expectError(error.Refused, pdf.expandCached(&cx, .{ .cache = &c, .key = Key.init(doc, .render, 4, test_lib) }, doc, &state, &out, Fake.spawn));
    try std.testing.expectEqualStrings(pdf.pixel_refusal, cx.message);
    try std.testing.expectEqual(@as(usize, 0), out.items.len);
}

test "PDF page cache: different documents, limits, modes or libraries do not share entries; refusals are not kept" {
    log.init(std.testing.io, false);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Cache = .{ .gpa = std.testing.allocator, .cap_bytes = 1 << 20 };
    defer c.deinit();
    const doc = "%PDF-2 two";
    // a measure first: its entry holds headers and never answers a render
    _ = try expand(a, &c, doc, .measure, 16);
    var spawned = Fake.spawned();
    _ = try expand(a, &c, doc, .measure, 16);
    try std.testing.expectEqual(spawned, Fake.spawned());
    const drawn = try expand(a, &c, doc, .render, 16);
    try std.testing.expectEqual(spawned + 1, Fake.spawned());
    for (drawn.pages) |p| try std.testing.expect(p.len > png.header_len);
    // another page limit, document or library: a miss each
    spawned = Fake.spawned();
    _ = try expand(a, &c, doc, .render, 8);
    _ = try expand(a, &c, "%PDF-2 two!", .render, 16);
    try std.testing.expectEqual(spawned + 2, Fake.spawned());
    var other = Key.init(doc, .render, 16, test_lib);
    other.library.mtime_ns += 1;
    var cx: Cx = .{ .a = a };
    var budget: pdf.Budget = .{};
    var out: std.ArrayList([]const u8) = .empty;
    try std.testing.expect(!try c.admit(&cx, other, &budget, &out));
    other.library.mtime_ns -= 1;
    try std.testing.expect(try c.admit(&cx, other, &budget, &out));
    try std.testing.expect(Key.init(doc, .render, 16, test_lib).render == pdf.render_fingerprint);
    // a refusal is not kept: the same document is tried again
    Fake.refuse = true;
    defer Fake.refuse = false;
    spawned = Fake.spawned();
    for (0..2) |_| try std.testing.expectError(error.Refused, expand(a, &c, "%PDF-1 corrupt", .render, 16));
    try std.testing.expectEqual(spawned + 2, Fake.spawned());
}

test "PDF page cache: the count route and generation agree on a hit, measured from rendered pages or kept headers" {
    log.init(std.testing.io, false);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Cache = .{ .gpa = std.testing.allocator, .cap_bytes = 1 << 20 };
    defer c.deinit();
    const doc = "%PDF-4 four";
    const fresh_measure = try expand(a, null, doc, .measure, 16);
    const drawn = try expand(a, &c, doc, .render, 16);
    const spawned = Fake.spawned();
    const counted = try expand(a, &c, doc, .measure, 16); // answered from the rendered pages
    try std.testing.expectEqual(spawned, Fake.spawned());
    try std.testing.expectEqual(drawn.budget, counted.budget);
    try std.testing.expectEqual(fresh_measure.budget, counted.budget);
    for (drawn.pages, counted.pages, fresh_measure.pages) |page, head, fresh| {
        try std.testing.expectEqualSlices(u8, page[0..png.header_len], head);
        try std.testing.expectEqualSlices(u8, fresh, head);
    }
    // a measure kept on its own answers the next count the same way
    c.clear(log.io());
    _ = try expand(a, &c, doc, .measure, 16);
    const again = try expand(a, &c, doc, .measure, 16);
    try std.testing.expectEqual(spawned + 1, Fake.spawned());
    try std.testing.expectEqual(drawn.budget, again.budget);
    for (counted.pages, again.pages) |x, y| try std.testing.expectEqualSlices(u8, x, y);
}

test "PDF page cache: eviction keeps the byte bound, least recently used first" {
    log.init(std.testing.io, false);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = try expand(a, null, "%PDF-3 a", .render, 16);
    const size = entryBytes(one.pages);
    // room for three such documents and the map's eight slots, not four
    var c: Cache = .{ .gpa = std.testing.allocator, .cap_bytes = 3 * size + 8 * slot_bytes + size / 2 };
    defer c.deinit();
    const docs = [_][]const u8{ "%PDF-3 a", "%PDF-3 b", "%PDF-3 c", "%PDF-3 d", "%PDF-3 e", "%PDF-3 f" };
    for (docs[0..3]) |d| _ = try expand(a, &c, d, .render, 16);
    try std.testing.expectEqual(@as(u64, 3), c.fills);
    _ = try expand(a, &c, docs[0], .render, 16); // a is now the most recent; b the least
    _ = try expand(a, &c, docs[3], .render, 16);
    try std.testing.expectEqual(@as(u64, 1), c.evictions);
    try std.testing.expect(c.bytes() <= c.cap_bytes.?);
    var cx: Cx = .{ .a = a };
    var budget: pdf.Budget = .{};
    var out: std.ArrayList([]const u8) = .empty;
    try std.testing.expect(!try c.admit(&cx, Key.init(docs[1], .render, 16, test_lib), &budget, &out));
    try std.testing.expect(try c.admit(&cx, Key.init(docs[0], .render, 16, test_lib), &budget, &out));
    for (docs) |d| {
        _ = try expand(a, &c, d, .render, 16);
        try std.testing.expect(c.bytes() <= c.cap_bytes.?);
    }
    try std.testing.expectEqual(@as(u32, 3), c.map.count());
    // a document larger than the whole cap is not kept, and evicts nothing
    var small: Cache = .{ .gpa = std.testing.allocator, .cap_bytes = size - 1 };
    defer small.deinit();
    _ = try expand(a, &small, docs[0], .render, 16);
    try std.testing.expectEqual(@as(u64, 0), small.fills);
    try std.testing.expectEqual(@as(u64, 0), small.bytes());
    // off: nothing kept
    var off: Cache = .{ .gpa = std.testing.allocator, .cap_bytes = 0 };
    defer off.deinit();
    _ = try expand(a, &off, docs[0], .render, 16);
    try std.testing.expectEqual(@as(u64, 0), off.bytes());
}

test "PDF page cache: concurrent hits and fills are safe and render each document once" {
    log.init(std.testing.io, false);
    var c: Cache = .{ .gpa = std.testing.allocator, .cap_bytes = 1 << 20 };
    defer c.deinit();
    const docs = [_][]const u8{ "%PDF-4 w", "%PDF-3 x", "%PDF-2 y", "%PDF-5 z" };
    var expect_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer expect_arena.deinit();
    var want: [docs.len][]const []const u8 = undefined;
    for (docs, &want) |d, *w| w.* = (try expand(expect_arena.allocator(), null, d, .render, 16)).pages;
    const spawned = Fake.spawned();
    const Run = struct {
        fn run(cache: *Cache, wanted: *const [docs.len][]const []const u8, seed: usize, bad: *std.atomic.Value(u32)) void {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            for (0..200) |i| {
                const k = (seed + i) % docs.len;
                const mode: pdf.Mode = if (i % 3 == 0) .measure else .render;
                const got = expand(arena.allocator(), cache, docs[k], mode, 16) catch {
                    _ = bad.fetchAdd(1, .monotonic);
                    continue;
                };
                for (got.pages, wanted[k]) |g, w| if (!std.mem.eql(u8, g, if (mode == .measure) w[0..png.header_len] else w)) {
                    _ = bad.fetchAdd(1, .monotonic);
                };
                if (i % 50 == 49) _ = arena.reset(.retain_capacity);
            }
        }
    };
    var bad: std.atomic.Value(u32) = .init(0);
    var threads: [8]std.Thread = undefined;
    for (&threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Run.run, .{ &c, &want, i, &bad });
    for (threads) |t| t.join();
    try std.testing.expectEqual(@as(u32, 0), bad.load(.monotonic));
    // each document rendered once at most, measured once at most (before its render was kept)
    try std.testing.expect(Fake.spawned() - spawned <= 2 * docs.len);
    try std.testing.expect(c.hits >= 8 * 200 - 2 * docs.len);
    try std.testing.expect(c.bytes() <= c.cap_bytes.?);
}
