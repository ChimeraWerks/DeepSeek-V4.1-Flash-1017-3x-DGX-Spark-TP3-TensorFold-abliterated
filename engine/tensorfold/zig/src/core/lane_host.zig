//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
//!
//! Pending durable restores and continuation telemetry port wojo's GLM 0110 (PR #124; the scheduler hooks) and 0111
//! (PR #128). A request may be parked before its prompt pass on a restore job reading a longer kept state from the
//! prompt cache's tier (shared by every request that wants the same point). Parked requests hold no lane, count as
//! waiting work and wake back to the front of the queue, where an ordinary lookup runs again. With no tier nothing is
//! parked, nothing is serviced and nothing else changes.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const peer_watch = @import("peer_watch.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Status = api.Status;
const Memory = api.Memory;
const Reason = api.Reason;
const Stats = api.Stats;
const SubmitError = api.SubmitError;

const Share = @import("prefill_share.zig").Share;
const prefix = @import("lane_host_prefix.zig");

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(Id) = .empty,
    /// Engine-thread custody if cancellation cleanup exposes a poisoned backend. Never finish it after fatal.
    cancel_held: ?*Job = null,
    /// Intrusive, allocation-free custody of unfinished finalizations until fatal exit.
    finish_held: ?*Job = null,
    closing: bool = false,
    cache_reset: enum { idle, pending, done, unsupported } = .idle,
    thread: ?std.Thread = null,
    peer_probe: ?peer_watch.Probe = null,
    /// Host-only device health, independent of optional peer-monitor policy.
    device_probe: ?peer_watch.Probe = null,
    peer_monitor: ?peer_watch.Monitor = null,
    decoded: std.ArrayList(Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    prefill_telemetry: api.PrefillTelemetry = .{},
    prompt_cache: ?api.PromptCacheTelemetry = null, // a copy of the store's counts, taken on the engine thread
    adaptive: ?api.AdaptiveTelemetry = null,
    token_pool: ?lanes.backend.TokenPool = null,
    live_tokens: std.ArrayList(u32) = .empty,
    /// A Metal engine's keepalive target, set by the family that owns the queue; null keeps the ticker off.
    keepalive_target: ?api.keepalive.Target = null,
    live_generated: u64 = 0,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    /// Runs first on the lane thread, before anything else there: the backend makes its device context current (CUDA
    /// binds a context per thread, and admission can reach the device through the prompt cache's tier, a restore,
    /// before any backend call has bound it). null: nothing to do.
    enter: ?Enter = null,
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    memory: ?api.MemorySource = null, // the backend's memory counts; null: Engine.memory reports none
    explain: ?api.Explain = null, // the backend's words for a request it refuses; null: the error's name
    // the request at the queue's front waits for room in the backend's pool (its prompt pass refused ContextFull while
    // other streams ran): admitted again once fewer than this many streams are active
    full_wait: ?usize = null,

    // prefill interleave (TF_DS_PREFILL_INTERLEAVE_MS, 0/unset: off): between two chunks of a long prompt pass, decode
    // rounds for the other live streams for up to this many ms, their tokens delivered, so a cold long prompt slows a
    // chat instead of freezing it for the whole pass; the other jobs' cancels are taken there too (issue #3). The
    // filling job itself is left alone: its own cancel stays in `cancels` for its pass to see, admissions wait.
    interleave_ns: i96 = 0,
    filling: ?*Job = null,
    between_err: ?anyerror = null,
    share: Share = .{},
    slice_fill_at: i96 = 0, // end of the previous park, or chunk start
    slice_decode_ns: i64 = 0, // current long chunk: parks are decode occupancy, not prefill
    prefix: prefix.Wait = .{},
    ahead_on: bool = false,
    ahead_grid: ?Grid = null,
    ahead_ids: Id = 1 << 63,
    ahead_counts: AheadCounts = .{},
    ahead_sink: ?Sink = null,

    // -- parked requests (0110). The lists are the engine thread's; changes take the lock (status reads their lengths).
    ready: std.ArrayList(*Job) = .empty, // woken: admitted before the queue, in the order they woke
    loading: std.ArrayList(Load) = .empty, // each waits for a restore job (several may share one)
    turn: usize = 0, // which restore a service polls first (one budget, shared fairly)
    lone_commits: u32 = 0, // the lone driver's commits since its last pump
    tier: TierPolicy = .{},
    host_counts: HostCounts = .{},
    /// What status reports under `cache` (published on the engine thread at each noteLive; null until configured).
    tele: ?api.CacheStats = null,
    /// A backend's KV-ownership stats (its parts), read on the engine thread at each noteLive, published as `kv_tele`.
    kv_parts: ?lanes.kv_parts.Source = null,
    kv_tele: ?api.KvParts = null,

    const Mark = struct { at: i96, tokens: u64 };
    const window_ns: i96 = 2 * std.time.ns_per_s;
    /// How often an idle host polls work in flight (restores, staged writes, deferred retries).
    const poll_ns: i64 = 2 * std.time.ns_per_ms;
    /// The lone driver pumps staged writes every this many commits.
    const lone_pump_commits: u32 = 16;

    const Load = struct { job: *Job, restore: *pc.Restore, since: i96 };

    /// The backend's hook for the lane thread's start.
    pub const Enter = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) void };

    /// How parked work is serviced (LaneHost.configure; the defaults are off).
    pub const TierPolicy = struct {
        /// One elapsed-time budget per service for every restore together, then staged writes (TF_SPILL_LOAD_MS).
        service_ns: u64 = 40 * std.time.ns_per_ms,
        /// Staging's share of a service while streams decode (GLM pumped 4 blocks a round); idle, all that is left.
        pump_ns: u64 = 2 * std.time.ns_per_ms,
        /// How long a request waits parked on a restore before it is admitted without it (TF_SPILL_LOAD_WAIT_MS).
        load_wait_ns: i96 = 30 * std.time.ns_per_s,
        room_wait_ns: i96 = 0, // TF_SPILL_ROOM_WAIT_MS, nonsharded only
        /// A clean shutdown's bound for draining writes (TF_SPILL_FLUSH_MS).
        flush_ns: u64 = 60 * std.time.ns_per_s,
    };

    pub const HostCounts = struct {
        restore_room_waits: u64 = 0,
        restore_room_expiries: u64 = 0,
        restore_waits: u64 = 0, // requests parked on a restore
        restore_wait_timeouts: u64 = 0, // of them admitted without it at load_wait_ns
        restore_wait_ns: u64 = 0,
        restore_wait_max_ns: u64 = 0,
        cached_tokens: u64 = 0, // prompt tokens passes resumed (memory or restored)
        computed_tokens: u64 = 0, // and computed
    };

    pub const Job = struct {
        host: *LaneHost,
        finish_next: ?*Job = null,
        finish_retained: bool = false,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        delivered: usize = 0,
        started: bool = false,
        fill_began: bool = false,
        fill_wait: u32 = 0,
        fill_at: u32 = 0,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        began: i96 = 0,
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        reached: u32 = 0, // furthest mark reached, whether kept or not
        prefix: prefix.Waiting = .{},
        warm: ?*Warm = null,
        cache_request: pc.RequestState = .{},
        kept0: u64 = 0, // the store's kept count when the job looked it up
        room_since: ?i96 = null, // first ContextFull refusal, never renewed by a retry
        room_retry_at: i96 = 0,
        tried_disk: bool = false, // it waited for a restore once: a later lookup takes what memory has
        join: pc.Join = .{}, // its conversation (0111)
        parent: ?u64 = null, // the durable point its pass resumes from, and where
        fork: u32 = 0,
        cache_ready: bool = true, // submit validates image identity once; warm text jobs use the normal cache
        lineage: []pc.Fork = &.{}, // the computation its pass's rows belong to (gpa-owned; Store.passLineage)

        /// A completed CUDA event confirms a chunk, without making the engine wait for it.
        fn progress(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            const h = job.host;
            const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
            h.noteLive();
            h.lock();
            defer h.unlock();
            job.fill_at = at;
            h.fillGauges();
            if (now > job.began and at > s.cached) {
                h.prefill_rate = @as(f64, @floatFromInt(at - s.cached)) / (@as(f64, @floatFromInt(now - job.began)) / 1e9);
                h.prefill_at = now;
            }
        }

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reached = @max(job.reached, at);
            job.reported(); // before a keep can evict the entry the pass restored
            if (!job.cache_ready) return;
            const h = job.host;
            const store = h.cache orelse return;
            const r = job.request;
            const pics = r.images.len > 0;
            const key = if (pics) keyOf(r) else s.context.items;
            _ = store.keepWithRequestScoped(r.cache_namespace, key, at, s, r.chunks, .{
                .group = job.join.group, .images = pics, .media = if (pics) tierMedia(r) else &.{},
                .raw = if (pics) r.prompt else &.{}, .parent = job.parent, .fork = job.fork,
                .lineage = job.lineage,
            }, &job.cache_request);
        }

        /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
        fn reported(job: *Job) void {
            const e = job.entry orelse return;
            job.entry = null;
            const store = job.host.cache orelse return;
            defer store.unpin(e);
            if (!job.started) return;
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            const ok = !job.stream.reuse_failed and job.stream.cached == e.at;
            if (!ok) { // Never inherit a failed or mismatched restore's durable lineage.
                job.parent = null;
                job.fork = 0;
                if (job.lineage.len > 0) {
                    const fresh: []pc.Fork = store.passLineage(null) catch &.{};
                    job.host.gpa.free(job.lineage);
                    job.lineage = fresh;
                }
            }
            if (job.stream.reuse_failed) store.resumedRequest(e, job.request.cacheKey(), false, &job.cache_request)
            else if (ok) store.resumedRequest(e, job.request.cacheKey(), true, &job.cache_request);
        }
    };

    /// The pictures' identities the durable tier gets for a request: its `cache_media` only when they name exactly the
    /// spans the engine runs (`images`: the spans its restored rows and tower work are placed by), one for one; anything
    /// else gives none, and the tier then misses and keeps nothing for it (a disk hit must not resume pictures placed
    /// elsewhere than the point recorded them).
    pub fn tierMedia(r: *const Request) []const pc.Media {
        if (r.cache_media.len != r.images.len) return &.{};
        for (r.cache_media, r.images) |m, img| if (m.start != img.at or m.len != img.tokens) return &.{};
        return r.cache_media;
    }

    /// The ids the prompt cache keys a request by: its prompt, or with images the prompt with each picture's positions made
    /// from the picture's content (Request.cache_key).
    /// Image states require a complete, content-bound key and full media identities. A missing/forged key must not
    /// find an unrelated text state in RAM, or a disk point when only its shortened per-position ids happen to match.
    fn cacheable(r: *const Request) bool {
        if (r.images.len == 0) return true; // keep the text-only path exactly as it was
        if (r.cache_key.len != r.prompt.len or r.cache_media.len != r.images.len) return false;
        for (r.images, r.cache_media) |img, m| {
            if (m.start != img.at or m.len != img.tokens) return false;
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(img.bytes, &digest, .{});
            if (!std.mem.eql(u8, &digest, &m.digest)) return false;
        }
        return pc.mediaKeys(r.cache_key, r.cache_key.len, r.cache_media);
    }

    fn keyOf(r: *const Request) []const u32 {
        return if (r.cache_key.len == r.prompt.len) r.cache_key else r.prompt;
    }

    pub const Grid = struct { chunk: u32, tail: u32 };
    pub const AheadCounts = struct { queued: u64 = 0, dropped: u64 = 0, ran: u64 = 0, tokens: u64 = 0 };
    const Warm = struct { request: Request, tokens: []u32, from: u32, at: u32 };
    fn aheadQuiet(_: *anyopaque, _: Id, _: *const Event) void {}

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        enforced.cache_salt = true;
        var ms: i96 = 0;
        if (std.c.getenv("TF_DS_PREFILL_INTERLEAVE_MS")) |v| ms = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced, .share = Share.fromEnv(), .prefix = prefix.Wait.fromEnv(), .interleave_ns = ms * std.time.ns_per_ms };
    }

    /// The tier's switches (before start). Telemetry is published from here on when either is on: a tier on the cache
    /// (Store.setTier), or grouping.
    pub fn configure(h: *LaneHost, sp: api.Spill) void {
        h.tier.service_ns = @as(u64, @max(sp.load_ms, 1)) * std.time.ns_per_ms;
        h.tier.pump_ns = @min(h.tier.pump_ns, h.tier.service_ns);
        h.tier.load_wait_ns = @as(i96, @max(sp.load_wait_ms, 1)) * std.time.ns_per_ms;
        h.tier.room_wait_ns = @as(i96, sp.room_wait_ms) * std.time.ns_per_ms;
        h.tier.flush_ns = @as(u64, sp.flush_ms) * std.time.ns_per_ms;
        const store_on = if (h.cache) |s| s.tier != null or s.groups else false;
        h.tele = if (store_on) .{} else null;
    }

    pub fn start(h: *LaneHost) !void {
        if (h.peer_probe) |probe| {
            h.peer_monitor = .{ .io = h.io, .probe = probe, .policy = try peer_watch.Policy.fromEnv() };
            try h.peer_monitor.?.start();
        }
        errdefer if (h.peer_monitor) |*monitor| monitor.stop();
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what is left and joins the engine thread.
    pub fn stop(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
        if (h.peer_monitor) |*monitor| monitor.stop();
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
        h.ready.deinit(h.gpa);
        h.loading.deinit(h.gpa);
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .keepalive = keepaliveFn, .reset_cache = resetCacheFn } };
    }

    fn resetCacheFn(ctx: *anyopaque) api.ResetCacheError!void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        if (h.closing or h.thread == null) return error.Closed;
        if (h.cache_reset != .idle or h.queued.items.len != 0 or h.admitted.items.len != 0) return error.Busy;
        h.cache_reset = .pending;
        h.wake.broadcast(h.io);
        while (h.cache_reset == .pending and !h.closing) h.wake.waitUncancelable(h.io, &h.mutex);
        const result = h.cache_reset;
        h.cache_reset = .idle;
        if (h.closing) return error.Closed;
        if (result == .unsupported) return error.Unsupported;
        std.debug.assert(result == .done);
    }

    /// Only the engine thread may invoke snapshot drop callbacks (which can
    /// send distributed release messages). Submissions are refused until the
    /// waiting caller observes completion.
    fn takeCacheReset(h: *LaneHost) void {
        h.lock();
        defer h.unlock();
        if (h.cache_reset != .pending) return;
        std.debug.assert(h.queued.items.len == 0 and h.admitted.items.len == 0 and h.core.activeCount() == 0);
        if (h.cache) |store| store.clear() catch {
            h.cache_reset = .unsupported;
            h.wake.broadcast(h.io);
            return;
        };
        h.full_wait = null;
        h.cache_reset = .done;
        h.wake.broadcast(h.io);
    }

    /// The family's queue as a keepalive target, when it set one.
    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        return self(ctx).keepalive_target;
    }

    fn self(ctx: *anyopaque) *LaneHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        const h = self(ctx);
        // Cache salts are isolated in RAM. Durable namespaces are not yet part
        // of this store format, so never read or publish an unsalted disk point.
        if (request.cache_namespace != null) if (h.cache) |store| {
            if (store.tier != null) return error.Busy;
        };
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink, .cache_ready = cacheable(request) };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing or h.broken()) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        if (h.cache_reset != .idle) {
            h.gpa.destroy(job);
            return error.Busy;
        }
        // A foreground arrival cancels queued warmers before admission too.
        // Keep ownership until the engine thread performs normal job cleanup.
        for (h.queued.items) |a| if (a.warm != null) {
            h.cancels.append(h.gpa, a.id) catch {
                h.gpa.destroy(job);
                return error.Busy;
            };
        };
        for (h.admitted.items) |a| if (a.warm != null) {
            h.cancels.append(h.gpa, a.id) catch {
                h.gpa.destroy(job);
                return error.Busy;
            };
        };
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        if (h.peer_monitor) |*monitor| monitor.submitted();
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: Id) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn statusFn(ctx: *anyopaque, out: *Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= now - window_ns) tokens += m.tokens;
        }
        const n = @min(stream_tokens.len, h.live_tokens.items.len);
        @memcpy(stream_tokens[0..n], h.live_tokens.items[0..n]);
        out.* = .{
            .healthy = !h.broken(),
            .running = @intCast(h.admitted.items.len),
            .waiting = @intCast(h.queued.items.len + h.ready.items.len + h.loading.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = h.live_generated,
            .token_pool = h.token_pool,
            .prefill_telemetry = h.prefill_telemetry,
            .prompt_cache = h.prompt_cache,
            .adaptive = h.adaptive,
            .cache = h.tele,
            .kv_parts = h.kv_tele,
        };
    }

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?Memory {
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery.
    fn send(h: *LaneHost, job: *Job) void {
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            emit(job, .{ .tokens = emitted[job.delivered..] });
            h.noteDecoded(emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished (out of `admitted`, then freed: a submission walks the
    /// admitted jobs under the lock and must never meet a freed one).
    fn deliver(h: *LaneHost, job: *Job) bool {
        h.send(job);
        if (!job.stream.finished) return false;
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        const warm = if (reason == .stop or reason == .length) h.aheadOf(job) else null;
        h.remove(job);
        h.finish(job, reason, job.stream.problem);
        if (warm) |w| {
            if (h.broken()) h.finish(w, .cancelled, "") else h.queueAhead(w);
        }
        return true;
    }

    /// The step-ahead pass after `job`'s reply: null when off, when the reply moves no kept state a chunk on, or while
    /// other work is in the engine (it would compete with it).
    fn aheadOf(h: *LaneHost, job: *Job) ?*Job {
        if (!h.ahead_on or job.warm != null or !job.started or h.filling != null) return null;
        const g = h.ahead_grid orelse return null;
        const store = h.cache orelse return null;
        if (g.chunk == 0 or g.tail == 0) return null;
        const r = job.request;
        if (r.background or r.images.len > 0 or r.cache_key.len != 0 or r.prompt.len < store.rules.min_prompt) return null;
        const seq = job.stream.context.items; // prompt ++ reply: the next step's prompt starts so
        if (seq.len < @as(usize, g.chunk) + g.tail) return null;
        const at = (seq.len - g.tail) / g.chunk * g.chunk;
        const from: usize = if (store.findScoped(r.cache_namespace, seq[0 .. at + g.tail], &.{})) |e| e.at else 0;
        if (at <= from) return null;
        h.lock();
        const busy = h.closing or h.broken() or h.queued.items.len > 0 or h.ready.items.len > 0 or h.loading.items.len > 0 or h.admitted.items.len > 1;
        h.unlock();
        if (busy) return null;
        const w = h.gpa.create(Warm) catch return null;
        const tokens = h.gpa.dupe(u32, seq[0 .. at + g.tail]) catch {
            h.gpa.destroy(w);
            return null;
        };
        // one token, no drafts (the pass is all it is for), its history mark the kept point; nothing of the
        // reply's request (its slices go with it) beyond its sampling, a value
        w.* = .{ .tokens = tokens, .from = @intCast(from), .at = @intCast(at), .request = .{ .prompt = tokens, .max_tokens = 0, .drafts = false, .background = true, .history_len = @intCast(at), .cache_namespace = r.cache_namespace } };
        const wj = h.gpa.create(Job) catch {
            h.gpa.free(tokens);
            h.gpa.destroy(w);
            return null;
        };
        h.ahead_ids +%= 1;
        wj.* = .{ .host = h, .id = h.ahead_ids, .request = &w.request, .sink = h.ahead_sink orelse .{ .ctx = h, .event = aheadQuiet }, .warm = w };
        return wj;
    }

    /// Queues a step-ahead pass behind nothing; anything that arrived meanwhile wins and the pass goes.
    fn queueAhead(h: *LaneHost, job: *Job) void {
        h.lock();
        const ok = !h.closing and !h.broken() and h.queued.items.len == 0 and h.ready.items.len == 0 and h.loading.items.len == 0;
        if (ok) if (h.queued.append(h.gpa, job)) {
            h.ahead_counts.queued += 1;
            h.wake.signal(h.io);
            h.unlock();
            return;
        } else |_| {};
        h.unlock();
        h.finish(job, .cancelled, "");
    }


    fn retainFinish(h: *LaneHost, job: *Job) void {
        if (job.finish_retained) return;
        job.finish_retained = true;
        job.finish_next = h.finish_held;
        h.finish_held = job;
    }

    /// Consumes caller custody even on poison: retained jobs must never be retried or destroyed.
    /// Keep the void consuming API: callers may remove the old list reference after this returns.
    fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        if (job.finish_retained) return;
        if (h.broken()) return h.retainFinish(job);
        job.reported();
        // Final unpin can release a retired snapshot through tier/family callbacks.
        // entry is already null; never report again or touch job storage after poison.
        if (h.broken()) return h.retainFinish(job);
        h.gpa.free(job.marks);
        h.gpa.free(job.lineage);
        if (job.started) if (h.cache) |store| {
            const end = std.Io.Clock.awake.now(h.io).toNanoseconds();
            store.completedRequest(job.cache_request, job.stream.cached,
                @intCast(@max(0, end - job.began)),
                if (job.prefilled) |done| @intCast(@max(0, done - job.began)) else 0,
                switch (reason) { .failed => 2, .cancelled => 3, else => 1 });
        };
        const s = &job.stream;
        var stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .min_rows = s.min_rows, .loop_period = s.loop_period, .loop_novelty = s.loop_novelty, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        if (job.started) for (&stats.depths, s.depths) |*o, d| {
            o.* = .{ .drafted = d.drafted, .reached = d.reached, .kept = d.kept };
        };
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            if (s.grammar) |g| g.deinit(h.gpa);
            s.deinit(h.gpa);
            job.proposer.deinit();
        }
        if (h.peer_monitor) |*monitor| monitor.finished();
        if (job.warm) |w| {
            if (job.prefilled != null and (reason == .stop or reason == .length)) {
                h.ahead_counts.ran += 1;
                h.ahead_counts.tokens += w.at -| w.from;
            }
            h.gpa.free(w.tokens);
            h.gpa.destroy(w);
        }
        // A terminal sink can poison after the last health fence. Only retained jobs keep custody.
        if (h.cancel_held == job) h.cancel_held = null;
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *LaneHost, n: usize) void {
        if (h.peer_monitor) |*monitor| monitor.progress();
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < now - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
    }

    /// Cancels queued and admitted jobs named since the last round.
    fn takeCancels(h: *LaneHost) void {
        h.takeCancelsKeeping(null);
    }

    /// takeCancels, except a cancel of `keep` (the job in its prompt pass) stays in `cancels` for that pass to see.
    fn takeCancelsKeeping(h: *LaneHost, keep: ?Id) void {
        if (h.broken()) return;
        h.lock();
        // Only visit the original prefix. Producers may append while callbacks run unlocked;
        // those new cancellations remain queued for the next drain. Retained keep IDs
        // stay in place without allocating, including when the allocator is exhausted.
        var remaining = h.cancels.items.len;
        var cursor: usize = 0;
        while (remaining > 0) : (remaining -= 1) {
            const id = h.cancels.items[cursor];
            if (keep) |k| if (id == k) {
                cursor += 1;
                continue;
            };
            _ = h.cancels.orderedRemove(cursor);
            var restore: ?*pc.Restore = null;
            var admitted = false;
            const job: ?*Job = found: {
                for (h.queued.items, 0..) |job, i| if (job.id == id) break :found h.queued.orderedRemove(i);
                for (h.ready.items, 0..) |job, i| if (job.id == id) break :found h.ready.orderedRemove(i);
                for (h.loading.items, 0..) |l, i| if (l.job.id == id) {
                    _ = h.loading.orderedRemove(i);
                    // A shared restore belongs to the remaining waiters until the last
                    // one leaves. restoreCancel frees it, so never save duplicate pointers.
                    const wanted = for (h.loading.items) |other| {
                        if (other.restore == l.restore) break true;
                    } else false;
                    if (!wanted) restore = l.restore;
                    break :found l.job;
                };
                for (h.admitted.items, 0..) |job, i| if (job.id == id) {
                    admitted = true;
                    const taken = h.admitted.orderedRemove(i);
                    h.fillGauges();
                    break :found taken;
                };
                break :found null;
            };
            std.debug.assert(h.cancel_held == null);
            h.cancel_held = job;
            h.unlock();
            if (restore) |r| if (h.cache) |store| store.restoreCancel(r);
            // The sole detached job stays owned if restore/discard poisons the world.
            if (h.broken()) return;
            if (job) |j| {
                if (admitted and j.started) h.core.discard(&j.stream);
                if (h.broken()) return;
                h.finish(j, .cancelled, "");
                if (h.broken()) return;
            }
            h.cancel_held = null;
            if (h.broken()) return;
            h.lock();
        }
        h.unlock();
    }

    fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    /// Prefills the next queued request into a free lane; false when none waits or no lane is free.
    fn admitOne(h: *LaneHost) bool {
        h.lock();
        // Deadline retirement is safety work: do it even when capacity or decode debt blocks admission.
        if (!h.broken() and h.tier.room_wait_ns > 0) h.expireRoomWaits(std.Io.Clock.awake.now(h.io).toNanoseconds());
        if (h.broken() or (h.queued.items.len == 0 and h.ready.items.len == 0) or h.admitted.items.len >= h.info_.lanes) {
            h.unlock();
            return false;
        }
        if (h.full_wait) |n| {
            const active = h.residentCount(null);
            if (active > 0 and active >= n) {
                h.unlock();
                return false;
            }
            h.full_wait = null;
        }
        // restoreBegin can evict/guard sources and reserve/copy on the family. Leave ownership in the queue
        // until decode pays its debt; do not mark tried_disk or fall back to an uncached prefill.
        if (h.parking() and !h.tierWorkAllowed()) {
            h.unlock();
            return false;
        }
        // Keep the refused request at the queue front, but let completed restores leave ready first.
        // Throttle retries without waiting for resident count to change (paused/suspended residents count).
        if (h.ready.items.len == 0 and h.queued.items.len > 0) {
            const front = h.queued.items[0];
            if (front.room_since != null and h.residentCount(null) > 0 and
                std.Io.Clock.awake.now(h.io).toNanoseconds() < front.room_retry_at)
            {
                h.unlock();
                return false;
            }
        }
        const job = if (h.ready.items.len > 0) h.ready.orderedRemove(0) else blk: {
            prefix.release(h);
            while (true) {
                const at = for (h.queued.items, 0..) |candidate, i| {
                    if (!prefix.waiting(candidate)) break i;
                } else {
                    h.unlock();
                    return false;
                };
                if (!prefix.waits(h, h.queued.items[at], h.stepped())) break :blk h.queued.orderedRemove(at);
            }
        };
        if (job.warm != null and (h.admitted.items.len > 0 or h.core.activeCount() > 0)) {
            h.ahead_counts.dropped += 1;
            h.unlock();
            h.finish(job, .cancelled, "");
            return true;
        }
        if (h.parking()) {
            h.unlock();
            if (h.park(job)) return job.room_since == null; // room refusal ends this admission batch
            h.lock();
        }
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.unlock();
        if (h.broken()) return false; // retain admission ownership, but never enter cache/backend after fatal restore
        const r = job.request;
        var reuse: lanes.stream.Reuse = .{};
        // Scoped pins keep every admitted resume alive across later admissions, restores and keeps.
        if (h.cache) |store| {
            // Image states need a complete, content-bound key and full media identities, checked once at submit
            // (cache_ready). Without them bypass lookup as well as keep; a prior text/image entry is not authority.
            if (!job.cache_ready) {
                if (r.images.len > 0) store.noteImageBypass(r.prompt.len);
            } else if (store.lookupMediaScoped(r.cache_namespace, h.gpa, r.cacheKey(), r.history_len, r.shared_prefixes, r.chunks, r.images.len > 0, tierMedia(r))) |l| {
            job.entry = l.entry;
            if (l.entry) |e| store.pin(e);
            job.kept0 = store.counts.kept;
            job.marks = l.marks;
            job.cache_request = l.request;
            reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
            if (store.groups) job.join = store.join(keyOf(r), l.entry, r.images.len == 0 and r.drafts);
            if (l.entry) |e| {
                job.parent = e.point;
                job.fork = e.at;
            }
            // its rows' computation: the resumed state's forked at its position (written or not), else its own
            job.lineage = store.passLineage(l.entry) catch &.{};
            } else |_| {}
        }
        job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch return h.drop(job, "the drafter could not start");
        const grammar_state = if (r.structure) |st| h.constraint(st) catch |e| {
            job.proposer.deinit();
            return h.drop(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
        } else null;
        job.stream = lanes.Stream.init(h.gpa, .{
            .id = "request",
            .prompt = r.prompt,
            .max_new = r.max_tokens,
            .prefill_only = job.warm != null,
            .eos = r.eos,
            .sampling = r.sampling,
            .drafts = r.drafts,
            .proposer = job.proposer.proposer(),
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .loop_guard = r.loop_guard,
            .loop_dry_windows = r.loop_dry_windows,
            .call = if (r.call) |c| .{
                .opener = c.opener,
                .think_open = if (c.think_open) |t| t else -1,
                .think_end = if (c.think_end) |t| t else -1,
                .lead = c.lead,
                .names = c.names,
                .tail = c.tail,
                .lex = .{ .ctx = c.lex.ctx, .blank = c.lex.blank, .text = c.lex.text, .encode = c.lex.encode },
            } else null,
            .chunks = r.chunks,
            .reuse = reuse,
            .prefill_progress = .{ .ptr = job, .at = Job.progress },
            .images = r.images,
            .grammar = grammar_state,
        }) catch {
            if (grammar_state) |g| g.deinit(h.gpa);
            job.proposer.deinit();
            return h.drop(job, "out of memory");
        };
        job.started = true;
        const began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        job.began = began;
        h.lock();
        h.prefill_rate = 0;
        h.fillGauges();
        h.unlock();
        defer h.noteLive();
        if (h.stepped() and h.core.log == null) {
            job.stream.between = .{ .ptr = h, .wanted = sliceWanted, .run = sliceBetween };
            return true;
        } // image fills keep their uncached rows per slot
        if (h.loneFits(job)) return h.runLone(job, began);
        // between two prompt chunks: the other streams' rounds (interleave), and parked work's service, which runs
        // with the interleave off too
        if (r.images.len == 0 and (h.interleave_ns > 0 or h.servicing())) job.stream.between = .{ .ptr = h, .wanted = betweenWanted, .run = between };
        h.filling = job;
        defer h.filling = null;
        h.core.addStream(&job.stream) catch |e| {
            if (h.peer_monitor) |*monitor| monitor.signal(e);
            if (h.broken()) return false; // retain the world for hard exit; no backend cleanup
            if (e == error.Cancelled) return h.cancel(job);
            if (e == error.ContextFull and h.residentCount(job) > 0) return h.requeue(job);
            return h.drop(job, h.words(e));
        };
        h.prefilled(job, began);
        _ = h.deliver(job);
        return true;
    }

    fn sliceWanted(ptr: *anyopaque) bool {
        const h: *LaneHost = @ptrCast(@alignCast(ptr));
        return !h.broken() and h.core.log == null and h.core.runnableCount() > 0;
    }

    /// One round, or a bounded share payment. No cancellation draining, terminal release,
    /// snapshots, admission, placement, restore pumping or graph capture.
    fn sliceBetween(ptr: *anyopaque) anyerror!void {
        const h: *LaneHost = @ptrCast(@alignCast(ptr));
        if (h.broken()) return error.PeerDown;
        if (!sliceWanted(ptr)) return;
        const began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        defer {
            const ended = std.Io.Clock.awake.now(h.io).toNanoseconds();
            h.slice_decode_ns +|= @intCast(@max(0, ended - began));
            if (h.share.p < 1) h.slice_fill_at = ended;
        }
        // Local budget only: sliced charges the full chunk and all park payments once.
        var budget: Share = .{ .p = h.share.p };
        if (h.share.p < 1) budget.stepped(@intCast(@max(0, began - h.slice_fill_at)), true);
        h.core.hold_finished = true;
        defer h.core.hold_finished = false;
        while (true) {
            h.core.step() catch |err| {
                if (h.peer_monitor) |*monitor| monitor.signal(err);
                h.between_err = err;
                return err;
            };
            if (h.broken()) return error.PeerDown;
            for (h.admitted.items) |job| {
                if (job == h.filling or job.prefilled == null) continue;
                h.send(job); // streaming only; finish/release waits until the chunk closes
                if (h.broken()) return error.PeerDown;
            }
            if (h.share.p >= 1) break;
            if (!sliceWanted(ptr)) break;
            if (h.filling) |job| if (job.stream.isCancelled()) break;
            const spent: i64 = @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - began));
            if (spent >= budget.debt_ns or (h.share.park_max_ns > 0 and spent >= h.share.park_max_ns)) break;
        }
    }

    /// A complete chunk (or ordered abort) releases and finishes decoders
    /// that settled during a park. Otherwise an empty core could strand them.
    fn finishSlices(h: *LaneHost) !void {
        try h.core.settle();
        if (h.broken()) return error.PeerDown;
        var i: usize = 0;
        while (i < h.admitted.items.len) {
            const job = h.admitted.items[i];
            if (job == h.filling or job.prefilled == null or !job.stream.finished) { i += 1; continue; }
            if (!h.deliver(job)) i += 1; // a finished job left `admitted` inside deliver
            if (h.broken()) return error.PeerDown;
        }
    }

    fn betweenWanted(ptr: *anyopaque) bool {
        const h: *LaneHost = @ptrCast(@alignCast(ptr));
        if (h.broken()) return false;
        if (h.core.log != null) return false;
        return (h.interleave_ns > 0 and h.core.runnableCount() > 0) or h.pending();
    }

    /// Stream.between: decode rounds for the other live streams, interleave_ns at most, between two of the filling
    /// job's prompt chunks; their tokens go out as they land. Then parked work's service. Engine thread only (inside
    /// the filling job's addStream).
    fn between(ptr: *anyopaque) anyerror!void {
        const h: *LaneHost = @ptrCast(@alignCast(ptr));
        if (h.broken()) return error.PeerDown;
        if (h.core.log != null) return; // a step resets the core's arena, which a trail's open events live in
        if (h.filling) |job| job.reported();
        if (h.interleave_ns > 0) {
            const t0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
            while (true) {
                h.takeCancelsKeeping(if (h.filling) |f| f.id else null);
                if (h.broken()) return error.PeerDown;
                if (h.core.runnableCount() == 0 or std.Io.Clock.awake.now(h.io).toNanoseconds() - t0 >= h.interleave_ns) break;
                if (h.core.learner) |l| l.advanceTo(@intCast(std.Io.Clock.awake.now(h.io).toNanoseconds()));
                h.core.step() catch |e| {
                    if (h.peer_monitor) |*monitor| monitor.signal(e);
                    h.between_err = e; // the pass fails with it; run() then fails the streams the round held
                    return e;
                };
                var i: usize = 0;
                while (i < h.admitted.items.len) {
                    const job = h.admitted.items[i];
                    if (job == h.filling or job.prefilled == null) {
                        i += 1;
                        continue;
                    }
                    if (!h.deliver(job)) i += 1; // a finished job left `admitted` inside deliver
                    if (h.broken()) return error.PeerDown;
                }
                h.noteLive();
            }
        }
        if (h.servicing()) {
            if (h.filling) |f| f.reported(); // its restore is done (a chunk ran): a restore's room may evict that entry now
            h.service(h.tier.service_ns);
            if (h.broken()) return error.PeerDown;
        }
    }

    fn constraint(h: *LaneHost, st: api.Structure) !*lanes.grammar.Constraint {
        const c = st.compiler orelse return error.NoGrammarCompiler;
        const k = try c.constraint(h.gpa, h.io, @fromBackingInt(@intCast(@backingInt(st.kind))), st.text, st.after);
        errdefer k.deinit(h.gpa);
        if (st.ban.len > 0) k.ban = try h.gpa.dupe(u32, st.ban);
        return k;
    }

    fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        const cached = @min(job.request.prompt.len, job.stream.cached);
        h.prefill_telemetry.input += job.request.prompt.len;
        h.prefill_telemetry.cached += cached;
        h.prefill_telemetry.computed += job.request.prompt.len - cached;
        h.prefill_telemetry.seconds += @as(f64, @floatFromInt(@max(0, done - began))) / 1e9;
        h.prefill_telemetry.count += 1;
        h.fillGauges();
        h.unlock();
        job.reported();
        h.host_counts.cached_tokens += job.stream.cached;
        h.host_counts.computed_tokens += job.request.prompt.len -| job.stream.cached;
        if (h.cache) |store| {
            store.continued(job.join, job.stream.cached, job.request.prompt.len);
            store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
        }
        emit(job, .{ .prefilled = job.stream.cached });
    }

    /// An idle backend driver takes a lone drafted request, including sampling when supported.
    fn loneFits(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        const lone = h.lone orelse return false;
        if (job.warm != null) return false;
        if ((r.sampling != null and !lone.sampled) or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null) return false;
        h.lock();
        defer h.unlock();
        return h.admitted.items.len == 1 and h.queued.items.len == 0 and h.cancels.items.len == 0 and h.core.activeCount() == 0 and h.waitingCount() == 0;
    }

    /// Requests parked or woken (the lock held).
    fn waitingCount(h: *const LaneHost) usize {
        return h.ready.items.len + h.loading.items.len;
    }

    /// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
    fn runLone(h: *LaneHost, job: *Job, began: i96) bool {
        h.lone_job = job;
        job.delivered = 0;
        const lone = h.lone.?;
        const Hooks = struct {
            fn committed(ctx: *anyopaque) void {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                if (host.broken()) return;
                const j = host.lone_job.?;
                if (!j.prefill_sent) {
                    j.prefill_sent = true;
                    host.prefilled(j, j.began);
                }
                host.send(j);
                host.noteLive();
                // writes enqueued at keep are staged during a long lone decode too (parked work makes it yield)
                if (lonePumpDue(&host.lone_commits, host.servicing())) {
                    if (!host.broken()) if (host.cache) |store| store.pump(host.tier.pump_ns);
                }
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.queued.items.len > 0 or host.cancels.items.len > 0 or host.closing or host.broken() or host.waitingCount() > 0;
            }
        };
        job.began = began;
        const paused = lone.run(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
        h.lone_job = null;
        const handed = paused catch |e| {
            if (h.peer_monitor) |*monitor| monitor.signal(e);
            if (h.broken()) return false;
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            h.remove(job);
            h.finish(job, if (e == error.Cancelled) .cancelled else .failed, if (e == error.Cancelled) "" else h.words(e));
            return true;
        };
        if (h.broken()) return false; // committed hooks can fail even when the lone driver succeeds
        if (!job.prefill_sent) h.prefilled(job, began);
        if (handed) {
            h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
            h.send(job);
            return true;
        }
        _ = h.deliver(job);
        return true;
    }

    fn words(h: *const LaneHost, e: anyerror) []const u8 {
        const x = h.explain orelse return @errorName(e);
        return x.text(x.ctx, e) orelse @errorName(e);
    }

    fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        if (h.broken()) return false;
        h.remove(job);
        if (job.started) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// A request whose prompt pass the backend's pool cannot hold while other streams run (error.ContextFull)
    /// discards its attempt and goes back to the queue's front until one of them ends, as the Python
    /// lane's window waits, instead of failing. Returns false: nothing more is admitted now.
    fn requeue(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.core.discard(&job.stream);
        if (job.entry) |e| if (h.cache) |store| store.unpin(e);
        job.entry = null; // the discarded attempt no longer owns the scoped borrow
        h.gpa.free(job.marks);
        job.marks = &.{};
        if (job.stream.grammar) |g| g.deinit(h.gpa);
        job.stream.deinit(h.gpa);
        job.proposer.deinit();
        job.started = false;
        job.fill_began = false;
        job.fill_wait = 0;
        job.fill_at = 0; // progress belongs to the destroyed attempt, not the retry
        job.reached = 0;
        job.began = 0;
        job.prefilled = null;
        job.delivered = 0;
        job.prefill_sent = false;
        job.join = .{};
        job.parent = null;
        job.fork = 0;
        h.gpa.free(job.lineage);
        job.lineage = &.{};
        h.lock();
        h.queued.insert(h.gpa, 0, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.full_wait = h.residentCount(null);
        h.unlock();
        return false;
    }

    /// A job cancelled in its prompt pass, its lane already released.
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    /// All outstanding admitted fills, not just the last scheduled quantum. Caller holds the lock.
    fn fillGauges(h: *LaneHost) void {
        h.prefill_telemetry.active = 0;
        h.prefill_telemetry.progress = 0;
        h.prefill_telemetry.pending = 0;
        for (h.admitted.items) |j| {
            if (!j.started or j.prefilled != null) continue;
            h.prefill_telemetry.active += 1;
            h.prefill_telemetry.progress += j.fill_at;
            h.prefill_telemetry.pending += j.request.prompt.len -| j.fill_at;
        }
    }

    fn remove(h: *LaneHost, job: *Job) void {
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            h.fillGauges();
            return;
        };
    }

    fn noteLive(h: *LaneHost) void {
        if (h.broken()) return;
        if (h.peer_monitor) |*monitor| monitor.progress();
        const pool = if (h.core.backend.vtable.token_pool) |read| read(h.core.backend.ptr) else null;
        const tele = h.telemetry();
        var parts: ?api.KvParts = null;
        if (h.kv_parts) |src| {
            parts = .{};
            src.read(src.ctx, &parts.?);
        }
        h.lock();
        defer h.unlock();
        h.token_pool = pool;
        if (h.cache) |s| h.prompt_cache = .{
            .used_bytes = s.held,
            .limit_bytes = s.budget,
            .states = s.entries.items.len,
            .hits = s.counts.hits,
            .misses = s.counts.misses,
            .evicted = s.counts.evicted,
        };
        if (h.core.learner) |l| h.adaptive = .{
            .rounds = l.rounds,
            .cost_updates = l.cost_updates,
            .acceptance_updates = l.acceptance_updates,
            .probes = l.probes,
            .rejected_times = l.rejected_times,
            .prediction_samples = l.prediction_samples,
            .prediction_abs_error_microseconds = @intFromFloat(@min(1.0e18, l.prediction_error_ms * 1000)),
            .verify_microseconds = @intFromFloat(@min(1.0e18, l.observed_verify_ms * 1000)),
            .last_rows = l.frame.rows,
            .last_streams = l.frame.count,
            .last_context_tokens = l.frame.context,
            .last_predicted_verify_microseconds = @intFromFloat(@min(1.0e18, @max(0, l.frame.predicted_ms * 1000))),
            .last_verify_microseconds = @intFromFloat(@min(1.0e18, @max(0, (l.frame.verify_ms orelse 0) * 1000))),
        };

        h.live_tokens.clearRetainingCapacity();
        h.live_generated = 0;
        for (h.admitted.items) |job| if (job.started) {
            h.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
            h.live_generated += @intCast(job.stream.emitted().len);
        };
        if (tele) |t| h.tele = t;
        if (parts) |p| h.kv_tele = p;
    }

    fn admitBatch(h: *LaneHost) void {
        // Failed/cancelled attempts may drain only while no decoder is live.
        while (h.admitOne()) {
            if (h.broken()) return;
            if (h.between_err != null or (!h.stepped() and h.core.activeCount() > 0)) break;
        }
    }

    fn stepped(h: *const LaneHost) bool {
        if (h.core.log != null) return false;
        const v = h.core.backend.vtable;
        return v.prefill_begin != null and v.prefill_step != null and v.prefill_end != null;
    }

    /// Physical residents include suspended fills, but never the failed admission itself.
    fn residentCount(h: *const LaneHost, skip: ?*Job) usize {
        var n = h.core.activeCount();
        for (h.admitted.items) |job| {
            if (job != skip and job.fill_began and job.prefilled == null) n += 1;
        }
        return n;
    }

    fn hasFills(h: *const LaneHost) bool {
        for (h.admitted.items) |job| if (job.prefilled == null) return true;
        return false;
    }

    /// Select within one class, aging after eight skips, otherwise foreground then least work.
    fn pickFill(h: *LaneHost, short: bool) ?*Job {
        var best: ?*Job = null;
        for (h.admitted.items) |job| {
            if (job.prefilled != null) continue;
            if ((job.stream.prompt_len -| job.stream.cached <= Share.short_rows) != short) continue;
            const b = best orelse { best = job; continue; };
            const aged = job.fill_wait >= 8;
            const baged = b.fill_wait >= 8;
            const remaining = job.stream.prompt_len -| job.fill_at;
            if (if (aged != baged) aged else if (aged) job.fill_wait > b.fill_wait else if ((job.warm == null) != (b.warm == null)) job.warm == null else remaining < b.stream.prompt_len -| b.fill_at) best = job;
        }
        return best;
    }

    /// One actual native chunk, never an in-prefill callback. Error cleanup is skipped in a poisoned world.
    fn fillAt(h: *LaneHost, job: *Job) void {
        for (h.admitted.items) |other| if (other.prefilled == null) {
            other.fill_wait = if (other == job) 0 else other.fill_wait +| 1;
        };
        h.filling = job;
        defer h.filling = null;
        if (h.share.p < 1) h.slice_fill_at = std.Io.Clock.awake.now(h.io).toNanoseconds();
        const b = h.core.backend;
        const first = !job.fill_began;
        job.fill_began = true;
        const done = blk: {
            if (first) job.stream.backend_owned = true; // begin can fail after acquiring resources
            if (first) b.vtable.prefill_begin.?(b.ptr, &job.stream) catch |err| {
                h.fillFailed(job, err);
                return;
            };
            if (h.broken()) return;
            break :blk b.vtable.prefill_step.?(b.ptr, &job.stream) catch |err| {
                if (h.peer_monitor) |*monitor| monitor.signal(err);
                if (!h.broken()) h.finishSlices() catch {};
                h.fillFailed(job, err);
                return;
            };
        };
        if (h.broken()) return;
        h.finishSlices() catch |err| {
            h.fillFailed(job, err);
            return;
        };
        job.reported(); // restore ownership has transferred before any later admission may evict the entry
        if (!done) return;
        b.vtable.prefill_end.?(b.ptr, &job.stream);
        h.core.openStream(&job.stream) catch |err| {
            h.fillFailed(job, err);
            return;
        };
        if (h.broken()) return;
        h.prefilled(job, job.began);
        _ = h.deliver(job);
    }

    fn fillFailed(h: *LaneHost, job: *Job, err: anyerror) void {
        if (h.peer_monitor) |*monitor| monitor.signal(err);
        if (h.broken()) return;
        if (err == error.ContextFull and h.residentCount(job) > 0) { _ = h.requeue(job); return; }
        // discard releases begin/step resources on cancellation and all nonfatal errors.
        if (err == error.Cancelled) {
            h.core.discard(&job.stream);
            _ = h.cancel(job);
        } else _ = h.drop(job, h.words(err));
    }

    fn runFills(h: *LaneHost) void {
        if (!h.stepped()) return;
        // A tier may have just deferred a queued admission: absence of an active restore is not paid debt.
        if ((!h.hasFills() and !h.pending() and !h.parking()) or h.core.runnableCount() == 0) h.share.debt_ns = 0;
        var burst: usize = 0;
        while (h.pickFill(true)) |job| {
            if (!job.fill_began) {
                if (burst == h.info_.lanes and (h.core.runnableCount() > 0 or h.pickFill(false) != null)) break;
                burst += 1;
            }
            h.fillAt(job);
            if (h.broken()) return;
            h.takeCancels();
            h.admitBatch();
            if (h.broken()) return;
        }
        const job = h.pickFill(false) orelse return;
        if (h.share.debt_ns > 0 and h.core.runnableCount() > 0) return;
        const decoding = h.core.runnableCount() > 0;
        const before = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.slice_decode_ns = 0;
        h.fillAt(job);
        if (h.broken()) return;
        h.share.sliced(@intCast(std.Io.Clock.awake.now(h.io).toNanoseconds() - before), h.slice_decode_ns, decoding);
    }

    fn broken(h: *LaneHost) bool {
        if (h.device_probe) |probe| if (probe.failed(probe.ctx)) return true;
        if (h.peer_monitor) |*monitor| return monitor.tripped.load(.acquire) or monitor.forced.load(.acquire);
        return false;
    }

    fn run(h: *LaneHost) void {
        if (h.enter) |e| e.run(e.ctx);
        while (h.runIteration()) {}
    }

    fn runIteration(h: *LaneHost) bool {
        if (h.device_probe) |probe| if (probe.failed(probe.ctx)) {
            std.Io.sleep(h.io, .fromMilliseconds(1), .awake) catch {};
            return !h.closing;
        };
        if (h.peer_monitor) |*monitor| {
            if (monitor.tripped.load(.acquire) or monitor.forced.load(.acquire)) {
                // No further GPU rounds or admissions in a world already known broken.
                std.Io.sleep(h.io, .fromMilliseconds(1), .awake) catch {};
                return true;
            }
        }
        h.takeCancels();
        if (h.broken()) return true;
        h.takeCacheReset();
        h.admitBatch();
        if (h.broken()) return true;
        h.runFills();
        if (h.broken()) return true;
        if (h.servicing()) h.service(h.tier.service_ns);
        if (h.broken()) return true; // admission can break the world in this iteration
        if (h.between_err) |e| {
            if (h.peer_monitor) |*monitor| monitor.signal(e);
            h.between_err = null;
            h.failAll(@errorName(e));
            return true;
        }
        h.noteLive();
        h.lock();
        if (h.closing) {
            const left = h.queued.items.len + h.admitted.items.len + h.waitingCount();
            h.unlock();
            if (left == 0) {
                h.shutdownTier();
                return h.broken();
            }
            h.closeAll();
            return true;
        }
        if (h.core.activeCount() == 0) {
            if (h.core.learner) |l| l.refresh.idle();
            if (!h.hasFills() and h.cancels.items.len == 0 and ((h.queued.items.len == 0 and h.ready.items.len == 0) or h.admitted.items.len >= h.info_.lanes)) {
                const ms: i64 = if (h.pending()) poll_ns / std.time.ns_per_ms else 100;
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(ms), .clock = .awake } }) catch {};
            }
            h.unlock();
            return true;
        }
        h.unlock();
        if (h.core.learner) |l| l.advanceTo(@intCast(std.Io.Clock.awake.now(h.io).toNanoseconds()));
        const owed = h.share.debt_ns > 0;
        const round_start = if (owed) std.Io.Clock.awake.now(h.io).toNanoseconds() else 0;
        h.core.step() catch |e| {
            if (h.peer_monitor) |*monitor| monitor.signal(e);
            if (h.broken()) return true;
            h.failAll(@errorName(e));
            return true;
        };
        if (owed) h.share.rounded(@intCast(std.Io.Clock.awake.now(h.io).toNanoseconds() - round_start));
        var i: usize = 0;
        while (i < h.admitted.items.len) {
            const job = h.admitted.items[i];
            if (job.prefilled == null) { i += 1; continue; }
            if (!h.deliver(job)) i += 1; // a finished job left `admitted` inside deliver
            if (h.broken()) return true;
        }
        return true;
    }

    /// A failed round ends every stream it held, with the backend's error.
    fn failAll(h: *LaneHost, message: []const u8) void {
        if (h.broken()) return;
        h.lock();
        // Transfer ownership, including capacity, without allocating on the failure path.
        var jobs = h.admitted;
        h.admitted = .empty;
        h.fillGauges();
        h.unlock();
        while (jobs.items.len > 0) {
            const job = jobs.orderedRemove(0);
            if (!h.broken() and job.started) h.core.discard(&job.stream);
            h.finish(job, .failed, message);
            if (h.broken()) {
                // No later discard or callback; return untouched jobs and capacity to host custody.
                h.lock();
                h.admitted = jobs;
                h.fillGauges();
                h.unlock();
                return;
            }
        }
        jobs.deinit(h.gpa);
    }

    fn closeAll(h: *LaneHost) void {
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.ready.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.loading.items) |l| h.cancels.append(h.gpa, l.job.id) catch {};
        h.unlock();
        h.takeCancels();
    }

    // -- parked work (0110) ---------------------------------------------------------------------------------------

    /// Whether admission may park a request: the cache has a tier.
    fn parking(h: *const LaneHost) bool {
        const store = h.cache orelse return false;
        return store.tier != null;
    }

    /// Whether the engine thread services parked work (restores, staged writes, deferred retries).
    fn servicing(h: *const LaneHost) bool {
        return h.parking();
    }

    /// Work in flight the host must poll (engine thread): restores, writes to stage, requests parked or woken.
    fn pending(h: *const LaneHost) bool {
        if (h.loading.items.len > 0 or h.ready.items.len > 0) return true;
        const store = h.cache orelse return false;
        return store.busy();
    }

    /// Only discretionary tier work yields to debt. With no runnable decoder it must progress (including
    /// paused/all-restoring workloads); necessary completion retirement and cancellation never use this gate.
    fn tierWorkAllowed(h: *const LaneHost) bool {
        return h.share.debt_ns <= 0 or h.core.runnableCount() == 0;
    }

    /// Before `job`'s pass (engine thread, no lock): park it on a restore of a longer durable point (once). False:
    /// admit it now.
    fn park(h: *LaneHost, job: *Job) bool {
        const store = h.cache orelse return false;
        const r = job.request;
        if (!job.cache_ready) return false;
        const key = keyOf(r);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        const decoding = h.core.runnableCount() > 0;
        defer if (!h.broken()) h.share.stepped(@intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - now)), decoding);
        const probe = store.probeMedia(key, r.chunks, r.images.len > 0, tierMedia(r));
        if (!job.tried_disk) if (probe.durable) |p| {
            job.tried_disk = true;
            h.lock();
            const room = h.loading.ensureUnusedCapacity(h.gpa, 1);
            h.unlock();
            if (room) |_| {
                if (store.restoreBeginMedia(p, key, tierMedia(r))) |rs| {
                    h.lock();
                    job.room_since = null;
                    h.loading.appendAssumeCapacity(.{ .job = job, .restore = rs, .since = now });
                    h.unlock();
                    h.host_counts.restore_waits += 1;
                    return true;
                } else |err| {
                    if (h.peer_monitor) |*monitor| monitor.signal(err);
                    if (!h.broken() and err == error.ContextFull and h.tier.room_wait_ns > 0 and
                        h.kv_parts == null and h.residentCount(null) > 0)
                    {
                        h.lock();
                        // Removal left a slot, but concurrent submit may have filled it. Failure retains the
                        // ordinary memory fallback; never drop or duplicate detached request ownership.
                        if (h.queued.insert(h.gpa, 0, job)) |_| {
                            if (job.room_since == null) {
                                job.room_since = now;
                                h.host_counts.restore_room_waits += 1;
                            }
                            job.room_retry_at = now + poll_ns;
                            job.tried_disk = false;
                            h.unlock();
                            return true;
                        } else |_| {}
                        h.unlock();
                    }
                    job.room_since = null;
                    note("reading {d} kept tokens from disk did not start ({s}); prefilling", .{ p.at, @errorName(err) });
                }
            } else |_| {}
        };
        job.room_since = null;
        return false;
    }

    /// Caller holds the host lock. Retire each bound once, even while full_wait/decode debt prevents admission.
    fn expireRoomWaits(h: *LaneHost, now: i96) void {
        for (h.queued.items) |job| {
            const since = job.room_since orelse continue;
            if (now - since < h.tier.room_wait_ns) continue;
            job.room_since = null;
            job.tried_disk = true;
            h.host_counts.restore_room_expiries += 1;
        }
    }

    /// A woken job to the ready list (the lock held; released here). Out of memory fails it alone.
    fn toReady(h: *LaneHost, job: *Job) void {
        h.ready.append(h.gpa, job) catch {
            h.unlock();
            return h.finish(job, .failed, "out of memory");
        };
        h.unlock();
    }

    /// Parked work's service (engine thread): restores progress under one elapsed-time budget shared fairly (each gets
    /// an equal share of what is left, a different one first each time); writes are staged with what remains. Restores
    /// that end wake every request waiting for them; deferred restores whose retry came ask again.
    /// Called inside a fill's `between` too, so a family's restore commands there must keep the follower's ordered
    /// stream valid (dsv41 joins Engram read-ahead before `between`, as recipe 0005 requires).
    /// Completion-only maintenance/expiry are exempt from debt: charging them while deferring copies could
    /// perpetually refinance the debt. Discretionary begin/poll/pump work still shares the fill account.
    fn service(h: *LaneHost, budget_ns: u64) void {
        if (h.broken()) return;
        const store = h.cache orelse return;
        const t0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
        // A nested service is already inside the outer long-fill interval. Do not charge twice.
        const decoding = h.core.runnableCount() > 0;
        const charge = !h.stepped() or h.filling == null;
        h.expireLoads(t0);
        if (h.broken()) return;
        store.maintain(); // custody/completions must progress even while new copies yield to decode
        if (h.broken() or !h.tierWorkAllowed()) return;
        const n = store.restoring.items.len;
        if (n > 0) { // at most 64 a service, a different first one each time: every restore is polled in turn
            var buf: [64]*pc.Restore = undefined;
            const m = @min(n, buf.len);
            h.turn = (h.turn + 1) % n;
            for (0..m) |k| buf[k] = store.restoring.items[(h.turn + k) % n];
            for (buf[0..m], 0..) |r, k| {
                if (h.broken()) return;
                const spent: u64 = @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t0));
                if (spent >= budget_ns and k > 0) break;
                const share = (budget_ns -| spent) / (m - k);
                if (h.broken()) return;
                const before = std.Io.Clock.awake.now(h.io).toNanoseconds();
                const st = store.restorePoll(r, share);
                if (h.broken()) return;
                if (st != .pending) h.restoreEnded(r);
                if (charge) h.share.stepped(@intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - before)), decoding);
                // Account each quantum before offering another restore or the write pump. Nested service is
                // charged by the outer fill; stop after one quantum there rather than spend its unbooked debt.
                if (!h.tierWorkAllowed() or (!charge and decoding and h.share.p < 1)) return;
            }
        }
        if (h.broken()) return;
        const spent: u64 = @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t0));
        const idle = h.filling == null and h.core.activeCount() == 0; // nothing decodes: staging takes what is left
        const before = std.Io.Clock.awake.now(h.io).toNanoseconds();
        store.pump(if (idle) budget_ns -| spent else @min(budget_ns -| spent, h.tier.pump_ns));
        if (charge and !h.broken()) h.share.stepped(@intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - before)), decoding);
    }

    /// Requests parked on a restore longer than load_wait_ns are admitted without it (they look up what memory holds);
    /// a restore no request waits for any more stops, once.
    fn expireLoads(h: *LaneHost, now: i96) void {
        if (h.broken()) return;
        const store = h.cache orelse return;
        var any = false;
        var i: usize = 0;
        while (i < h.loading.items.len) {
            if (h.broken()) return;
            const l = h.loading.items[i];
            if (now - l.since < h.tier.load_wait_ns) {
                i += 1;
                continue;
            }
            any = true;
            h.lock();
            _ = h.loading.orderedRemove(i);
            const waited: u64 = @intCast(@max(0, now - l.since));
            h.host_counts.restore_wait_timeouts += 1;
            h.host_counts.restore_wait_ns += waited;
            h.host_counts.restore_wait_max_ns = @max(h.host_counts.restore_wait_max_ns, waited);
            h.toReady(l.job); // (releases the lock)
            const wanted = for (h.loading.items) |o| {
                if (o.restore == l.restore) break true;
            } else false;
            if (h.broken()) return;
            if (!wanted) store.restoreCancel(l.restore);
            if (h.broken()) return;
        }
        if (!any) return;
        h.lock();
        h.wake.signal(h.io);
        h.unlock();
    }

    /// Restore `r` ended (adopted, dropped or failed; `r` is already freed, only its address is compared): its
    /// waiters wake in arrival order, to look up what memory holds now.
    fn restoreEnded(h: *LaneHost, r: *pc.Restore) void {
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        while (true) {
            h.lock();
            const i = for (h.loading.items, 0..) |l, i| {
                if (l.restore == r) break i;
            } else {
                h.wake.signal(h.io);
                return h.unlock();
            };
            const l = h.loading.orderedRemove(i);
            const waited: u64 = @intCast(@max(0, now - l.since));
            h.host_counts.restore_wait_ns += waited;
            h.host_counts.restore_wait_max_ns = @max(h.host_counts.restore_wait_max_ns, waited);
            h.toReady(l.job);
            if (h.broken()) return;
        }
    }

    /// Clean shutdown on the engine thread, nothing left to serve: restores stopped, writes drained within flush_ns.
    fn shutdownTier(h: *LaneHost) void {
        if (h.broken()) return;
        const store = h.cache orelse return;
        if (store.tier == null) return;
        while (store.restoring.items.len > 0) {
            if (h.broken()) return;
            store.restoreCancel(store.restoring.items[store.restoring.items.len - 1]);
        }
        if (h.broken()) return;
        _ = store.flush(h.tier.flush_ns);
        if (h.broken()) return;
        if (h.telemetry()) |t| {
            h.lock();
            h.tele = t;
            h.unlock();
        }
    }

    /// The cache's counters for status (engine thread); null while telemetry is off.
    fn telemetry(h: *LaneHost) ?api.CacheStats {
        if (h.tele == null) return null;
        var t: api.CacheStats = .{};
        if (h.cache) |store| {
            const c = store.counts;
            const s = store.stats;
            t.memory_hits = c.hits;
            t.memory_misses = c.misses;
            t.kept = c.kept;
            t.evicted = c.evicted;
            t.refused = c.refused;
            t.failed = c.failed;
            t.entries = store.entries.items.len;
            t.held_bytes = store.held;
            t.disk_lookups = s.disk_lookups;
            t.disk_hits = s.disk_hits;
            t.disk_hits_committed = s.disk_hits_committed;
            t.restores_begun = s.restores_begun;
            t.restores_shared = s.restores_shared;
            t.restores_failed = s.restores_failed;
            t.restores_cancelled = s.restores_cancelled;
            t.restores_dropped = s.restores_dropped;
            t.restores_refused = s.restores_refused;
            t.restores_active = store.restoring.items.len;
            t.restoring_bytes = store.restoring_bytes;
            for (store.restoring.items) |r| {
                t.restore_blocks_done += r.done;
                t.restore_blocks_total += r.total;
            }
            t.restored_tokens = s.restored_tokens;
            t.writes_enqueued = s.writes_enqueued;
            t.shared_promoted = s.shared_promoted;
            t.flushes = s.flushes;
            t.flush_timeouts = s.flush_timeouts;
            t.flush_unpublished = s.flush_unpublished;
            t.chats_continued = s.chats_continued;
            t.chats_continued_prefill_tokens = s.chats_continued_prefill_tokens;
            t.spill = store.tierCounts();
        }
        const hc = h.host_counts;
        t.restore_room_waits = hc.restore_room_waits;
        t.restore_room_expiries = hc.restore_room_expiries;
        t.restore_waits = hc.restore_waits;
        t.restore_wait_timeouts = hc.restore_wait_timeouts;
        t.restore_wait_ns = hc.restore_wait_ns;
        t.restore_wait_max_ns = hc.restore_wait_max_ns;
        t.cached_tokens = hc.cached_tokens;
        t.computed_tokens = hc.computed_tokens;
        t.parked = h.loading.items.len;
        return t;
    }
};

test "cache reset drops snapshots on engine thread and refuses overlapping work" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snap = struct {
        creator: std.Thread.Id,
        dropped: usize = 0,
        wrong_thread: bool = false,
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 1;
        }
        fn save(ctx: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved {
            return ctx;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
        fn drop(ctx: *anyopaque, _: pc.Saved) void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            s.dropped += 1;
            s.wrong_thread = s.creator == std.Thread.getCurrentId();
        }
        fn event(_: *anyopaque, _: Id, _: *const Event) void {
            unreachable;
        }
    };
    var snap: Snap = .{ .creator = std.Thread.getCurrentId() };
    var store = pc.Store.init(gpa, .{ .ptr = &snap, .vtable = &.{ .bytes = Snap.bytes, .save = Snap.save, .restore = Snap.restore, .drop = Snap.drop } }, .{ .min_prompt = 0 }, 100);
    defer store.deinit();
    try std.testing.expect(store.keep(&.{ 1, 2, 3, 4 }, 3, null, &.{}));
    try std.testing.expect(store.find(&.{ 1, 2, 3, 4 }, &.{}) != null);
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    const e = host.engine();
    try std.testing.expectError(error.Closed, e.resetCache());
    try host.start();
    defer host.stop();
    try e.resetCache();
    try std.testing.expectEqual(@as(usize, 1), snap.dropped);
    try std.testing.expect(!snap.wrong_thread);
    try std.testing.expectEqual(@as(u64, 0), store.held);
    try std.testing.expect(store.find(&.{ 1, 2, 3, 4 }, &.{}) == null);
    try e.resetCache(); // idempotent, no duplicate drop
    try std.testing.expectEqual(@as(usize, 1), snap.dropped);
    host.lock();
    host.cache_reset = .done; // completed reset still owned by its waiter
    host.unlock();
    try std.testing.expectError(error.Busy, e.resetCache());
    const r: Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    try std.testing.expectError(error.Busy, e.submit(1, &r, .{ .ctx = &snap, .event = Snap.event }));
    host.lock();
    host.cache_reset = .idle;
    host.unlock();
    var imprint: api.prompt_imprint.Imprint = undefined;
    store.imprint = &imprint; // clear checks presence before dereferencing
    try std.testing.expectError(error.Unsupported, e.resetCache());
    store.imprint = null;
    try e.resetCache();
}

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("lane host: " ++ fmt, args);
}

test "a lane host serves the core's own tokens, in order, and cancels between rounds" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (box.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 24), box.tokens.items.len);
    var gone: Box = .{};
    defer gone.tokens.deinit(gpa);
    const long: Request = .{ .prompt = &prompt, .max_tokens = 100000 };
    try e.submit(2, &long, .{ .ctx = &gone, .event = Box.event });
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, gone.wait());

    const CancelPrefill = struct {
        engine: Engine,
        id: Id,
        at: usize,

        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == c.at) c.engine.cancel(c.id);
        }
    };
    const chunked_prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var chunked: Box = .{};
    defer chunked.tokens.deinit(gpa);
    var prefill_cancel = CancelPrefill{ .engine = e, .id = 3, .at = 2 };
    target.prefill_chunks = 10;
    target.prefill_count = 0;
    target.prefill_hook = CancelPrefill.call;
    target.prefill_hook_ctx = &prefill_cancel;
    const chunked_request: Request = .{ .prompt = &chunked_prompt, .max_tokens = 1 };
    try e.submit(3, &chunked_request, .{ .ctx = &chunked, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, chunked.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released

    // the lone driver's prompt pass (gpu_round.run starts with Backend.opening), cancelled the same way
    const Lone = struct {
        be: lanes.backend.Backend,

        fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            _ = try l.be.opening(gpa, s);
            return error.NotCancelled;
        }
    };
    var lone: Box = .{};
    defer lone.tokens.deinit(gpa);
    var lone_driver = Lone{ .be = target.backend() };
    host.lone = .{ .ctx = &lone_driver, .run = Lone.run };
    prefill_cancel.id = 4;
    target.prefill_count = 0;
    try e.submit(4, &chunked_request, .{ .ctx = &lone, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, lone.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test {
    _ = @import("lane_host_test.zig");
}

test "prefill counters are available before generated reply completion" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    const req: api.Request = .{ .prompt = &.{ 1, 2, 3, 4, 5, 6 }, .max_tokens = 100 };
    var stream = try lanes.Stream.init(a, .{ .id = "telemetry", .prompt = req.prompt, .max_new = req.max_tokens });
    defer stream.deinit(a);
    stream.cached = 4;
    const SinkStub = struct {
        fn event(_: *anyopaque, _: Id, ev: *const Event) void {
            std.debug.assert(ev.* == .prefilled);
        }
    };
    var job: LaneHost.Job = .{ .host = &host, .id = 1, .request = &req, .stream = stream, .sink = .{ .ctx = &host, .event = SinkStub.event } };
    const now = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    host.prefilled(&job, now - std.time.ns_per_s);
    var status: api.Status = .{};
    host.engine().status(&status, &.{});
    const p = status.prefill_telemetry.?;
    try std.testing.expectEqual(@as(u64, 6), p.input);
    try std.testing.expectEqual(@as(u64, 4), p.cached);
    try std.testing.expectEqual(@as(u64, 2), p.computed);
    try std.testing.expectEqual(@as(u64, 1), p.count);
    try std.testing.expect(p.seconds >= 1);
    try std.testing.expect(!job.stream.finished);
}

test "f native admission yields to decode and preserves foreground FIFO cancellation" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(a, std.testing.io, &core, .{ .lanes = 4 });
    defer host.stop();
    defer host.closeAll();
    const Box = struct {
        prefilled: [4]u64 = @splat(0),
        n: usize = 0,
        fn event(ctx: *anyopaque, id: Id, ev: *const Event) void {
            const box: *@This() = @ptrCast(@alignCast(ctx));
            if (ev.* == .prefilled) {
                box.prefilled[box.n] = id;
                box.n += 1;
            }
        }
    };
    var box: Box = .{};
    const foreground: Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 100 };
    var background = foreground;
    background.background = true;
    const engine = host.engine();
    const sink: Sink = .{ .ctx = &box, .event = Box.event };
    try engine.submit(1, &background, sink);
    try engine.submit(2, &foreground, sink);
    try engine.submit(3, &foreground, sink);
    host.admitBatch();
    try std.testing.expectEqual(@as(usize, 1), core.activeCount());
    try std.testing.expectEqual(@as(usize, 2), host.queued.items.len);
    try std.testing.expectEqual(@as(u64, 2), box.prefilled[0]);
    const steps = core.steps;
    try core.step();
    try std.testing.expect(core.steps > steps);
    host.admitBatch();
    try std.testing.expectEqual(@as(usize, 2), core.activeCount());
    try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
    try std.testing.expectEqual(@as(u64, 3), box.prefilled[1]);
    engine.cancel(1);
    host.takeCancels();
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
    try std.testing.expectEqual(@as(usize, 2), box.n);
}

test "prefill interleave: a decoding stream keeps getting tokens and its cancel lands during another's prompt pass" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.interleave_ns = 5 * std.time.ns_per_ms;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: usize = 0,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn read(b: *@This()) struct { tokens: usize, done: ?Reason } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ .tokens = b.tokens, .done = b.done };
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                if (b.read().done) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    // A: a long decode, admitted and decoding before B arrives
    const prompt_a = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var a: Box = .{};
    const req_a: Request = .{ .prompt = &prompt_a, .max_tokens = 100000 };
    try e.submit(1, &req_a, .{ .ctx = &a, .event = Box.event });
    while (a.read().tokens == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    // B: a six-chunk prompt pass; at each of its chunks note A's tokens, and cancel A at chunk 3
    const Watch = struct {
        engine: Engine,
        a: *Box,
        seen: [6]usize = @splat(0),
        done_at_5: ?Reason = null,
        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            const r = w.a.read();
            if (chunk < w.seen.len) w.seen[chunk] = r.tokens;
            if (chunk == 3) w.engine.cancel(1);
            if (chunk == 5) w.done_at_5 = r.done;
        }
    };
    var watch = Watch{ .engine = e, .a = &a };
    target.prefill_chunks = 6;
    target.prefill_hook = Watch.call;
    target.prefill_hook_ctx = &watch;
    const prompt_b = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var b: Box = .{};
    const req_b: Request = .{ .prompt = &prompt_b, .max_tokens = 1 };
    try e.submit(2, &req_b, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b.wait());
    // A decoded during B's pass (no interleave: seen[0] == seen[2]), and its cancel ended it before B's last chunk
    try std.testing.expect(watch.seen[2] > watch.seen[0]);
    try std.testing.expectEqual(@as(?Reason, Reason.cancelled), watch.done_at_5);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
}

test "broken admissions stop before touching the backend or queued request" {
    var core: lanes.Engine = undefined;
    var host = LaneHost.init(std.testing.allocator, std.testing.io, &core, .{ .lanes = 1, .context_window = 128 });
    defer host.queued.deinit(std.testing.allocator);
    const Probe = struct {
        fn failed(_: *anyopaque) bool {
            return false;
        }
    };
    var byte: u8 = 0;
    host.peer_monitor = .{ .io = std.testing.io, .probe = .{ .ctx = &byte, .failed = Probe.failed }, .policy = .{} };
    host.peer_monitor.?.forced.store(true, .release);
    var request: Request = .{ .prompt = &.{1}, .max_tokens = 2 };
    var job: LaneHost.Job = .{ .host = &host, .id = 1, .request = &request, .sink = undefined };
    try host.queued.append(std.testing.allocator, &job);
    try std.testing.expect(!host.admitOne());
    try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
}

test "fatal admission fences the same iteration before decode telemetry and cleanup" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    const Inject = struct {
        fn prefill(ctx: *anyopaque, stream: *lanes.Stream) anyerror!void {
            if (stream.context.items[0] == 42) return error.ShardDiverged;
            var fake: *lanes.fake.Fake = @ptrCast(@alignCast(ctx));
            return fake.backend().vtable.prefill(ctx, stream);
        }
        fn failed(_: *anyopaque) bool {
            return false;
        }
        fn event(_: *anyopaque, _: Id, _: *const Event) void {}
    };
    var vt = target.backend().vtable.*;
    vt.prefill = Inject.prefill;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3, .context_window = 128 });
    defer {
        host.peer_monitor = null; // CPU-only teardown after assertions, never the production fatal path
        for (host.admitted.items) |job| {
            core.discard(&job.stream);
            host.finish(job, .failed, "test teardown");
        }
        for (host.queued.items) |job| host.finish(job, .failed, "test teardown");
        host.admitted.deinit(gpa);
        host.queued.deinit(gpa);
        host.decoded.deinit(gpa);
        host.live_tokens.deinit(gpa);
    }
    var byte: u8 = 0;
    const sink: api.Sink = .{ .ctx = &byte, .event = Inject.event };
    var a: Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 100 };
    var b: Request = .{ .prompt = &.{42}, .max_tokens = 100 };
    var c: Request = .{ .prompt = &.{3}, .max_tokens = 100 };
    try host.engine().submit(1, &a, sink);
    try std.testing.expect(host.admitOne());
    host.peer_monitor = .{ .io = std.testing.io, .probe = .{ .ctx = &byte, .failed = Inject.failed }, .policy = .{} };
    try host.engine().submit(2, &b, sink);
    try host.engine().submit(3, &c, sink);
    const rounds = target.rounds;
    try std.testing.expect(host.runIteration());
    try std.testing.expect(host.broken());
    try std.testing.expectEqual(rounds, target.rounds);
    try std.testing.expectEqual(@as(usize, 2), host.admitted.items.len); // no drop/discard of failed B
    try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
    try std.testing.expect(host.between_err == null);
}

test {
    _ = @import("lane_host_test.zig");
    _ = @import("lane_host_tier_test.zig");
}


fn lonePumpDue(counter: *u32, active: bool) bool {
    if (!active) return false;
    counter.* +|= 1;
    if (counter.* < LaneHost.lone_pump_commits) return false;
    counter.* = 0;
    return true;
}

test "spill lone pump counter stays unchanged off and cannot overflow on" {
    var count: u32 = std.math.maxInt(u32);
    try std.testing.expect(!lonePumpDue(&count, false));
    try std.testing.expectEqual(std.math.maxInt(u32), count);
    try std.testing.expect(lonePumpDue(&count, true));
    try std.testing.expectEqual(@as(u32, 0), count);
}

test "cache keep callback retains decoded continuation tokens beyond request length" {
    const gpa = std.testing.allocator;
    const Family = struct {
        fn bytes(_: *anyopaque, _: u32) u64 { return 1; }
        fn save(ctx: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved { return ctx; }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
        fn drop(_: *anyopaque, _: pc.Saved) void {}
    };
    var state: u8 = 0;
    var store = pc.Store.init(gpa, .{ .ptr = &state, .vtable = &.{ .bytes = Family.bytes, .save = Family.save, .restore = Family.restore, .drop = Family.drop } }, .{ .min_prompt = 0, .exact_end = true }, 100);
    defer store.deinit();
    var host: LaneHost = undefined;
    host.gpa = gpa;
    host.cache = &store;
    const request: Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 4 };
    var stream: lanes.Stream = undefined;
    stream.context = .empty;
    defer stream.context.deinit(gpa);
    try stream.context.appendSlice(gpa, &.{ 1, 2, 3, 4 });
    var job: LaneHost.Job = .{ .host = &host, .request = &request, .id = 0, .sink = undefined };
    LaneHost.Job.kept(&job, &stream, 4);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expectEqualSlices(u32, stream.context.items, store.entries.items[0].tokens);
}

test "fatal inside tier pump fences same iteration and direct shutdown callbacks" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    defer host.live_tokens.deinit(gpa);
    const Inject = struct {
        host: *LaneHost,
        pumps: usize = 0,
        flushes: usize = 0,
        fn failed(_: *anyopaque) bool { return false; }
        fn bytes(_: *anyopaque, _: u32) u64 { return 1; }
        fn save(ctx: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved { return ctx; }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
        fn drop(_: *anyopaque, _: pc.Saved) void {}
        fn pump(ptr: *anyopaque, _: u64) void {
            const x: *@This() = @ptrCast(@alignCast(ptr));
            x.pumps += 1;
            x.host.peer_monitor.?.signal(error.ShardDiverged);
        }
        fn flush(ptr: *anyopaque, _: u64) u64 {
            const x: *@This() = @ptrCast(@alignCast(ptr));
            x.flushes += 1;
            return 0;
        }
    };
    var x: Inject = .{ .host = &host };
    host.peer_monitor = .{ .io = std.testing.io, .probe = .{ .ctx = &x, .failed = Inject.failed }, .policy = .{} };
    var store = pc.Store.init(gpa, .{ .ptr = &x, .vtable = &.{ .bytes = Inject.bytes, .save = Inject.save, .restore = Inject.restore, .drop = Inject.drop } }, .{}, 100);
    defer store.deinit();
    store.tier = .{ .ptr = &x, .vtable = &.{ .lookup = undefined, .begin = undefined, .poll = undefined, .cancel = undefined, .keep = undefined, .forget = undefined, .pump = Inject.pump, .flush = Inject.flush, .counts = undefined } };
    host.cache = &store;
    try std.testing.expect(host.runIteration());
    try std.testing.expect(host.broken());
    try std.testing.expectEqual(@as(usize, 1), x.pumps);
    host.service(100);
    host.shutdownTier();
    try std.testing.expectEqual(@as(usize, 1), x.pumps);
    try std.testing.expectEqual(@as(usize, 0), x.flushes);
    try std.testing.expectEqual(@as(u64, 0), host.live_generated);
}

// CPU-only callback emergence rig: assertions precede ordinary test teardown.
const FatalCallbacks = struct {
    host: *LaneHost,
    calls: usize = 0,
    events: usize = 0,
    fn failed(_: *anyopaque) bool { return false; }
    fn bytes(_: *anyopaque, _: u32) u64 { return 1; }
    fn save(ctx: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved { return ctx; }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
    fn drop(_: *anyopaque, _: pc.Saved) void {}
    fn trip(ptr: *anyopaque) void {
        const x: *@This() = @ptrCast(@alignCast(ptr));
        x.calls += 1;
        x.host.peer_monitor.?.signal(error.ShardDiverged);
    }
    fn poll(ptr: *anyopaque, _: u64, _: u64) pc.Poll { trip(ptr); return .{ .pending = .{} }; }
    fn cancel(ptr: *anyopaque, _: u64) void { trip(ptr); }
    fn pump(ptr: *anyopaque, _: u64) void { trip(ptr); }
    fn flush(ptr: *anyopaque, _: u64) u64 { trip(ptr); return 0; }
    fn event(ptr: *anyopaque, _: Id, _: *const Event) void {
        const x: *@This() = @ptrCast(@alignCast(ptr));
        x.events += 1;
    }
    fn addRestore(store: *pc.Store) !*pc.Restore {
        const r = try store.gpa.create(pc.Restore);
        r.* = .{ .point = .{ .id = 1, .at = 1, .tokens = 1 }, .job = 1, .key = try store.gpa.dupe(u32, &.{1}) };
        try store.restoring.append(store.gpa, r);
        return r;
    }
    fn tier(x: *@This()) pc.Tier {
        return .{ .ptr = x, .vtable = &.{ .lookup = undefined, .begin = undefined, .poll = poll, .cancel = cancel, .keep = undefined, .forget = undefined, .pump = pump, .flush = flush, .counts = undefined } };
    }
};

test "fatal restorePoll and pump between prefill chunks stop enclosing admission" {
    for ([_]bool{ false, true }) |polling| {
        const gpa = std.testing.allocator;
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 6 };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        var x: FatalCallbacks = .{ .host = &host };
        var store = pc.Store.init(gpa, .{ .ptr = &x, .vtable = &.{ .bytes = FatalCallbacks.bytes, .save = FatalCallbacks.save, .restore = FatalCallbacks.restore, .drop = FatalCallbacks.drop } }, .{}, 100);
        defer { store.tier = null; store.deinit(); }
        defer {
            host.peer_monitor = null;
            host.cache = null;
            for (host.admitted.items) |job| { core.discard(&job.stream); host.finish(job, .failed, "test teardown"); }
            host.admitted.deinit(gpa);
            host.queued.deinit(gpa);
            host.decoded.deinit(gpa);
            host.live_tokens.deinit(gpa);
        }
        var req: Request = .{ .prompt = &.{ 1, 2, 3, 4, 5, 6 }, .max_tokens = 10 };
        try host.engine().submit(1, &req, .{ .ctx = &x, .event = FatalCallbacks.event });
        // Avoid admission lookup: this fixture targets the installed between callback.
        host.cache = &store;
        store.tier = x.tier();
        if (polling) _ = try FatalCallbacks.addRestore(&store);
        host.peer_monitor = .{ .io = std.testing.io, .probe = .{ .ctx = &x, .failed = FatalCallbacks.failed }, .policy = .{} };
        // Empty lookup keeps the test independent of durable namespace mechanics.
        var vt = store.tier.?.vtable.*;
        vt.busy = struct { fn busy(_: *anyopaque) bool { return true; } }.busy;
        vt.lookup = struct { fn lookup(_: *anyopaque, _: *const pc.Want) ?pc.Point { return null; } }.lookup;
        store.tier.?.vtable = &vt;
        try std.testing.expect(!host.admitOne());
        try std.testing.expect(host.broken());
        try std.testing.expectEqual(@as(usize, 1), x.calls);
        try std.testing.expect(target.prefill_count < 6);
        try std.testing.expectEqual(@as(usize, 1), host.admitted.items.len);
        try std.testing.expect(host.admitted.items[0].prefilled == null);
        try std.testing.expectEqual(@as(usize, 0), target.rounds);
    }
}

test "fatal final restoreCancel fences cancellation and enclosing between decode" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var x: FatalCallbacks = .{ .host = &host };
    var store = pc.Store.init(gpa, .{ .ptr = &x, .vtable = &.{ .bytes = FatalCallbacks.bytes, .save = FatalCallbacks.save, .restore = FatalCallbacks.restore, .drop = FatalCallbacks.drop } }, .{}, 100);
    defer { store.tier = null; store.deinit(); }
    var req: Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 100 };
    try host.engine().submit(1, &req, .{ .ctx = &x, .event = FatalCallbacks.event });
    try std.testing.expect(host.admitOne());
    const admitted = host.admitted.items[0];
    var parked: LaneHost.Job = .{ .host = &host, .id = 2, .request = &req, .sink = .{ .ctx = &x, .event = FatalCallbacks.event } };
    defer {
        host.peer_monitor = null; host.cache = null;
        core.discard(&admitted.stream); host.finish(admitted, .failed, "test teardown");
        host.admitted.deinit(gpa); host.queued.deinit(gpa); host.loading.deinit(gpa); host.cancels.deinit(gpa); host.decoded.deinit(gpa); host.live_tokens.deinit(gpa);
    }
    store.tier = x.tier(); host.cache = &store;
    try host.loading.append(gpa, .{ .job = &parked, .restore = try FatalCallbacks.addRestore(&store), .since = 0 });
    try host.cancels.appendSlice(gpa, &.{ 2, 1 });
    host.peer_monitor = .{ .io = std.testing.io, .probe = .{ .ctx = &x, .failed = FatalCallbacks.failed }, .policy = .{} };
    host.interleave_ns = std.time.ns_per_s;
    const events = x.events;
    const rounds = target.rounds;
    try std.testing.expectError(error.PeerDown, LaneHost.between(&host));
    try std.testing.expectEqual(@as(usize, 1), x.calls);
    try std.testing.expectEqual(events, x.events);
    try std.testing.expectEqual(rounds, target.rounds);
    try std.testing.expectEqual(@as(usize, 1), host.admitted.items.len);
    try std.testing.expect(!admitted.stream.finished);
    try std.testing.expect(host.cancel_held == &parked);
    try std.testing.expectEqualSlices(Id, &.{1}, host.cancels.items);
}

test "fatal flush emergence stops shutdown before telemetry callbacks" {
    var core: lanes.Engine = undefined;
    var host = LaneHost.init(std.testing.allocator, std.testing.io, &core, .{ .lanes = 1 });
    var x: FatalCallbacks = .{ .host = &host };
    var store = pc.Store.init(std.testing.allocator, .{ .ptr = &x, .vtable = &.{ .bytes = FatalCallbacks.bytes, .save = FatalCallbacks.save, .restore = FatalCallbacks.restore, .drop = FatalCallbacks.drop } }, .{}, 100);
    defer { store.tier = null; store.deinit(); }
    store.tier = x.tier(); host.cache = &store;
    host.tele = .{}; // counts is undefined: touching post-flush telemetry fails this test.
    host.peer_monitor = .{ .io = std.testing.io, .probe = .{ .ctx = &x, .failed = FatalCallbacks.failed }, .policy = .{} };
    host.shutdownTier();
    try std.testing.expect(host.broken());
    try std.testing.expectEqual(@as(usize, 1), x.calls);
    host.shutdownTier();
    try std.testing.expectEqual(@as(usize, 1), x.calls);
}

test "fatal direct lone committed callback fences successful handoff and finish" {
    for ([_]bool{ false, true }) |handed| {
        const gpa = std.testing.allocator;
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
        var x: FatalCallbacks = .{ .host = &host };
        const Driver = struct {
            x: *FatalCallbacks,
            handed: bool,
            fn run(ptr: *anyopaque, _: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
                const d: *@This() = @ptrCast(@alignCast(ptr));
                FatalCallbacks.trip(d.x);
                hooks.committed(hooks.ctx); // direct callback after poison; successful return still needs fencing
                return d.handed;
            }
        };
        var driver: Driver = .{ .x = &x, .handed = handed };
        host.lone = .{ .ctx = &driver, .run = Driver.run };
        host.peer_monitor = .{ .io = std.testing.io, .probe = .{ .ctx = &x, .failed = FatalCallbacks.failed }, .policy = .{} };
        var req: Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 10 };
        defer {
            host.peer_monitor = null;
            for (host.admitted.items) |job| host.finish(job, .failed, "test teardown");
            host.admitted.deinit(gpa); host.queued.deinit(gpa); host.live_tokens.deinit(gpa);
        }
        try host.engine().submit(1, &req, .{ .ctx = &x, .event = FatalCallbacks.event });
        try std.testing.expect(!host.admitOne());
        try std.testing.expectEqual(@as(usize, 0), x.events);
        try std.testing.expectEqual(@as(usize, 1), host.admitted.items.len);
        try std.testing.expectEqual(@as(usize, 0), core.activeCount());
        try std.testing.expectEqual(@as(usize, 0), target.rounds);
        try std.testing.expect(host.admitted.items[0].prefilled == null);
    }
}

test "device health fences readiness and admission without optional peer probing" {
    const Probe = struct { fn failed(_: *anyopaque) bool { return true; } };
    var byte: u8 = 0;
    var h: LaneHost = undefined;
    h.peer_monitor = null;
    h.device_probe = .{ .ctx = &byte, .failed = Probe.failed };
    try std.testing.expect(h.broken());
}

// The fake's history-dependent oracle detects incorrect opening, prompt mixing and round ordering.
test "stepped prefill long prompts interleave exact tokens and bound short bursts" {
    const gpa = std.testing.allocator;
    const Log = struct {
        tokens: [3]std.ArrayList(u32) = @splat(.empty),
        burst: usize = 0,
        most: usize = 0,
        done: usize = 0,
        decoder_done: bool = false,
        fn event(ctx: *anyopaque, id: Id, ev: *const Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            switch (ev.*) {
                .tokens => |ts| {
                    if (id <= 3) o.tokens[id - 1].appendSlice(std.testing.allocator, ts) catch unreachable;
                    if (id == 1) o.burst = 0;
                },
                .prefilled => if (!o.decoder_done and id != 2) { o.burst += 1; o.most = @max(o.most, o.burst); },
                .finished => { o.done += 1; if (id == 1) o.decoder_done = true; },
            }
        }
    };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
    host.share = .{ .p = 0.5 };
    defer host.stop();
    var log: Log = .{};
    defer for (&log.tokens) |*t| t.deinit(gpa);
    const short = [_]u32{ 2, 7, 1, 8 };
    const long: [1024]u32 = @splat(5);
    const requests = [_]Request{
        .{ .prompt = &short, .max_tokens = 120 },
        .{ .prompt = &long, .max_tokens = 7 },
        .{ .prompt = &short, .max_tokens = 1 },
    };
    const e = host.engine();
    for (&requests, 1..) |*req, id| try e.submit(id, req, .{ .ctx = &log, .event = Log.event });
    for (4..24) |id| try e.submit(id, &requests[2], .{ .ctx = &log, .event = Log.event });
    var iterations: usize = 0;
    while (log.done < 23 and iterations < 10000) : (iterations += 1) _ = host.runIteration();
    try std.testing.expectEqual(@as(usize, 23), log.done);
    try std.testing.expect(log.most <= 3);
    for (&requests, &log.tokens) |req, *tokens| {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, req.prompt);
        try std.testing.expectEqual(req.max_tokens, tokens.items.len);
        for (tokens.items) |token| {
            try std.testing.expectEqual(lanes.fake.next(history.items, req.sampling, history.items.len), token);
            try history.append(gpa, token);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test "stepped prefill cancellation and begin errors release all state" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Quiet = struct { fn event(_: *anyopaque, _: Id, _: *const Event) void {} };
    var dummy: u8 = 0;
    const rows: [1024]u32 = @splat(5);
    const req: Request = .{ .prompt = &rows, .max_tokens = 8 };
    const e = host.engine();
    try e.submit(1, &req, .{ .ctx = &dummy, .event = Quiet.event });
    _ = host.runIteration();
    try std.testing.expectEqual(@as(usize, 1), target.lanes.count());
    e.cancel(1);
    host.takeCancels();
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expectEqual(@as(usize, 0), host.admitted.items.len);
    target.refuse_sampled = true;
    const bad: Request = .{ .prompt = &rows, .max_tokens = 8, .sampling = .{ .seed = 1, .temperature = 0.5 } };
    try e.submit(2, &bad, .{ .ctx = &dummy, .event = Quiet.event });
    host.admitBatch();
    host.runFills();
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expectEqual(@as(usize, 0), host.admitted.items.len);
    host.share.debt_ns = 9000;
    host.runFills();
    try std.testing.expectEqual(@as(i64, 0), host.share.debt_ns);
    host.stop();
}

test "stepped prefill paused decoders and absent fills erase old debt" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    var decoder = try lanes.Stream.init(gpa, .{ .id = "paused", .prompt = &.{ 2, 7, 1, 8 }, .max_new = 8 });
    defer decoder.deinit(gpa);
    try core.addStream(&decoder);
    defer core.discard(&decoder);
    decoder.paused = true;
    const Quiet = struct { fn event(_: *anyopaque, _: Id, _: *const Event) void {} };
    var dummy: u8 = 0;
    const rows: [1024]u32 = @splat(5);
    const req: Request = .{ .prompt = &rows, .max_tokens = 1 };
    try host.engine().submit(1, &req, .{ .ctx = &dummy, .event = Quiet.event });
    host.admitBatch();
    host.share = .{ .p = 0.01, .debt_ns = 9000 };
    host.runFills();
    try std.testing.expectEqual(@as(i64, 0), host.share.debt_ns);
    try std.testing.expect(host.admitted.items[0].fill_began);
    host.engine().cancel(1);
    host.takeCancels();
    decoder.paused = false;
    host.share.debt_ns = 9000;
    host.runFills();
    try std.testing.expectEqual(@as(i64, 0), host.share.debt_ns);
}


test "stepped prefill gauges cover simultaneous admitted lifetime and cancellation" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Quiet = struct { fn event(_: *anyopaque, _: Id, _: *const Event) void {} };
    var dummy: u8 = 0;
    const rows: [1024]u32 = @splat(5);
    const req: Request = .{ .prompt = &rows, .max_tokens = 8 };
    const e = host.engine();
    try e.submit(1, &req, .{ .ctx = &dummy, .event = Quiet.event });
    try e.submit(2, &req, .{ .ctx = &dummy, .event = Quiet.event });
    host.admitBatch();
    try std.testing.expectEqual(@as(u32, 2), host.prefill_telemetry.active);
    try std.testing.expectEqual(@as(u64, 2048), host.prefill_telemetry.pending);
    const first = host.admitted.items[0];
    first.fill_at = 256;
    host.fillGauges();
    try std.testing.expectEqual(@as(u64, 1792), host.prefill_telemetry.pending);
    e.cancel(1);
    host.takeCancels();
    try std.testing.expectEqual(@as(u32, 1), host.prefill_telemetry.active);
    try std.testing.expectEqual(@as(u64, 1024), host.prefill_telemetry.pending);
    e.cancel(2);
    host.takeCancels();
    try std.testing.expectEqual(@as(u32, 0), host.prefill_telemetry.active);
    try std.testing.expectEqual(@as(u64, 0), host.prefill_telemetry.pending);
    host.stop();
}

test "custody stage stepped warm foreground arrival cancels without generation" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Seen = struct {
        host: *LaneHost,
        warm_cancelled: usize = 0,
        foreground_cancelled: usize = 0,
        fn event(ptr: *anyopaque, id: Id, ev: *const Event) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            switch (ev.*) {
                .finished => |f| {
                    self.host.lock();
                    defer self.host.unlock();
                    std.debug.assert(f.reason == .cancelled);
                    if (id == 901) self.warm_cancelled += 1 else self.foreground_cancelled += 1;
                },
                else => {},
            }
        }
    };
    var seen: Seen = .{ .host = &host };
    const long: [4096]u32 = @splat(5);
    const rows = try gpa.dupe(u32, &long);
    var rows_owned = true;
    errdefer if (rows_owned) gpa.free(rows);
    const warm = try gpa.create(LaneHost.Warm);
    var warm_owned = true;
    errdefer if (warm_owned) gpa.destroy(warm);
    const namespace = pc.namespaceFor("warm-cancel");
    warm.* = .{ .tokens = rows, .from = 0, .at = 2048, .request = .{ .prompt = rows, .max_tokens = 0, .drafts = false, .background = true, .cache_namespace = namespace } };
    const job = try gpa.create(LaneHost.Job);
    var job_owned = true;
    errdefer if (job_owned) gpa.destroy(job);
    job.* = .{ .host = &host, .id = 901, .request = &warm.request, .sink = .{ .ctx = &seen, .event = Seen.event }, .warm = warm };
    const foreground: Request = .{ .prompt = &.{ 8, 9 }, .max_tokens = 2 };
    try host.queued.append(gpa, job);
    rows_owned = false;
    warm_owned = false;
    job_owned = false;
    defer {
        failing.fail_index = std.math.maxInt(usize);
        failing.resize_fail_index = std.math.maxInt(usize);
        // No worker was started. Drain ownership directly, without allocating cancel IDs.
        while (host.queued.items.len > 0) host.finish(host.queued.orderedRemove(0), .cancelled, "");
        while (host.admitted.items.len > 0) {
            const owned = host.admitted.orderedRemove(0);
            if (owned.started and !owned.stream.finished) core.discard(&owned.stream);
            host.finish(owned, .cancelled, "");
        }
        host.stop();
    }
    _ = host.runIteration();
    try std.testing.expectEqual(@as(usize, 1), target.warm_steps);
    try std.testing.expectEqual(@as(usize, 1), host.admitted.items.len);
    try std.testing.expect(host.admitted.items[0] == job);
    try std.testing.expect(job.fill_began and job.prefilled == null);
    try std.testing.expect(job.fill_at > 0 and job.fill_at < rows.len);
    try std.testing.expectEqual(@as(usize, 1), target.lanes.count());
    try std.testing.expectEqual(namespace, job.request.cache_namespace.?);
    try std.testing.expectEqual(@as(usize, 0), job.request.images.len);
    try std.testing.expectEqual(@as(usize, 0), job.request.cache_key.len);
    try std.testing.expectEqual(@as(usize, 0), job.request.cache_media.len);
    const steps_before = target.warm_steps;
    try host.engine().submit(902, &foreground, .{ .ctx = &seen, .event = Seen.event });
    // Arrival must see the still-owned suspended fill, not an already finished warmer.
    try std.testing.expectEqual(@as(usize, 1), host.admitted.items.len);
    try std.testing.expect(std.mem.indexOfScalar(Id, host.cancels.items, 901) != null);
    const allocations = failing.alloc_index;
    const resizes = failing.resize_index;
    failing.fail_index = allocations;
    failing.resize_fail_index = resizes;
    host.takeCancels();
    try std.testing.expectEqual(allocations, failing.alloc_index);
    try std.testing.expectEqual(resizes, failing.resize_index);
    try std.testing.expect(!failing.has_induced_failure);
    try std.testing.expectEqual(@as(usize, 1), seen.warm_cancelled);
    failing.fail_index = std.math.maxInt(usize);
    failing.resize_fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(usize, 0), host.admitted.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    host.runFills(); // removed work must not receive the remaining three chunks
    try std.testing.expectEqual(steps_before, target.warm_steps);
    try std.testing.expectEqual(@as(u32, 0), host.prefill_telemetry.active);
    try std.testing.expectEqual(@as(u64, 0), host.prefill_telemetry.pending);
    try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
    try std.testing.expect(host.queued.items[0].request == &foreground);
    for ([_]usize{ target.warm_first, target.warm_draft, target.warm_verify, target.warm_sample, target.warm_commit }) |calls| try std.testing.expectEqual(@as(usize, 0), calls);
    try std.testing.expectEqual(@as(u32, 0), target.warm_budget);
    host.engine().cancel(902);
    host.takeCancels();
    try std.testing.expectEqual(@as(usize, 1), seen.warm_cancelled);
    try std.testing.expectEqual(@as(usize, 1), seen.foreground_cancelled);
}

// UNRUN: source-only regressions; no compiler, fixture process or runtime was executed.
test "ContextFull waits for a suspended fill without a live decoder" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Quiet = struct { fn event(_: *anyopaque, _: Id, _: *const Event) void {} };
    var dummy: u8 = 0;
    const rows: [1024]u32 = @splat(5);
    const req: Request = .{ .prompt = &rows, .max_tokens = 1 };
    try host.engine().submit(1, &req, .{ .ctx = &dummy, .event = Quiet.event });
    try host.engine().submit(2, &req, .{ .ctx = &dummy, .event = Quiet.event });
    host.admitBatch();
    const first = host.admitted.items[0];
    const blocked = host.admitted.items[1];
    host.fillAt(first);
    try std.testing.expectEqual(@as(usize, 0), core.activeCount());
    try std.testing.expectEqual(@as(usize, 1), host.residentCount(blocked));
    // Model a begin refusal before allocation, while first retains the physical reservation.
    blocked.fill_began = true;
    host.fillFailed(blocked, error.ContextFull);
    try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
    try std.testing.expectEqual(@as(usize, 1), host.admitted.items.len);
    try std.testing.expectEqual(@as(usize, 1), host.full_wait.?);
    try std.testing.expect(!host.admitOne());
    host.engine().cancel(1);
    host.takeCancels();
    try std.testing.expect(host.admitOne());
    try std.testing.expect(host.admitted.items[0] == blocked);
    // No other residents: a genuinely impossible placement must fail, not requeue forever.
    host.fillFailed(blocked, error.ContextFull);
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
    try std.testing.expectEqual(@as(usize, 0), host.admitted.items.len);
}

// UNRUN: exercises the real bulk-failure path with a suspended backend allocation.
test "failAll clears suspended prefill gauges" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Quiet = struct { fn event(_: *anyopaque, _: Id, _: *const Event) void {} };
    var dummy: u8 = 0;
    const rows: [1024]u32 = @splat(5);
    const req: Request = .{ .prompt = &rows, .max_tokens = 1 };
    try host.engine().submit(1, &req, .{ .ctx = &dummy, .event = Quiet.event });
    host.admitBatch();
    host.fillAt(host.admitted.items[0]);
    try std.testing.expect(host.prefill_telemetry.active > 0);
    try std.testing.expect(host.prefill_telemetry.pending > 0);
    host.failAll("recoverable round failure");
    try std.testing.expectEqual(@as(usize, 0), host.admitted.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expectEqual(@as(u32, 0), host.prefill_telemetry.active);
    try std.testing.expectEqual(@as(u64, 0), host.prefill_telemetry.pending);
    try std.testing.expectEqual(@as(u64, 0), host.prefill_telemetry.progress);
}

// UNRUN: deterministic callback-order regressions; debt is injected, never inferred from clock speed.
const AggregateDebtTier = struct {
    host: *LaneHost,
    begins: usize = 0,
    cancels: usize = 0,
    polls: usize = 0,
    pumps: usize = 0,
    maintenance: usize = 0,
    order: [8]u64 = @splat(0),
    writes_pending: bool = true,
    debit_on_poll: bool = false,
    complete: bool = false,
    trip_on_maintain: bool = false,
    tripped: bool = false,

    fn of(p: *anyopaque) *@This() { return @ptrCast(@alignCast(p)); }
    fn bytes(_: *anyopaque, _: u32) u64 { return 1; }
    fn save(p: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved { return p; }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
    fn drop(_: *anyopaque, _: pc.Saved) void {}
    fn lookup(_: *anyopaque, _: *const pc.Want) ?pc.Point { return .{ .id = 9, .at = 1, .tokens = 1 }; }
    fn begin(p: *anyopaque, point: *const pc.Point) !u64 { of(p).begins += 1; return point.id; }
    fn poll(p: *anyopaque, id: u64, _: u64) pc.Poll {
        const x = of(p);
        x.order[x.polls] = id;
        x.polls += 1;
        if (x.debit_on_poll) x.host.share.debt_ns = 1000;
        return if (x.complete) .{ .ready = .{ .saved = p, .bytes = 1 } } else .{ .pending = .{} };
    }
    fn cancel(p: *anyopaque, _: u64) void { of(p).cancels += 1; }
    fn keep(_: *anyopaque, _: *const pc.Kept) ?u64 { return null; }
    fn forget(_: *anyopaque, _: u64) void {}
    fn pump(p: *anyopaque, _: u64) void { of(p).pumps += 1; }
    fn maintain(p: *anyopaque) void {
        const x = of(p);
        x.maintenance += 1;
        if (x.trip_on_maintain) x.tripped = true;
    }
    fn failed(p: *anyopaque) bool { return of(p).tripped; }
    fn busy(p: *anyopaque) bool { return of(p).writes_pending; }
    fn flush(_: *anyopaque, _: u64) u64 { return 0; }
    fn counts(_: *anyopaque, _: []pc.TierCounts) usize { return 0; }
    fn event(_: *anyopaque, _: Id, _: *const Event) void {}
    fn cache(x: *@This()) pc.Store {
        var s = pc.Store.init(std.testing.allocator, .{ .ptr = x, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .{ .min_prompt = 0 }, 100);
        s.tier = .{ .ptr = x, .vtable = &.{ .lookup = lookup, .begin = begin, .poll = poll, .cancel = cancel, .keep = keep, .forget = forget, .pump = pump, .maintain = maintain, .busy = busy, .flush = flush, .counts = counts } };
        return s;
    }
    fn add(s: *pc.Store, id: u64) !void {
        const r = try s.gpa.create(pc.Restore);
        errdefer s.gpa.destroy(r);
        const key = try s.gpa.dupe(u32, &.{@intCast(id)});
        errdefer s.gpa.free(key);
        r.* = .{ .point = .{ .id = id, .at = 1, .tokens = 1 }, .job = id, .key = key };
        try s.restoring.append(s.gpa, r);
    }
};

test "aggregate tier debt offers decode before restore admission poll and pump" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    var x: AggregateDebtTier = .{ .host = &host, .writes_pending = false };
    var store = x.cache();
    defer store.deinit();
    host.cache = &store;
    var decoder = try lanes.Stream.init(gpa, .{ .id = "decode", .prompt = &.{ 2, 7, 1, 8 }, .max_new = 8 });
    defer decoder.deinit(gpa);
    try core.addStream(&decoder);
    defer core.discard(&decoder);
    const req: Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 1 };
    try host.engine().submit(91, &req, .{ .ctx = &x, .event = AggregateDebtTier.event });
    defer {
        host.engine().cancel(91);
        host.takeCancels();
    }
    // No active restore: runFills must not erase the debt of this deferred admission.
    host.share = .{ .p = 0.5, .debt_ns = std.math.maxInt(i64) / 4 };
    const before = decoder.context.items.len;
    try std.testing.expect(host.runIteration());
    try std.testing.expect(decoder.context.items.len > before);
    try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
    try std.testing.expect(!host.queued.items[0].tried_disk);
    try std.testing.expectEqual(@as(usize, 0), x.begins);
    try std.testing.expectEqual(@as(usize, 0), x.polls);
    try std.testing.expectEqual(@as(usize, 0), x.pumps);
    try std.testing.expectEqual(@as(usize, 1), x.maintenance);
    // Pausing the only decoder removes the gate even with old debt, without losing the disk attempt.
    decoder.paused = true;
    try std.testing.expect(host.admitOne());
    try std.testing.expectEqual(@as(usize, 1), x.begins);
    try std.testing.expectEqual(@as(usize, 1), host.loading.items.len);
    decoder.paused = false;
    host.share.debt_ns = 1000;
    host.tier.load_wait_ns = 0; // expiry is safety work, not another discretionary copy grant
    host.service(0);
    try std.testing.expectEqual(@as(usize, 0), host.loading.items.len);
    try std.testing.expectEqual(@as(usize, 1), host.ready.items.len);
    try std.testing.expectEqual(@as(usize, 0), store.restoring.items.len);
    try std.testing.expectEqual(@as(usize, 1), x.cancels);
    try std.testing.expectEqual(@as(usize, 0), x.polls);
    try std.testing.expectEqual(@as(usize, 0), x.pumps);
}

test "aggregate tier poll debits before another restore or pump quantum" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    var x: AggregateDebtTier = .{ .host = &host, .debit_on_poll = true };
    var store = x.cache();
    defer store.deinit();
    host.cache = &store;
    try AggregateDebtTier.add(&store, 1);
    try AggregateDebtTier.add(&store, 2);
    var decoder = try lanes.Stream.init(gpa, .{ .id = "decode", .prompt = &.{ 2, 7, 1, 8 }, .max_new = 8 });
    defer decoder.deinit(gpa);
    try core.addStream(&decoder);
    defer core.discard(&decoder);
    host.share = .{ .p = 0.5 };
    host.service(std.math.maxInt(u64));
    try std.testing.expectEqual(@as(usize, 1), x.polls);
    try std.testing.expectEqual(@as(usize, 0), x.pumps);
    host.service(std.math.maxInt(u64)); // still owed: completion maintenance only
    try std.testing.expectEqual(@as(usize, 1), x.polls);
    try std.testing.expectEqual(@as(usize, 2), x.maintenance);
    host.share.rounded(std.math.maxInt(i64)); // deterministic model of paid decode time
    host.service(std.math.maxInt(u64));
    try std.testing.expectEqual(@as(usize, 2), x.polls);
    try std.testing.expectEqualSlices(u64, &.{ 2, 1 }, x.order[0..2]);
    try std.testing.expectEqual(@as(usize, 0), x.pumps);
}

test "aggregate tier pending restores complete when all decoders are paused" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    var x: AggregateDebtTier = .{ .host = &host, .complete = true };
    var store = x.cache();
    defer store.deinit();
    host.cache = &store;
    try AggregateDebtTier.add(&store, 1);
    try AggregateDebtTier.add(&store, 2);
    var decoder = try lanes.Stream.init(gpa, .{ .id = "paused", .prompt = &.{ 2, 7, 1, 8 }, .max_new = 8 });
    defer decoder.deinit(gpa);
    try core.addStream(&decoder);
    defer core.discard(&decoder);
    decoder.paused = true;
    host.share = .{ .p = 0.5, .debt_ns = 1000 };
    host.service(std.math.maxInt(u64));
    try std.testing.expectEqual(@as(usize, 2), x.polls);
    try std.testing.expectEqual(@as(usize, 1), x.pumps);
    try std.testing.expectEqual(@as(usize, 0), store.restoring.items.len);
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);
}

test "aggregate tier maintenance fatal fences the decode offer" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    var x: AggregateDebtTier = .{ .host = &host, .trip_on_maintain = true };
    var store = x.cache();
    defer store.deinit();
    host.cache = &store;
    host.device_probe = .{ .ctx = &x, .failed = AggregateDebtTier.failed };
    var decoder = try lanes.Stream.init(gpa, .{ .id = "decode", .prompt = &.{ 2, 7, 1, 8 }, .max_new = 8 });
    defer decoder.deinit(gpa);
    try core.addStream(&decoder);
    defer core.discard(&decoder);
    host.share = .{ .p = 0.5, .debt_ns = 1000 };
    const before = decoder.context.items.len;
    try std.testing.expect(host.runIteration());
    try std.testing.expect(host.broken());
    try std.testing.expectEqual(before, decoder.context.items.len);
    try std.testing.expectEqual(@as(usize, 1), x.maintenance);
    try std.testing.expectEqual(@as(usize, 0), x.polls);
    try std.testing.expectEqual(@as(usize, 0), x.pumps);
}

// UNRUN: arm allocation failure only after both backend allocations and host admissions exist.
test "failAll drains admitted ownership with allocator failure after setup" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Seen = struct {
        host: *LaneHost,
        done: [2]usize = .{ 0, 0 },
        fn event(ptr: *anyopaque, id: Id, ev: *const Event) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            switch (ev.*) {
                .finished => |f| {
                    // Taking the lock here also detects callbacks incorrectly made under the lock.
                    self.host.lock();
                    defer self.host.unlock();
                    std.debug.assert(f.reason == .failed);
                    std.debug.assert(std.mem.eql(u8, f.message, "recoverable round failure"));
                    std.debug.assert(self.host.admitted.items.len == 0);
                    self.done[@intCast(id - 1)] += 1;
                },
                else => {},
            }
        }
    };
    var seen: Seen = .{ .host = &host };
    const rows: [1024]u32 = @splat(5);
    const req: Request = .{ .prompt = &rows, .max_tokens = 8 };
    try host.engine().submit(1, &req, .{ .ctx = &seen, .event = Seen.event });
    try host.engine().submit(2, &req, .{ .ctx = &seen, .event = Seen.event });
    host.admitBatch();
    const decoder = host.admitted.items[0];
    for (0..4) |_| host.fillAt(decoder);
    host.fillAt(host.admitted.items[1]);
    try std.testing.expectEqual(@as(usize, 1), core.activeCount());
    try std.testing.expectEqual(@as(usize, 2), target.lanes.count());
    try std.testing.expect(host.prefill_telemetry.progress > 0);
    const allocations = failing.alloc_index;
    const resizes = failing.resize_index;
    failing.fail_index = allocations;
    failing.resize_fail_index = resizes;
    host.failAll("recoverable round failure");
    // No allocation was even attempted; a duplicate-and-catch implementation loses both jobs.
    try std.testing.expectEqual(allocations, failing.alloc_index);
    try std.testing.expectEqual(resizes, failing.resize_index);
    try std.testing.expect(!failing.has_induced_failure);
    try std.testing.expectEqual([2]usize{ 1, 1 }, seen.done);
    try std.testing.expectEqual(@as(usize, 0), host.admitted.items.len);
    try std.testing.expectEqual(@as(usize, 0), core.activeCount());
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expectEqual(@as(u32, 0), host.prefill_telemetry.active);
    try std.testing.expectEqual(@as(u64, 0), host.prefill_telemetry.pending);
    try std.testing.expectEqual(@as(u64, 0), host.prefill_telemetry.progress);
    host.failAll("recoverable round failure");
    try std.testing.expectEqual([2]usize{ 1, 1 }, seen.done);
    failing.fail_index = std.math.maxInt(usize);
}

// UNRUN: exercise the generic post-progress ContextFull branch, not just begin refusal.
test "ContextFull retry resets destroyed attempt progress before readmission" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Quiet = struct { fn event(_: *anyopaque, _: Id, _: *const Event) void {} };
    var dummy: u8 = 0;
    const rows: [1024]u32 = @splat(5);
    const req: Request = .{ .prompt = &rows, .max_tokens = 1 };
    try host.engine().submit(1, &req, .{ .ctx = &dummy, .event = Quiet.event });
    try host.engine().submit(2, &req, .{ .ctx = &dummy, .event = Quiet.event });
    host.admitBatch();
    const resident = host.admitted.items[0];
    const retry = host.admitted.items[1];
    host.fillAt(resident);
    for (0..3) |_| host.fillAt(retry);
    try std.testing.expectEqual(@as(u32, 768), retry.fill_at);
    try std.testing.expectEqual(@as(usize, 2), target.lanes.count());
    host.fillFailed(retry, error.ContextFull);
    try std.testing.expectEqual(@as(u32, 0), retry.fill_at);
    try std.testing.expect(!retry.started and !retry.fill_began);
    try std.testing.expectEqual(@as(usize, 1), target.lanes.count());
    try std.testing.expect(!host.admitOne());
    host.engine().cancel(1);
    host.takeCancels();
    try std.testing.expect(host.admitOne());
    try std.testing.expect(host.admitted.items[0] == retry);
    // Before any callback on the new stream, all 1024 rows remain outstanding.
    try std.testing.expectEqual(@as(u64, 0), host.prefill_telemetry.progress);
    try std.testing.expectEqual(@as(u64, 1024), host.prefill_telemetry.pending);
    try std.testing.expect(host.pickFill(false).? == retry);
    host.fillAt(retry);
    try std.testing.expectEqual(@as(u32, 256), retry.fill_at);
    host.engine().cancel(2);
    host.takeCancels();
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

// UNRUN: no compiler or runtime invoked. Every allocation fault is armed after setup.
test "cancellation drain retains keep and callback submissions without allocation" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var core: lanes.Engine = undefined; // no admitted jobs: backend must never be touched
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const req: Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    const Seen = struct {
        host: *LaneHost,
        done: [4]usize = @splat(0),
        fn event(ptr: *anyopaque, id: Id, ev: *const Event) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            switch (ev.*) {
                .finished => |f| {
                    self.host.lock();
                    defer self.host.unlock();
                    std.debug.assert(f.reason == .cancelled);
                    self.done[@intCast(id - 1)] += 1;
                    // Model a producer arriving during the unlocked callback. Capacity
                    // was reserved before fault injection, just as a concurrent cancel.
                    if (id == 1) self.host.cancels.appendAssumeCapacity(4);
                },
                else => {},
            }
        }
    };
    var seen: Seen = .{ .host = &host };
    defer {
        failing.fail_index = std.math.maxInt(usize);
        failing.resize_fail_index = std.math.maxInt(usize);
        host.closeAll();
        host.stop();
    }
    for (1..5) |id| try host.engine().submit(@intCast(id), &req, .{ .ctx = &seen, .event = Seen.event });
    // Exercise ready ownership as well as queued ownership.
    const ready = host.queued.orderedRemove(1);
    host.ready.append(gpa, ready) catch |err| {
        host.finish(ready, .cancelled, "");
        return err;
    };
    try host.cancels.ensureTotalCapacity(gpa, 8);
    try host.cancels.appendSlice(gpa, &.{ 1, 2, 3, 1 });
    const allocations = failing.alloc_index;
    const resizes = failing.resize_index;
    failing.fail_index = allocations;
    failing.resize_fail_index = resizes;
    host.takeCancelsKeeping(3);
    try std.testing.expectEqual([4]usize{ 1, 1, 0, 0 }, seen.done);
    try std.testing.expectEqualSlices(Id, &.{ 3, 4 }, host.cancels.items);
    try std.testing.expectEqual(@as(usize, 0), host.ready.items.len);
    host.takeCancels();
    try std.testing.expectEqual([4]usize{ 1, 1, 1, 1 }, seen.done);
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
    try std.testing.expectEqual(@as(usize, 0), host.cancels.items.len);
    try std.testing.expect(host.cancel_held == null);
    try std.testing.expectEqual(allocations, failing.alloc_index);
    try std.testing.expectEqual(resizes, failing.resize_index);
    try std.testing.expect(!failing.has_induced_failure);
}

test "cancellation drain cancels shared restore at last waiter without allocation" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var core: lanes.Engine = undefined;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Seen = struct {
        host: *LaneHost,
        cancelled: usize = 0,
        finished: usize = 0,
        fn bytes(_: *anyopaque, _: u32) u64 { return 1; }
        fn save(ctx: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved { return ctx; }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
        fn drop(_: *anyopaque, _: pc.Saved) void {}
        fn cancel(ptr: *anyopaque, _: u64) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.host.lock();
            defer self.host.unlock();
            std.debug.assert(self.host.loading.items.len == 0);
            self.cancelled += 1;
        }
        fn event(ptr: *anyopaque, _: Id, ev: *const Event) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            switch (ev.*) {
                .finished => |f| {
                    self.host.lock();
                    defer self.host.unlock();
                    std.debug.assert(f.reason == .cancelled);
                    self.finished += 1;
                },
                else => {},
            }
        }
    };
    var seen: Seen = .{ .host = &host };
    var store = pc.Store.init(gpa, .{ .ptr = &seen, .vtable = &.{ .bytes = Seen.bytes, .save = Seen.save, .restore = Seen.restore, .drop = Seen.drop } }, .{}, 100);
    defer { store.tier = null; store.deinit(); }
    const req: Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    defer {
        failing.fail_index = std.math.maxInt(usize);
        failing.resize_fail_index = std.math.maxInt(usize);
        host.closeAll();
        host.stop();
    }
    for (1..3) |id| try host.engine().submit(@intCast(id), &req, .{ .ctx = &seen, .event = Seen.event });
    try host.loading.ensureTotalCapacity(gpa, 2);
    // Reserve every stage before constructing the restore so teardown always owns it.
    try store.restoring.ensureUnusedCapacity(gpa, 1);
    const key = try gpa.dupe(u32, &.{1});
    const r = gpa.create(pc.Restore) catch |err| { gpa.free(key); return err; };
    r.* = .{ .point = .{ .id = 1, .at = 1, .tokens = 1 }, .job = 1, .key = key };
    store.restoring.appendAssumeCapacity(r);
    store.tier = .{ .ptr = &seen, .vtable = &.{ .lookup = undefined, .begin = undefined, .poll = undefined, .cancel = Seen.cancel, .keep = undefined, .forget = undefined, .pump = undefined, .flush = undefined, .counts = undefined } };
    host.cache = &store;
    while (host.queued.items.len > 0) host.loading.appendAssumeCapacity(.{ .job = host.queued.orderedRemove(0), .restore = r, .since = 0 });
    try host.cancels.appendSlice(gpa, &.{ 1, 1, 2, 2 });
    const allocations = failing.alloc_index;
    const resizes = failing.resize_index;
    failing.fail_index = allocations;
    failing.resize_fail_index = resizes;
    host.takeCancelsKeeping(2);
    try std.testing.expectEqual(@as(usize, 0), seen.cancelled);
    try std.testing.expectEqual(@as(usize, 1), seen.finished);
    try std.testing.expectEqual(@as(usize, 1), store.restoring.items.len);
    host.takeCancels();
    try std.testing.expectEqual(@as(usize, 1), seen.cancelled);
    try std.testing.expectEqual(@as(usize, 2), seen.finished);
    try std.testing.expectEqual(@as(usize, 0), store.restoring.items.len);
    try std.testing.expectEqual(@as(usize, 0), host.loading.items.len);
    try std.testing.expectEqual(allocations, failing.alloc_index);
    try std.testing.expectEqual(resizes, failing.resize_index);
    try std.testing.expect(!failing.has_induced_failure);
}

// SOURCE ONLY / UNRUN: final retired unpin is the callback boundary under test.
test "final reported unpin retains poisoned job and healthy finish is once only" {
    const gpa = std.testing.allocator;
    const Seen = struct {
        host: *LaneHost,
        poison: bool,
        drops: usize = 0,
        terminals: usize = 0,
        fn failed(_: *anyopaque) bool { return false; }
        fn bytes(_: *anyopaque, _: u32) u64 { return 1; }
        fn save(ctx: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved { return ctx; }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
        fn drop(ptr: *anyopaque, _: pc.Saved) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.drops += 1;
            if (self.poison) self.host.peer_monitor.?.signal(error.ShardDiverged);
        }
        fn event(ptr: *anyopaque, _: Id, ev: *const Event) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            switch (ev.*) {
                .finished => self.terminals += 1,
                else => {},
            }
        }
    };
    for ([_]bool{ true, false }) |poison| {
        var core: lanes.Engine = undefined; // queued cancellation must not use the backend
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
        var seen: Seen = .{ .host = &host, .poison = poison };
        host.peer_monitor = .{ .io = std.testing.io, .probe = .{ .ctx = &seen, .failed = Seen.failed }, .policy = .{} };
        var store = pc.Store.init(gpa, .{ .ptr = &seen, .vtable = &.{ .bytes = Seen.bytes, .save = Seen.save, .restore = Seen.restore, .drop = Seen.drop } }, .{ .min_prompt = 0 }, 100);
        defer store.deinit();
        host.cache = &store;
        const req: Request = .{ .prompt = &.{ 1, 2, 3, 4 }, .max_tokens = 1 };
        try std.testing.expect(store.keep(req.prompt, 3, null, &.{}));
        const entry = store.find(req.prompt, &.{}).?;
        store.pin(entry);
        store.forget(req.prompt, 3); // pinned-safe retirement: drop only at reported's final unpin
        try std.testing.expectEqual(@as(usize, 0), seen.drops);
        const job = try gpa.create(LaneHost.Job);
        job.* = .{ .host = &host, .id = 1, .request = &req, .sink = .{ .ctx = &seen, .event = Seen.event }, .entry = entry };
        job.marks = try gpa.dupe(u32, &.{ 9, 10 });
        job.lineage = try gpa.alloc(pc.Fork, 1);
        const marks = job.marks;
        const lineage = job.lineage;
        try host.queued.append(gpa, job);
        try host.cancels.appendSlice(gpa, &.{ 1, 2 });
        host.peer_monitor.?.submitted(); // synthetic submission follows normal monitor accounting
        defer {
            // Explicit CPU-only teardown AFTER assertions. Production never clears poison.
            host.peer_monitor = null;
            if (host.finish_held) |held| {
                host.finish_held = null;
                host.cancel_held = null;
                gpa.free(held.marks);
                gpa.free(held.lineage);
                gpa.destroy(held);
            }
            host.queued.deinit(gpa);
            host.cancels.deinit(gpa);
        }
        host.takeCancels();
        try std.testing.expectEqual(@as(usize, 1), seen.drops);
        try std.testing.expectEqual(@as(u64, 0), store.held);
        try std.testing.expectEqual(@as(u32, if (poison) 1 else 0), host.peer_monitor.?.pending.load(.acquire));
        try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
        if (poison) {
            try std.testing.expect(host.broken());
            try std.testing.expect(host.cancel_held == job);
            try std.testing.expect(host.finish_held == job);
            try std.testing.expect(job.finish_retained and job.finish_next == null);
            try std.testing.expect(job.entry == null);
            try std.testing.expectEqualSlices(u32, &.{ 9, 10 }, job.marks);
            try std.testing.expect(job.marks.ptr == marks.ptr and job.lineage.ptr == lineage.ptr);
            try std.testing.expectEqual(@as(usize, 1), job.lineage.len);
            try std.testing.expectEqual(@as(usize, 0), seen.terminals);
            try std.testing.expectEqualSlices(Id, &.{2}, host.cancels.items);
            // Accidental finalization retry is inert, including after callback consumed entry.
            host.finish(job, .cancelled, "");
            host.takeCancels();
            try std.testing.expect(host.finish_held == job and job.finish_next == null);
            try std.testing.expectEqual(@as(usize, 0), seen.terminals);
            try std.testing.expectEqual(@as(usize, 1), seen.drops);
        } else {
            try std.testing.expect(!host.broken());
            try std.testing.expect(host.finish_held == null and host.cancel_held == null);
            try std.testing.expectEqual(@as(usize, 1), seen.terminals);
            host.takeCancels(); // no reference to the freed job, no repeated terminal or drop
            try std.testing.expectEqual(@as(usize, 1), seen.terminals);
            try std.testing.expectEqual(@as(usize, 1), seen.drops);
        }
    }
}

// SOURCE ONLY / UNRUN: real step terminal commit, then recoverable failure before settlement.
test "FR01 terminal step failure drains release custody exactly once" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 1 }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Seen = struct {
        done: [2]usize = .{ 0, 0 },
        failed: bool = false,
        fn event(ptr: *anyopaque, id: Id, ev: *const Event) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            switch (ev.*) {
                .finished => |end| {
                    self.done[@intCast(id - 1)] += 1;
                    if (id == 1) self.failed = end.reason == .failed and std.mem.eql(u8, end.message, "TestPostCommitFailure");
                },
                else => {},
            }
        }
    };
    var seen: Seen = .{};
    const req: Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 2, .drafts = false };
    try host.engine().submit(1, &req, .{ .ctx = &seen, .event = Seen.event });
    host.admitBatch();
    host.fillAt(host.admitted.items[0]);
    try std.testing.expectEqual(@as(usize, 1), core.live.items.len);
    try std.testing.expectEqual(@as(usize, 1), host.admitted.items[0].stream.emitted().len);
    core.test_fail_terminal_commit = true;
    try std.testing.expect(host.runIteration());
    try std.testing.expect(!core.test_fail_terminal_commit); // actual terminal commit consumed the seam
    try std.testing.expect(seen.failed);
    try std.testing.expectEqual(@as(usize, 1), target.release_calls);
    try std.testing.expectEqual(@as(usize, 0), core.live.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expectEqual(@as(usize, 0), host.admitted.capacity);
    host.failAll("repeat");
    try std.testing.expectEqual(@as(usize, 1), target.release_calls);
    try host.engine().submit(2, &req, .{ .ctx = &seen, .event = Seen.event });
    try std.testing.expect(host.runIteration());
    try std.testing.expectEqual([2]usize{ 1, 1 }, seen.done);
    try std.testing.expectEqual(@as(usize, 2), target.release_calls);
    try std.testing.expectEqual(@as(usize, 0), core.live.items.len);
}

// Includes a finished release receipt and a terminal open that never reached core.live.
test "FR01 partial terminal open and released finished control drain once" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 1 }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Seen = struct {
        done: usize = 0,
        fn event(ptr: *anyopaque, _: Id, ev: *const Event) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            if (ev.* == .finished) self.done += 1;
        }
    };
    var seen: Seen = .{};
    const req: Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 1, .drafts = false };
    try host.engine().submit(1, &req, .{ .ctx = &seen, .event = Seen.event });
    try host.engine().submit(2, &req, .{ .ctx = &seen, .event = Seen.event });
    host.admitBatch();
    const released = host.admitted.items[0];
    // Open without deliver so the already released Job is still host-owned.
    try core.addStream(&released.stream);
    try std.testing.expect(released.stream.finished and !released.stream.backend_owned);
    try std.testing.expectEqual(@as(usize, 1), target.release_calls);
    core.test_fail_terminal_commit = true;
    host.fillAt(host.admitted.items[1]); // first token terminal, openStream errors before live insertion
    try std.testing.expect(!core.test_fail_terminal_commit);
    try std.testing.expectEqual(@as(usize, 2), target.release_calls);
    host.failAll("control drain");
    try std.testing.expectEqual(@as(usize, 2), target.release_calls);
    try std.testing.expectEqual(@as(usize, 2), seen.done);
    try std.testing.expectEqual(@as(usize, 0), core.live.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}


// Single-threaded fake backend + durable tier: no timing races or GPU dependencies.
const RoomRig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    store: pc.Store,
    tier: @import("prompt_cache_fake_tier.zig").FakeTier,
    host: LaneHost,
    decoder: lanes.Stream,
    const a = std.testing.allocator;
    const key = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 7, 7, 5, 3, 8, 8 };
    const Reply = struct {
        cached: ?u32 = null,
        done: ?Reason = null,
        tokens: std.ArrayList(u32) = .empty,
        fn event(ptr: *anyopaque, _: Id, e: *const Event) void {
            const b: *Reply = @ptrCast(@alignCast(ptr));
            switch (e.*) {
                .prefilled => |v| b.cached = v,
                .finished => |v| b.done = v.reason,
                .tokens => |v| b.tokens.appendSlice(a, v) catch unreachable,
            }
        }
        fn sink(b: *Reply) Sink { return .{ .ctx = b, .event = event }; }
        fn deinit(b: *Reply) void { b.tokens.deinit(a); }
    };
    fn bytes(_: *anyopaque, at: u32) u64 { return 64 + 4 * @as(u64, at); }
    fn save(p: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(p));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void { return error.BackendRestores; }
    fn drop(p: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(p)); f.drop(saved);
    }
    fn pack(_: *anyopaque, alloc: Allocator, saved: pc.Saved) ![]u8 {
        const list: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        return alloc.dupe(u8, std.mem.sliceAsBytes(list.items));
    }
    fn unpack(_: *anyopaque, data: []const u8) !pc.Saved {
        const list = try a.create(std.ArrayList(u32));
        errdefer a.destroy(list);
        list.* = .empty;
        try list.resize(a, data.len / 4);
        @memcpy(std.mem.sliceAsBytes(list.items), data);
        return list;
    }
    fn create(ms: u32) !*RoomRig {
        const x = try a.create(RoomRig);
        x.* = .{ .cfg = undefined, .target = .{ .gpa = a }, .core = undefined, .store = undefined,
            .tier = .{ .gpa = a, .codec = .{ .ctx = x, .pack = pack, .unpack = unpack } },
            .host = undefined, .decoder = undefined };
        x.cfg = try lanes.Config.init(a, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
        x.core = lanes.Engine.init(a, &x.cfg, x.target.backend(), x.clock.clock());
        x.store = pc.Store.init(a, .{ .ptr = &x.target, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        x.store.setTier(x.tier.tier(), 0);
        x.host = LaneHost.init(a, std.testing.io, &x.core, .{ .lanes = 4 });
        x.host.cache = &x.store;
        x.host.configure(.{ .room_wait_ms = ms });
        x.host.share.p = 1;
        x.host.interleave_ns = 0;
        x.decoder = try lanes.Stream.init(a, .{ .id = "resident", .prompt = &.{ 2, 7, 1, 8 }, .max_new = 1000 });
        try x.core.addStream(&x.decoder);
        _ = try x.tier.put(key[0..8], 8, std.mem.sliceAsBytes(key[0..8]), 0, 0, false);
        x.tier.begin_error = error.ContextFull;
        return x;
    }
    fn destroy(x: *RoomRig) void {
        x.host.closeAll();
        x.host.stop();
        x.core.discard(&x.decoder);
        x.decoder.deinit(a);
        x.store.deinit();
        x.tier.deinit();
        x.core.deinit();
        x.target.deinit();
        x.cfg.deinit(a);
        a.destroy(x);
    }
    fn submit(x: *RoomRig, id: Id, req: *const Request, out: *Reply) !void {
        try x.host.engine().submit(id, req, out.sink());
    }
    fn retry(x: *RoomRig) void {
        if (x.host.queued.items.len > 0) x.host.queued.items[0].room_retry_at = 0;
    }
};

test "restore room refusal then room restores exact tokens without double reservation" {
    const x = try RoomRig.create(30_000);
    defer x.destroy();
    var b: RoomRig.Reply = .{};
    defer b.deinit();
    var c: RoomRig.Reply = .{};
    defer c.deinit();
    const req: Request = .{ .prompt = &RoomRig.key, .max_tokens = 4, .history_len = 14 };
    try x.submit(1, &req, &b);
    try x.submit(2, &req, &c);
    x.host.admitBatch();
    try std.testing.expectEqual(@as(usize, 2), x.host.queued.items.len);
    try std.testing.expectEqual(@as(u64, 1), x.host.host_counts.restore_room_waits);
    try std.testing.expect(!x.host.queued.items[0].tried_disk);
    try std.testing.expectEqual(@as(usize, 0), x.store.restoring.items.len);
    const since = x.host.queued.items[0].room_since;
    x.retry();
    x.host.admitBatch();
    try std.testing.expectEqual(since, x.host.queued.items[0].room_since);
    try std.testing.expectEqual(@as(u64, 1), x.host.host_counts.restore_room_waits);
    // Decoders continue during refusal.
    const before = x.decoder.context.items.len;
    _ = x.host.runIteration();
    try std.testing.expect(x.decoder.context.items.len > before);
    x.tier.begin_error = null;
    x.retry();
    x.host.admitBatch(); // whole-pass mode parks one per decoder round
    x.host.admitBatch();
    try std.testing.expectEqual(@as(usize, 2), x.host.loading.items.len);
    try std.testing.expectEqual(@as(usize, 1), x.store.restoring.items.len);
    try std.testing.expectEqual(@as(u64, 1), x.store.stats.restores_shared);
    for (0..100) |_| {
        if (b.done != null and c.done != null) break;
        _ = x.host.runIteration();
    }
    try std.testing.expectEqual(@as(?Reason, .length), b.done);
    try std.testing.expectEqual(@as(?Reason, .length), c.done);
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try std.testing.expectEqual(@as(?u32, 14), c.cached); // first request already kept the longer prefix
    try std.testing.expectEqualSlices(u32, b.tokens.items, c.tokens.items);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(RoomRig.a);
    try history.appendSlice(RoomRig.a, &RoomRig.key);
    for (b.tokens.items) |token| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), token);
        try history.append(RoomRig.a, token);
    }
    try std.testing.expectEqual(@as(u64, 0), x.host.host_counts.restore_room_expiries);
}

test "restore room expiry is once only under decode debt and full wait then unblocks second request" {
    const x = try RoomRig.create(1);
    defer x.destroy();
    var b: RoomRig.Reply = .{}; defer b.deinit();
    var c: RoomRig.Reply = .{}; defer c.deinit();
    const req: Request = .{ .prompt = &RoomRig.key, .max_tokens = 1 };
    const other: Request = .{ .prompt = &.{ 8, 2, 8, 4 }, .max_tokens = 1 };
    try x.submit(1, &req, &b);
    try x.submit(2, &other, &c);
    x.host.admitBatch();
    const job = x.host.queued.items[0];
    job.room_since = std.Io.Clock.awake.now(x.host.io).toNanoseconds() - 2 * std.time.ns_per_ms;
    x.host.full_wait = 1;
    x.host.share.debt_ns = 1000;
    try std.testing.expect(!x.host.admitOne());
    try std.testing.expect(job.tried_disk);
    try std.testing.expect(job.room_since == null);
    try std.testing.expect(!x.host.admitOne());
    try std.testing.expectEqual(@as(u64, 1), x.host.host_counts.restore_room_expiries);
    x.host.full_wait = null;
    x.host.share.debt_ns = 0;
    for (0..30) |_| { if (b.done != null and c.done != null) break; _ = x.host.runIteration(); }
    try std.testing.expectEqual(@as(?Reason, .length), b.done);
    try std.testing.expectEqual(@as(?Reason, .length), c.done);
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    try std.testing.expectEqual(@as(u64, 0), x.store.stats.restores_begun);
}

test "restore room cancellation releases front and never cancels an unstarted restore" {
    const x = try RoomRig.create(30_000);
    defer x.destroy();
    var b: RoomRig.Reply = .{}; defer b.deinit();
    var c: RoomRig.Reply = .{}; defer c.deinit();
    const req: Request = .{ .prompt = &RoomRig.key, .max_tokens = 1 };
    const other: Request = .{ .prompt = &.{ 8, 2, 8, 4 }, .max_tokens = 1 };
    try x.submit(1, &req, &b);
    try x.submit(2, &other, &c);
    x.host.admitBatch();
    x.host.engine().cancel(1);
    x.host.takeCancels();
    try std.testing.expectEqual(@as(?Reason, .cancelled), b.done);
    for (0..30) |_| { if (c.done != null) break; _ = x.host.runIteration(); }
    try std.testing.expectEqual(@as(?Reason, .length), c.done);
    try std.testing.expectEqual(@as(u64, 0), x.store.stats.restores_cancelled);
    try std.testing.expectEqual(@as(u64, 0), x.host.host_counts.restore_room_expiries);
}

test "restore room flag off no resident and non-room errors preserve immediate fallback" {
    for (0..4) |mode| {
        const x = try RoomRig.create(if (mode == 0) 0 else 30_000);
        defer x.destroy();
        var b: RoomRig.Reply = .{}; defer b.deinit();
        if (mode == 1) x.core.discard(&x.decoder);
        if (mode == 2) x.tier.begin_error = error.PointMissing;
        if (mode == 3) x.host.kv_parts = .{ .ctx = x, .read = struct {
            fn read(_: *anyopaque, out: *lanes.kv_parts.Stats) void { out.* = .{}; }
        }.read };
        const req: Request = .{ .prompt = &RoomRig.key, .max_tokens = 1 };
        try x.submit(1, &req, &b);
        for (0..30) |_| { if (b.done != null) break; _ = x.host.runIteration(); }
        try std.testing.expectEqual(@as(?Reason, .length), b.done);
        try std.testing.expectEqual(@as(?u32, 0), b.cached);
        try std.testing.expectEqual(@as(u64, 0), x.host.host_counts.restore_room_waits);
        try std.testing.expectEqual(@as(u64, 0), x.host.host_counts.restore_room_expiries);
    }
}

test "restore room waits for suspended fill then falls back when no resident can release room" {
    const x = try RoomRig.create(30_000);
    defer x.destroy();
    x.core.discard(&x.decoder);
    x.target.prefill_chunks = 4;
    x.core.backend = x.target.stepped();
    var resident: RoomRig.Reply = .{}; defer resident.deinit();
    var waiting: RoomRig.Reply = .{}; defer waiting.deinit();
    const rows: [1024]u32 = @splat(8);
    const fill: Request = .{ .prompt = &rows, .max_tokens = 1 };
    const req: Request = .{ .prompt = &RoomRig.key, .max_tokens = 1 };
    try x.submit(1, &fill, &resident);
    try std.testing.expect(x.host.admitOne());
    x.host.fillAt(x.host.admitted.items[0]);
    try std.testing.expectEqual(@as(usize, 0), x.core.activeCount());
    try std.testing.expectEqual(@as(usize, 1), x.host.residentCount(null));
    try x.submit(2, &req, &waiting);
    try std.testing.expect(!x.host.admitOne());
    try std.testing.expectEqual(@as(u64, 1), x.host.host_counts.restore_room_waits);
    x.host.engine().cancel(1);
    x.host.takeCancels();
    try std.testing.expectEqual(@as(usize, 0), x.host.residentCount(null));
    // No timer wait now: impossible placement cannot leave an idle host stalled.
    try std.testing.expect(x.host.admitOne());
    try std.testing.expectEqual(@as(usize, 0), x.host.queued.items.len);
    for (0..30) |_| { if (waiting.done != null) break; _ = x.host.runIteration(); }
    try std.testing.expectEqual(@as(?Reason, .length), waiting.done);
    try std.testing.expectEqual(@as(?u32, 0), waiting.cached);
    try std.testing.expectEqual(@as(u64, 0), x.host.host_counts.restore_room_expiries);
}

test "prefill slice parks decode only with terminal custody and cancellation" {
    for ([_]bool{ false, true }) |cancel_fill| {
        const gpa = std.testing.allocator;
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
        defer core.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
        defer host.stop();
        const Watch = struct {
            host: *LaneHost,
            target: *lanes.fake.Fake,
            cancel_fill: bool,
            parks: usize = 0,
            release_at: usize = 0,
            good: bool = true,
            tokens: [2]std.ArrayList(u32) = @splat(.empty),
            done: usize = 0,
            fn event(ctx: *anyopaque, id: Id, ev: *const Event) void {
                const w: *@This() = @ptrCast(@alignCast(ctx));
                switch (ev.*) {
                    .tokens => |ts| if (id <= 2) {
                        w.tokens[id - 1].appendSlice(std.testing.allocator, ts) catch unreachable;
                    },
                    .finished => w.done += 1,
                    else => {},
                }
            }
            fn park(ctx: *anyopaque, _: *lanes.Stream) void {
                const w: *@This() = @ptrCast(@alignCast(ctx));
                if (w.parks == 0) w.release_at = w.target.release_calls;
                w.good = w.good and w.target.release_calls == w.release_at;
                w.good = w.good and w.host.queued.items.len == 1;
                w.parks += 1;
                if (w.cancel_fill and w.parks == 2) w.host.engine().cancel(2);
            }
        };
        var watch: Watch = .{ .host = &host, .target = &target, .cancel_fill = cancel_fill };
        defer for (&watch.tokens) |*ts| ts.deinit(gpa);
        const prompt_a = [_]u32{ 2, 7, 1, 8 };
        const prompt_b: [1024]u32 = @splat(5);
        const req_a: Request = .{ .prompt = &prompt_a, .max_tokens = 3 };
        const req_b: Request = .{ .prompt = &prompt_b, .max_tokens = 5 };
        const req_c: Request = .{ .prompt = &prompt_a, .max_tokens = 1 };
        try host.engine().submit(1, &req_a, .{ .ctx = &watch, .event = Watch.event });
        host.admitBatch();
        host.runFills(); // decoder is resident, but no round has run
        try host.engine().submit(2, &req_b, .{ .ctx = &watch, .event = Watch.event });
        host.admitBatch();
        try host.engine().submit(3, &req_c, .{ .ctx = &watch, .event = Watch.event });
        target.prefill_slices = 5;
        target.slice_hook = Watch.park;
        target.slice_ctx = &watch;
        const filling = host.admitted.items[1];
        host.fillAt(filling); // one chunk, with four possible parks
        try std.testing.expect(watch.good);
        try std.testing.expectEqual(@as(usize, if (cancel_fill) 2 else 4), watch.parks);
        try std.testing.expect(!core.hold_finished);
        try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
        target.prefill_slices = 0;
        target.slice_hook = null;
        var rounds: usize = 0;
        while (watch.done < 3 and rounds < 100) : (rounds += 1) _ = host.runIteration();
        try std.testing.expectEqual(@as(usize, 3), watch.done);
        for ([_]*const Request{ &req_a, &req_b }, &watch.tokens, 0..) |req, *ts, i| {
            if (i == 1 and cancel_fill) {
                try std.testing.expectEqual(@as(usize, 0), ts.items.len);
                continue;
            }
            var history: std.ArrayList(u32) = .empty;
            defer history.deinit(gpa);
            try history.appendSlice(gpa, req.prompt);
            try std.testing.expectEqual(req.max_tokens, ts.items.len);
            for (ts.items) |token| {
                try std.testing.expectEqual(lanes.fake.next(history.items, req.sampling, history.items.len), token);
                try history.append(gpa, token);
            }
        }
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}

test "prefill slice fatal peer fences decode and leaves cleanup to the owner" {
    const Probe = struct { fn failed(_: *anyopaque) bool { return true; } };
    var byte: u8 = 0;
    var h: LaneHost = undefined;
    h.peer_monitor = null;
    h.device_probe = .{ .ctx = &byte, .failed = Probe.failed };
    try std.testing.expect(!LaneHost.sliceWanted(&h));
    try std.testing.expectError(error.PeerDown, LaneHost.sliceBetween(&h));
}

test "prefill slice share pays each park with cap runnable and held completion" {
    const Case = struct { p: f64 = 0.5, cap_ms: i64 = 250, finish: bool = false, survivor: bool = false, paused: bool = false, cancel: bool = false, rounds: u64, debt_ms: i64 };
    for ([_]Case{
        .{ .p = 1, .rounds = 2, .debt_ms = 0 },
        .{ .rounds = 6, .debt_ms = 60 },
        .{ .cap_ms = 0, .rounds = 6, .debt_ms = 60 },
        .{ .cap_ms = 80, .rounds = 4, .debt_ms = 140 },
        .{ .cap_ms = 1, .rounds = 2, .debt_ms = 220 },
        .{ .finish = true, .rounds = 1, .debt_ms = 260 },
        .{ .finish = true, .survivor = true, .rounds = 6, .debt_ms = 60 },
        .{ .paused = true, .rounds = 0, .debt_ms = 0 },
        .{ .cancel = true, .rounds = 1, .debt_ms = 60 },
    }) |case| {
        const gpa = std.testing.allocator;
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
        defer core.deinit();
        const Watch = struct {
            var ns: i96 = 0;
            var reads: usize = 0;
            host: *LaneHost,
            target: *lanes.fake.Fake,
            cancel: bool,
            baseline: usize = 0,
            rounds: u64 = 0,
            parks: usize = 0,
            park_rounds: u64 = 0,
            park_reads: usize = 0,
            expected: [2]u64,
            done: usize = 0,
            good: bool = true,
            fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp { reads += 1; return .{ .nanoseconds = ns }; }
            fn event(ctx: *anyopaque, _: Id, ev: *const Event) void {
                const w: *@This() = @ptrCast(@alignCast(ctx));
                if (ev.* == .finished) w.done += 1;
                if (w.host.core.hold_finished) {
                    w.good = w.good and w.target.release_calls == w.baseline and w.done == 0;
                    w.good = w.good and w.host.queued.items.len == 1;
                }
            }
            fn park(ctx: *anyopaque, _: *lanes.Stream) void {
                const w: *@This() = @ptrCast(@alignCast(ctx));
                w.good = w.good and w.target.release_calls == w.baseline and w.done == 0;
                if (w.parks > 0) w.checkPark();
                w.park_rounds = w.rounds;
                w.park_reads = reads;
                w.parks += 1;
                ns += 100 * std.time.ns_per_ms;
            }
            fn checkPark(w: *@This()) void {
                w.good = w.good and w.rounds - w.park_rounds == w.expected[w.parks - 1];
                if (w.host.share.p >= 1) w.good = w.good and reads - w.park_reads == 3;
            }
            fn round(ctx: *anyopaque) void {
                const w: *@This() = @ptrCast(@alignCast(ctx));
                w.good = w.good and w.host.core.hold_finished;
                w.good = w.good and w.target.release_calls == w.baseline;
                w.rounds += 1;
                ns += 40 * std.time.ns_per_ms;
                if (w.cancel) w.host.engine().cancel(2);
            }
            fn tail(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
                const w: *@This() = @ptrCast(@alignCast(ctx));
                w.checkPark();
                ns += 100 * std.time.ns_per_ms;
            }
        };
        Watch.ns = 0;
        Watch.reads = 0;
        var vt = std.testing.io.vtable.*;
        vt.now = Watch.now;
        var io = std.testing.io;
        io.vtable = &vt;
        var host = LaneHost.init(gpa, io, &core, .{ .lanes = 4 });
        defer host.stop();
        host.share = .{};
        const per_park: u64 = if (case.p >= 1 or case.cap_ms == 1) 1 else if (case.cap_ms == 80) 2 else 3;
        var watch: Watch = .{ .host = &host, .target = &target, .cancel = case.cancel,
            .expected = if (case.paused) .{ 0, 0 } else if ((case.finish and !case.survivor) or case.cancel) .{ 1, 0 } else .{ per_park, per_park } };
        const prompt_a = [_]u32{ 2, 7, 1, 8 };
        const prompt_b: [1024]u32 = @splat(5);
        const req_a: Request = .{ .prompt = &prompt_a, .max_tokens = if (case.finish) 2 else 100 };
        const req_b: Request = .{ .prompt = &prompt_b, .max_tokens = 1 };
        try host.engine().submit(1, &req_a, .{ .ctx = &watch, .event = Watch.event });
        host.admitBatch();
        host.runFills();
        if (case.paused) host.admitted.items[0].stream.paused = true;
        var req_survivor = req_a;
        req_survivor.max_tokens = 100;
        if (case.survivor) {
            try host.engine().submit(4, &req_survivor, .{ .ctx = &watch, .event = Watch.event });
            host.admitBatch();
            host.runFills();
        }
        try host.engine().submit(2, &req_b, .{ .ctx = &watch, .event = Watch.event });
        host.admitBatch();
        try host.engine().submit(3, &req_a, .{ .ctx = &watch, .event = Watch.event });
        host.share = .{ .p = case.p, .park_max_ns = case.cap_ms * std.time.ns_per_ms };
        watch.baseline = target.release_calls;
        target.prefill_slices = 3;
        target.slice_hook = Watch.park;
        target.slice_ctx = &watch;
        target.round_hook = Watch.round;
        target.round_ctx = &watch;
        target.prefill_hook = Watch.tail;
        target.prefill_hook_ctx = &watch;
        host.runFills();
        try std.testing.expect(watch.good);
        try std.testing.expectEqual(case.rounds, watch.rounds);
        try std.testing.expectEqual(case.debt_ms * std.time.ns_per_ms, host.share.debt_ns);
        try std.testing.expectEqual(@as(i64, @intCast(case.rounds)) * 40 * std.time.ns_per_ms, host.slice_decode_ns);
        try std.testing.expect(!core.hold_finished);
        try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
        if (case.finish) try std.testing.expectEqual(@as(usize, 2), watch.done);
        target.prefill_slices = 0;
        target.slice_hook = null;
        target.round_hook = null;
        target.prefill_hook = null;
        host.closeAll();
    }
}
