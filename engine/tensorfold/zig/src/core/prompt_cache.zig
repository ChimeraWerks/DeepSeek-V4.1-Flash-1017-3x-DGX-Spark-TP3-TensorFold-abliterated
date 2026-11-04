//! Exact prompt reuse for any family and backend: states kept at prompt-pass chunk ends, found by their tokens.
//!
//! The durable tier (`Tier`) and continuation grouping (`join`) port wojo's GLM 0088 (PR #78), 0110 (PR #124;
//! BadAd84's block-tier design) and 0111 (PR #128). Every hook is off until a tier is set (`setTier`) or grouping is
//! turned on (`groups`); with both off the store behaves and counts exactly as before.
const std = @import("std");
const Allocator = std.mem.Allocator;
const imprint = @import("prompt_imprint.zig");
const forks = @import("prompt_cache_fork.zig");
const x1 = @import("checkpoint_policy.zig");
const x3 = @import("adaptive_prefix.zig");
pub const RequestState = x1.Request;

/// A family's copy of one state.
pub const Saved = *anyopaque;

/// Cache identity only, never part of the model's token stream. Null preserves
/// the original unsalted namespace, including learned on-disk prefixes.
pub const Namespace = ?[32]u8;

pub fn namespaceFor(salt: []const u8) Namespace {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(salt, &digest, .{});
    return digest;
}

/// What a family gives the cache: copies of its state while the prompt pass stands at a chunk end.
pub const Snapshots = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Optional native checkpoint grid. Null keeps all other families unchanged.
        map_mark: ?*const fn (*anyopaque, ?Saved, usize, u32) u32 = null,
        automatic_marks: ?*const fn (*anyopaque, ?Saved, usize, *[2]u32) usize = null,
        tail_window: u32 = 0,
        /// Retention hint only: called AFTER a successful verified tail continuation.
        /// Never an authority for restore, and never increments actual hit counters.
        continuation_hint: ?*const fn (*anyopaque, Saved, u64) void = null,
        adaptive_hint: ?*const fn (*anyopaque, Saved, u32) void = null,
        ghost_range: ?*const fn (*anyopaque, Saved) [2]u32 = null,
        /// The most new storage `save` takes at `at` tokens (checked against the budget before any copy).
        bytes: *const fn (ptr: *anyopaque, at: u32) u64,
        /// Copy the live state after `at` prompt tokens into new storage; `owner` is the lane core's stream (null: one stream).
        save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved,
        /// Make the live state `saved`'s: the next prompt chunk starts at its position.
        restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: Saved) anyerror!void,
        drop: *const fn (ptr: *anyopaque, saved: Saved) void,
        /// Synchronous bounded release; all rank receipts complete before return.
        drop_many: ?*const fn (*anyopaque, []const Saved) void = null,
        first_hint: ?*const fn (*anyopaque, Saved) void = null,
        /// Optional storage liveness and request compatibility, checked before longest-prefix selection.
        live: ?*const fn (ptr: *anyopaque, saved: Saved) bool = null,
        dead_reason: ?*const fn (ptr: *anyopaque, saved: Saved) u8 = null,
        compatible: ?*const fn (ptr: *anyopaque, saved: Saved, prompt_len: usize) bool = null,
        /// The storage a kept state holds: new storage up to `bytes`, or the spare storage it took (null: `bytes`).
        charged: ?*const fn (ptr: *anyopaque, saved: Saved) u64 = null,
        /// Free storage the family keeps for later saves, inside the budget (null: none).
        spare: ?*const fn (ptr: *anyopaque) u64 = null,
        /// Free spare storage until at most `room` bytes stay spare.
        trim: ?*const fn (ptr: *anyopaque, room: u64) void = null,
        /// Whether a save at `at` would take spare storage (no new storage).
        reuses: ?*const fn (ptr: *anyopaque, at: u32) bool = null,
        /// Write `saved` as learned state `key`, in files of the family's own under `dir` (null: nothing kept on disk).
        write: ?*const fn (ptr: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64) anyerror!void = null,
        /// Learned state `key` of `at` tokens, read back from its files under `dir` into new storage.
        read: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!Saved = null,
        /// Learned state `key`'s files under `dir` removed (the cap needs room, or they no longer read back).
        forget: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64) void = null,
        /// Whether `saved` still restores for a prompt of `len` tokens (not stale: the family's own rule); null: always.
        /// dsv41 hands Lanes.usable here with its tier (snaps list membership, `stale`, its grid and window).
        /// Lookups pass the rest over before comparing lengths, so a stale long state never shadows a usable one.
        usable: ?*const fn (ptr: *anyopaque, saved: Saved, len: u32) bool = null,
        /// A verified decoded/complete short continuation, not a general relaxation of native eligibility.
        short_eligible: ?*const fn (ptr: *anyopaque, saved: Saved, len: u32) bool = null,
    };
};

// -- the durable tier (spill v2) --------------------------------------------------------------------------------------
// A Tier keeps kept states past the memory budget and past the process (dsv41's spill_tier glues core/spill's per-rank
// block store and the family's packing to it). The store calls it on the engine thread only and never waits on disk:
// `lookup` reads the tier's own index, a restore is a job polled between rounds and prompt chunks, a write is enqueued
// at keep and staged by `pump`. All-rank agreement (presence, validation, commit) is the family's, inside `begin` and
// `poll`: a job is ready only once every rank installed its copy, and failed everywhere when any rank's copy fails.
// families/dsv41/spill_tier.zig implements this vtable over core/spill's store and spill_state's packing; the wire
// commands (has, reserve, copied-block boundary, validate, commit/abort) ride rank_wire's ordered stream inside
// begin/poll/cancel, and native/cuda.zig openWith calls Store.setTier with the backend's tier, grid, window and
// Snapshots.usable.

/// A durable point the tier holds and found for a prompt (its ids are the store's keyed ids, pictures' keys included).
pub const Point = struct {
    /// The tier's name for the point while it lives (a digest of the point on disk).
    id: u64,
    /// Prompt tokens the state stands after, and the ids it covers (`at` plus the store's lookahead).
    at: u32,
    tokens: u32,
    /// Memory a restored copy takes in the store (0: the family's `bytes(at)`).
    bytes: u64 = 0,
    /// The conversation that wrote it and whether it is a shared prefix (persisted with the point).
    group: u64 = 0,
    shared: bool = false,
};

/// A durable lookup: the longest point whose ids are a prefix of `key`, `beyond < at <= max_at`, and on `grid` when
/// set (the native eligibility rule; the store checks the answer again).
pub const Want = struct {
    key: []const u32,
    beyond: u32,
    max_at: u32,
    grid: u32 = 0,
    lookahead: u32 = 0,
    /// The prompt holds pictures, and each one's identity (every picture, in prompt order; see `Media`). A point
    /// holding pictures matches only when every picture before its cut has the same span and payload digest; a prompt
    /// whose picture ids `media` does not account for (`mediaKeys`) finds nothing.
    images: bool = false,
    media: []const Media = &.{},
};

// -- pictures (recipe patch 0006's content keys; spill v2's disk identity) -------------------------------------------

/// A picture of a prompt as the durable tier names it: its positions [start, start + len) and the full SHA-256 of
/// its payload. Recipe 0006's per-position cache ids (`pictureId`) are shortened from the digest; disk identity takes
/// the whole digest and the span. (extern: the same bytes as core/spill's format.Media.)
pub const Media = extern struct {
    start: u32,
    len: u32,
    digest: [32]u8,

    pub fn eql(a: Media, b: Media) bool {
        return a.start == b.start and a.len == b.len and std.mem.eql(u8, &a.digest, &b.digest);
    }
};

/// TF_DS_IMAGE_CACHE's one reading (the server's content keys, the disk tier and the rank fingerprint alike): on unless
/// it is exactly "0".
pub fn imageCacheOn(v: ?[]const u8) bool {
    return !std.mem.eql(u8, v orelse return true, "0");
}

test imageCacheOn {
    try std.testing.expect(!imageCacheOn("0"));
    try std.testing.expect(imageCacheOn(null) and imageCacheOn("1") and imageCacheOn(""));
    try std.testing.expect(imageCacheOn(" 0") and imageCacheOn("0 ")); // (exact, as round 2 read it)
}

/// Cache ids at or above this are pictures' (`pictureId`); a vocabulary id is always below it.
pub const picture_ids: u32 = 0x8000_0000;

/// The prompt cache's id for position `j` of a picture whose payload's SHA-256 is `digest` (recipe 0006): above every
/// vocabulary id, so it never equals a text token, and the same for the same picture at any place in any prompt.
pub fn pictureId(digest: *const [32]u8, j: u32) u32 {
    const seed = std.mem.readInt(u64, digest[0..8], .little);
    return picture_ids | (@as(u32, @truncate(std.hash.Wyhash.hash(seed, std.mem.asBytes(&j)))) & 0x7fff_ffff);
}

/// Whether `media` names every picture in `key[0..n]` and nothing else: spans in order, not overlapping, not empty;
/// each span position below `n` holds `pictureId(digest, offset)`, every other position below `n` a vocabulary id.
/// Spans may run past `n` (a picture straddling a cut); spans starting at or past `n` are not looked at. The check
/// that binds the shortened per-position ids to the full digests a point records.
pub fn mediaKeys(key: []const u32, n: usize, media: []const Media) bool {
    if (n > key.len) return false;
    var at: usize = 0; // positions before it are checked
    var prev_end: u64 = 0;
    for (media) |m| {
        if (m.len == 0 or m.start < prev_end) return false;
        prev_end = @as(u64, m.start) + m.len;
        if (m.start >= n) continue;
        for (key[at..m.start]) |id| if (id >= picture_ids) return false;
        const stop: usize = @intCast(@min(prev_end, n));
        for (key[m.start..stop], 0..) |id, j| if (id != pictureId(&m.digest, @intCast(j))) return false;
        at = stop;
    }
    for (key[at..n]) |id| if (id >= picture_ids) return false;
    return true;
}

/// A kept state the tier writes (enqueued at keep). `saved` stays the family's: the tier reads it until `release`.
pub const Kept = struct {
    saved: Saved,
    /// Its ids (`at` plus lookahead), valid during the call only.
    key: []const u32,
    at: u32,
    bytes: u64,
    group: u64 = 0,
    shared: bool = false,
    images: bool = false,
    /// With pictures: each one's identity (every picture of the prompt, in order; the tier records those before `at`)
    /// and the prompt's model ids (the placeholder at a picture's positions; at least `at` of them), valid during the
    /// call. A picture before `at` that `media` does not name keeps nothing on disk.
    media: []const Media = &.{},
    raw: []const u32 = &.{},
    /// The point its rows' computation forks from (the state its pass resumed from; null: a fresh prefill, or a resumed
    /// state that was never written), and where.
    parent: ?u64 = null,
    fork: u32 = 0,
    /// The computation its rows belong to (0110's lineage; block names are salted with it), valid during the call:
    /// set whether or not the resumed state was ever written, so sibling forks name their shared blocks alike.
    lineage: []const Fork = &.{},
};

/// One step of a state's lineage: computation `name` produced the rows from `at` on (the first step's `at` is 0). A
/// pass that resumes a state at `at` forks that state's lineage there (GLM 0110 `fork`: the same names for the blocks
/// before `at`, its own from the block holding `at` on); a fresh prefill starts a lineage of its own.
pub const Fork = struct { name: u64, at: u32 };

/// A restore job after a poll.
pub const Poll = union(enum) {
    /// Still reading or copying: blocks copied so far of its total.
    pending: struct { done: u64 = 0, total: u64 = 0 },
    /// Installed on every rank: a family state the store adopts (it owns it from here; `bytes` of memory, 0: bytes(at)),
    /// and the point's lineage (valid during the call; the store copies it).
    ready: struct { saved: Saved, bytes: u64 = 0, lineage: []const Fork = &.{} },
    /// Aborted everywhere (absent on a rank, CRC/size, a guard): the tier released what it reserved.
    failed: anyerror,
};

/// The tier's own counters (core/spill fills them; totals unless marked a gauge).
pub const TierCounts = struct {
    points: u64 = 0, // gauge: published points
    blocks: u64 = 0, // gauge: unique blocks on disk
    stored_bytes: u64 = 0, // gauge: unique bytes on disk (blocks, point files, temporaries)
    cap_bytes: u64 = 0, // gauge: --spill-gib
    disk_free_bytes: u64 = 0, // gauge: free bytes on the store's file system
    staging_bytes: u64 = 0, // gauge: host staging held now (writes, reads, small states)
    staging_peak: u64 = 0, // gauge: its peak
    writes_pending: u64 = 0, // gauge: queued points not yet published
    writes_queued: u64 = 0,
    writes_committed: u64 = 0,
    writes_cancelled: u64 = 0, // unpublished points cancelled, for any of the reasons below
    writes_cancelled_guard: u64 = 0, // their source was about to change and staging could not finish in budget
    writes_cancelled_staging: u64 = 0, // the staging cap had no room
    writes_cancelled_shutdown: u64 = 0, // the flush deadline passed
    writes_failed: u64 = 0,
    forced_staging: u64 = 0, // ranges staged at once because their source was about to change
    no_space: u64 = 0, // writes refused by the cap or the free-disk floor
    dedup_blocks: u64 = 0,
    dedup_bytes: u64 = 0,
    written_bytes: u64 = 0,
    presence_refused: u64 = 0, // a rank lacked the point: no rank restored it
    compatible_misses: u64 = 0, // lookups with no compatible point
    crc_errors: u64 = 0,
    size_errors: u64 = 0, // truncated or oversized files, manifest bounds broken
    reads: u64 = 0,
    read_bytes: u64 = 0,
    read_failed: u64 = 0,
    evicted_turns: u64 = 0,
    evicted_forks: u64 = 0,
    evicted_conversations: u64 = 0,
    evicted_shared: u64 = 0,
    quota_trims: u64 = 0,
    recovered_points: u64 = 0, // published points found at start
    dropped_temporaries: u64 = 0, // incomplete records removed at start
    // (appended: the stats wire carries fields in order, and an older peer's reply zero-fills those it lacks)
    pictures_kept: u64 = 0, // points written with pictures before their cut (rank 0)
    pictures_refused: u64 = 0, // keeps and lookups with pictures turned away: ids and media disagree, or image cache off (rank 0)

    pub const gauges = [_][]const u8{ "points", "blocks", "stored_bytes", "cap_bytes", "disk_free_bytes", "staging_bytes", "staging_peak", "writes_pending" };

    /// Each field's /metrics help (a missing one fails the build).
    pub const help = .{
        .points = "Published points in this rank's disk tier.",
        .blocks = "Unique blocks on this rank's disk.",
        .stored_bytes = "Unique bytes the disk tier holds (blocks, point files, temporaries).",
        .cap_bytes = "The disk tier's cap (--spill-gib).",
        .disk_free_bytes = "Free bytes on the disk tier's file system.",
        .staging_bytes = "Host staging the disk tier holds now.",
        .staging_peak = "The most host staging the disk tier has held.",
        .writes_pending = "Kept states queued for the disk, not yet published.",
        .writes_queued = "Kept states queued for the disk.",
        .writes_committed = "Points published on disk (manifest last).",
        .writes_cancelled = "Unpublished points cancelled (a guard could not stage, staging full, shutdown).",
        .writes_cancelled_guard = "Unpublished points cancelled because their source was about to change.",
        .writes_cancelled_staging = "Unpublished points cancelled because staging had no room.",
        .writes_cancelled_shutdown = "Unpublished points cancelled at the flush deadline.",
        .writes_failed = "Disk writes that failed.",
        .forced_staging = "Ranges staged at once because their source was about to change.",
        .no_space = "Writes refused by the cap or the free-disk floor.",
        .dedup_blocks = "Blocks linked to equal bytes already on disk.",
        .dedup_bytes = "Bytes those links saved.",
        .written_bytes = "Bytes written to disk.",
        .presence_refused = "Restores refused because a rank lacked the point.",
        .compatible_misses = "Durable lookups with no compatible point.",
        .crc_errors = "Blocks or sections that failed their CRC.",
        .size_errors = "Files truncated or past their manifest's bounds.",
        .reads = "Points read for restores.",
        .read_bytes = "Bytes read for restores.",
        .read_failed = "Restore reads that failed.",
        .evicted_turns = "Old turns evicted from disk.",
        .evicted_forks = "Abandoned forks evicted from disk.",
        .evicted_conversations = "Whole conversations evicted from disk.",
        .evicted_shared = "Shared-prefix points evicted from disk.",
        .quota_trims = "Writes that trimmed their conversation to its quota.",
        .recovered_points = "Published points found at start.",
        .dropped_temporaries = "Incomplete records removed at start.",
        .pictures_kept = "Points written to disk with pictures before their cut.",
        .pictures_refused = "Disk keeps and lookups with pictures turned away (ids and media disagree, or image cache off).",
    };
};

/// The ranks a tier reports counters for (TP ranks; more are not reported).
pub const max_ranks = 8;

/// Each rank's TierCounts (rank 0 first): `n` of `ranks` are filled.
pub const RankCounts = struct {
    n: u32 = 0,
    ranks: [max_ranks]TierCounts = @splat(.{}),

    pub fn slice(r: *const RankCounts) []const TierCounts {
        return r.ranks[0..r.n];
    }
};

/// Lifetime and cost (the store relies on both): the tier outlives the Store (Store.deinit cancels restores and
/// releases written entries through it, so the backend deinitializes the tier after the store). `lookup` and
/// `counts` run on every admission and every round: O(1)-ish, from memory, no I/O and no CUDA. `poll`, `pump` and
/// `release` run between rounds, between a fill's prompt chunks (inside Stream.between, with any Engram read-ahead
/// already joined) and inside the lone driver's commit hook (`pump` only), so each must be bounded per call: by its
/// `budget_ns` and by a block cap of its own (GLM's PUMP_BLOCKS).
pub const Tier = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The longest durable point `want` allows, from the tier's own index (no disk I/O, no CUDA); null: none.
        lookup: *const fn (ptr: *anyopaque, want: *const Want) ?Point,
        /// Start restoring `point` into a new family state (reserve on every rank, then read): a job for `poll`.
        begin: *const fn (ptr: *anyopaque, point: *const Point) anyerror!u64,
        /// Progress the job by at most `budget_ns` of engine-thread copies; after ready or failed the job is gone.
        poll: *const fn (ptr: *anyopaque, job: u64, budget_ns: u64) Poll,
        /// Stop a pending job: reads stop, issued DMA is waited for, reservations go (called once per job at most).
        cancel: *const fn (ptr: *anyopaque, job: u64) void,
        /// Enqueue the write of a state just kept: the point's name (null: not written, e.g. under the minimum).
        keep: *const fn (ptr: *anyopaque, kept: *const Kept) ?u64,
        /// The point's memory twin is about to be dropped: stage what the write still reads from `saved`, or cancel it.
        release: ?*const fn (ptr: *anyopaque, point: u64, saved: Saved) void = null,
        /// The point became a shared prefix: persist the flag (atomically; never resurrecting a deleted point).
        share: ?*const fn (ptr: *anyopaque, point: u64) void = null,
        /// The point's state was resumed: its last use, for eviction order.
        touch: ?*const fn (ptr: *anyopaque, point: u64) void = null,
        /// The point is unusable (a peer failed it, its install failed): drop it.
        forget: *const fn (ptr: *anyopaque, point: u64) void,
        /// Stage queued writes for at most `budget_ns` (engine thread: between rounds and chunks, and at idle).
        pump: ?*const fn (ptr: *anyopaque, budget_ns: u64) void = null,
        /// Retire already-issued copies and collect completions only: no new stage/unpack, restore admission,
        /// or wire asks that grant copies. Kept separate: poll(0) may still launch work. Called even when decode
        /// is owed; must not wait for unfinished copies. Null: no independent maintenance required.
        maintain: ?*const fn (ptr: *anyopaque) void = null,
        /// Whether writes still wait for `pump` (the host keeps servicing while idle).
        busy: ?*const fn (ptr: *anyopaque) bool = null,
        /// Clean shutdown: stage and drain within `budget_ns`; returns the points left unpublished.
        flush: *const fn (ptr: *anyopaque, budget_ns: u64) u64,
        /// Each rank's counters into `out` (rank 0 first; the peers' as their last replies carried them): how many.
        counts: *const fn (ptr: *anyopaque, out: []TierCounts) usize,
    };
};

/// A restore in flight: one job per point, shared by every request waiting for it (0110's duplicate-load avoidance).
pub const Restore = struct {
    point: Point,
    job: u64,
    key: []u32, // the point's ids (store-owned)
    media: []const Media = &.{}, // full picture identities supplied by the request that found the point
    done: u64 = 0, // blocks copied so far of `total` (its last poll)
    total: u64 = 0,
    bytes: u64 = 0, // the memory its destination holds from begin (charged to the budget until it ends)
};

/// How a polled restore stands.
pub const Restored = enum { pending, adopted, dropped, failed };

/// The store's own counters for the tier and continuation (totals).
pub const Stats = struct {
    disk_lookups: u64 = 0,
    disk_hits: u64 = 0, // lookups that found a durable point longer than the best usable memory state
    disk_hits_committed: u64 = 0, // restores adopted as memory states (every rank agreed)
    restores_begun: u64 = 0,
    restores_shared: u64 = 0, // requests that joined a restore already in flight
    restores_failed: u64 = 0,
    restores_cancelled: u64 = 0,
    restores_dropped: u64 = 0, // adopted nothing: the budget cannot hold it, or memory had it meanwhile
    restores_refused: u64 = 0, // not begun: the budget could not hold its destination
    restored_tokens: u64 = 0,
    writes_enqueued: u64 = 0,
    shared_promoted: u64 = 0,
    flushes: u64 = 0,
    flush_timeouts: u64 = 0,
    flush_unpublished: u64 = 0,
    chats_continued: u64 = 0,
    chats_continued_prefill_tokens: u64 = 0,
};

// -- continuation (0111) ----------------------------------------------------------------------------------------------

/// A conversation's first reply: the tokens of it a later prompt must repeat to continue that conversation.
pub const chat_match: u32 = 32;
/// A continued conversation's prefill from this many tokens on is logged.
pub const chat_log_tokens: u32 = 4096;

/// The conversation a request's kept states belong to (Store.join).
pub const Join = struct {
    group: u64 = 0, // 0: grouping off
    /// It continues `group` without resuming that conversation's own turn state (0111's counted case).
    continued: bool = false,
    /// The matched state's ids (`at` plus lookahead), and where the prompt left its evidence (the last prompt that kept
    /// or resumed it): `back < prior_len` is a rewind before the state itself.
    prior_len: u32 = 0,
    back: u32 = 0,
    /// What the lookup resumed from.
    source: Source = .nothing,

    pub const Source = enum { nothing, shared, other };
};

/// What a family's states depend on beyond their position.
pub const Rules = struct {
    /// Tokens past the position a state read (Flash Next's MTP head keys row at-1 with token at: 1).
    lookahead: u32 = 0,
    /// Learn useful shared-prefix cuts on misses. Candidate backends enable this with their native grid/window.
    fork: bool = false,
    /// Only families saving both complete target state and prompt logits may opt in.
    exact_end: bool = false,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and keep only at the request's chunk starts.
    planned: bool = false,
    /// A mark other than the history's is kept only this far from every other mark and the resume point.
    min_gap: u32 = 256,
    /// Shorter prompts keep nothing: below it a mark's extra prompt call costs more than a later turn's reuse saves.
    min_prompt: u32 = 4096,
    /// The family's own resume grid (dsv41: its 2048-row chunk; 0: none): durable points and shared marks stand on it.
    grid: u32 = 0,
    /// The rows a resumed pass replays at its end (dsv41: the window): a state is usable only this far from the end.
    window: u32 = 0,

    /// The native rule (Lanes.usable): 0 < at < len, on the grid, at most len - window, its lookahead inside the prompt.
    pub fn eligible(r: Rules, at: u32, len: usize) bool {
        if (at == 0 or at >= len) return false;
        if (r.grid > 0 and at % r.grid != 0) return false;
        return @as(usize, at) + r.window <= len and @as(usize, at) + r.lookahead <= len;
    }

    /// The grid point at or before `at` (`at` itself without a grid).
    pub fn floorGrid(r: Rules, at: u32) u32 {
        return if (r.grid == 0) at else at / r.grid * r.grid;
    }

    /// The largest position `eligible` allows for a prompt of `len` tokens (null: none).
    pub fn maxAt(r: Rules, len: usize) ?u32 {
        if (len < 2) return null;
        var hi: usize = len - 1;
        hi = @min(hi, len -| @max(r.window, r.lookahead));
        const at = r.floorGrid(@intCast(@min(hi, std.math.maxInt(u32))));
        return if (r.eligible(at, len)) at else null;
    }
};

pub const Entry = struct {
    pins: usize = 0,
    retired: bool = false,
    namespace: Namespace = null,
    tokens: []u32, // the state's tokens and `lookahead` more
    at: u32,
    saved: Saved,
    bytes: u64,
    born: u32, // the length of the prompt that kept it: a later turn's is longer
    used: u64, // the store's clock at its last keep or resume
    last: []u32, // the last prompt that kept or resumed it
    shared: bool = false, // kept at a shared system block's cut: other conversations resume it, so its own never supersedes it
    group: u64 = 0, // the conversation that kept it (0111; 0: grouping off)
    point: ?u64 = null, // its durable point in the tier (written at keep, or restored from)
    images: bool = false, // its key holds pictures' keys
    media: []const Media = &.{}, // full identities of pictures before at, owned for kept/restored states
    restored: bool = false, // read back from the tier: not a turn's own state (it counts against no conversation)
    lineage: []Fork = &.{}, // the computation its rows belong to (store-owned; empty without a tier)
    x: x1.Entry = .{},
    reuse: Reuse = .{}, // K2a: CPU observation only; never changes real eviction scoring
};

/// Bounded CPU-only observations. A keep/dedup is NOT a successful restore.
/// Marginal tokens are a conservative lower bound, not measured time saved.
pub const Reuse = struct {
    restores: u64 = 0,
    last_restore: u64 = 0,
    restored_tokens: u64 = 0,
    marginal_lower_bound_tokens: u64 = 0,
    pending_marginal: u32 = 0,
    dedups: u32 = 0,
};

/// Candidate skips and removals have different denominators; never sum them as misses.
pub const Reasons = struct {
    image_bypass: u64 = 0, image_bypass_prompt_tokens: u64 = 0,
    lookup_candidate: u64 = 0, no_eligible_prefix_match: u64 = 0,
    restore_ok: u64 = 0, restore_failed: u64 = 0,
    dead_candidate_skips: u64 = 0,
    incompatible_candidate_skips: u64 = 0,
    snapshot_budget: u64 = 0,
    reset_entries: u64 = 0,
    physical_overlap: u64 = 0,
    placement_pressure: u64 = 0,
    physical_unknown: u64 = 0,
};

pub const Audit = struct {
    dead_at_reserve: u64 = 0,
    dead_at_keep: u64 = 0,
    valid_budget_evictions: u64 = 0,
    upgrades: u64 = 0,
    dedups: u64 = 0,
    restored_tokens: u64 = 0,
    marginal_lower_bound_tokens: u64 = 0,
    shadow_decisions: u64 = 0,
    shadow_disagreements: u64 = 0,
    shadow_eligible: u64 = 0,
};

pub const Counts = struct { hits: u64 = 0, misses: u64 = 0, kept: u64 = 0, evicted: u64 = 0, refused: u64 = 0, failed: u64 = 0 };

/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// The entry a backend restores itself (null: the pass starts at 0) and the pass's marks.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{}, request: RequestState = .{} };

/// A side-effect-free look at what a prompt would resume: the best usable memory state, and a longer durable point.
pub const Probe = struct { memory: ?*Entry = null, durable: ?Point = null };

/// What a keep records beyond the state: its conversation, pictures and lineage (the tier writes them with it).
pub const KeepInfo = struct {
    group: u64 = 0,
    images: bool = false,
    media: []const Media = &.{}, // with pictures: each one's identity and the prompt's model ids (Kept's, for the tier)
    raw: []const u32 = &.{},
    parent: ?u64 = null,
    fork: u32 = 0,
    lineage: []const Fork = &.{}, // the pass's (Store.passLineage), copied into the entry
};

pub const Store = struct {
    gpa: Allocator,
    family: Snapshots,
    rules: Rules,
    budget: u64,
    held: u64 = 0,
    clock: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,
    counts: Counts = .{},
    shared_keys: std.ArrayList(u64) = .empty, // the tokens of recently planned shared cuts, hashed (at most SHARED_KEYS)
    imprint: ?*imprint.Imprint = null, // --learn: shared cuts' states on disk, read back by later sessions and servers
    tier: ?Tier = null, // spill v2: kept states written at keep, read back by restore jobs (setTier)
    groups: bool = false, // 0111: kept states carry their conversation (setTier turns it on)
    next_group: u64 = 0,
    roots: std.AutoHashMapUnmanaged(u64, u32) = .empty, // each conversation's first prompt length (join)
    restoring: std.ArrayList(*Restore) = .empty,
    restoring_bytes: u64 = 0, // destinations the restores in flight reserved (charged like kept states)
    next_lineage: u64 = 0,
    stats: Stats = .{},

    audit: Audit = .{},
    reasons: Reasons = .{},
    diagnostics: bool = true,
    prune_dead: bool = true,
    batch_dead: bool = false,
    first_continuation: bool = false,
    victim_dead: bool = false,
    reuse_eviction: bool = false, // 126b: pressure-only; no timer, GPU work or extra state storage

    normalize_marks: bool = false,
    x: x1.State = .{},
    adaptive: bool = false,
    history: x3.History = .{},

    const SHARED_KEYS = 64;

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        var s: Store = .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
        if (!@import("builtin").is_test) {
            if (std.c.getenv("TF_K2A_DIAGNOSTICS")) |v| s.diagnostics = v[0] != '0';
            if (std.c.getenv("TF_K2A_PRUNE_DEAD")) |v| s.prune_dead = v[0] != '0';
            if (std.c.getenv("TF_CACHE_REUSE_EVICTION")) |v| s.reuse_eviction = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_CACHE_NATIVE_PLAN")) |v| s.normalize_marks = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_CACHE_BALANCED")) |v| s.x.enabled = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_CACHE_X1_TRACE")) |v| s.x.trace = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_DS_STREAMS")) |v| s.x.slots = std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch 12, 1, 64);
            if (std.c.getenv("TF_CACHE_ADAPTIVE_PREFIX")) |v| s.adaptive = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_CACHE_BATCH_DEAD")) |v| s.batch_dead = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_CACHE_FIRST_CONTINUATION")) |v| s.first_continuation = std.mem.eql(u8, std.mem.span(v), "1");
            if (s.adaptive) s.x.enabled = true;
            if (s.x.enabled) s.diagnostics = true;
            if (s.reuse_eviction) s.diagnostics = true; // real restore observations are required
        }
        return s;
    }

    pub fn deinit(s: *Store) void {
        while (s.restoring.items.len > 0) s.restoreCancel(s.restoring.items[s.restoring.items.len - 1]);
        s.restoring.deinit(s.gpa);
        for (s.entries.items) |e| s.free(e);
        s.entries.deinit(s.gpa);
        s.shared_keys.deinit(s.gpa);
        s.roots.deinit(s.gpa);
        s.held = 0;
    }

    /// Drop every reusable state on the owning engine thread, with no live
    /// borrowers. Preserve rules/budget and lifetime counters. Learned disk
    /// states can repopulate RAM, so refuse rather than claim a cold cache.
    pub fn clear(s: *Store) error{LearnedStatesEnabled}!void {
        if (s.imprint != null or s.tier != null) return error.LearnedStatesEnabled;
        const count = s.entries.items.len;
        if (s.diagnostics) s.reasons.reset_entries +|= count;
        s.x.last_reason = .reset;
        while (s.entries.items.len > 0) s.remove(s.entries.items.len - 1);
        s.x.ghosts = @splat(.{});
        s.history = .{};
        s.shared_keys.clearRetainingCapacity();
        s.counts.evicted += count;
        std.debug.assert(s.held == 0);
    }

    /// The durable tier from here (null: none); a tier turns continuation grouping on (its quotas count conversations).
    /// Conversation ids continue from `group_seed << 32` so they do not repeat a previous process's on disk.
    pub fn setTier(s: *Store, tier: ?Tier, group_seed: u32) void {
        s.tier = tier;
        if (tier != null) {
            s.groups = true;
            if (s.next_group == 0) s.next_group = @as(u64, group_seed) << 32;
            if (s.next_lineage == 0) s.next_lineage = @as(u64, group_seed) << 32;
        }
    }

    fn free(s: *Store, e: *Entry) void {
        if (e.point) |p| if (s.tier) |t| if (t.vtable.release) |f| f(t.ptr, p, e.saved); // a write still reading it stages first
        s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.free(e.lineage);
        if (e.media.len > 0) s.gpa.free(e.media);
        s.gpa.destroy(e);
    }

    fn remove(s: *Store, i: usize) void {
        const e = s.entries.orderedRemove(i);
        s.xRemoved(e);
        e.retired = true;
        if (e.pins == 0) s.releaseRetired(e);
    }

    /// Only unborrowed, already-dead entries in the existing keep scan.
    /// All host accounting remains charged until the synchronous all-rank receipt.
    fn removeDeadRun(s: *Store, start: usize) bool {
        if (!s.batch_dead) return false;
        const many = s.family.vtable.drop_many orelse return false;
        const is_live = s.family.vtable.live orelse return false;
        var saved: [16]Saved = undefined;
        var n: usize = 0;
        while (n < saved.len and start + n < s.entries.items.len) : (n += 1) {
            const e = s.entries.items[start + n];
            if (e.pins != 0 or is_live(s.family.ptr, e.saved)) break;
            saved[n] = e.saved;
        }
        if (n < 2) return false;
        const t0 = if (s.x.trace) x1.now() else 0;
        for (s.entries.items[start..][0..n]) |e| {
            s.x.last_reason = .dead;
            s.audit.dead_at_keep +|= 1;
            s.noteDead(e);
            s.xRemoved(e); // Ghost metadata is read before storage release.
            // Preserve the durable tier's capture custody before freeing any buffer.
            if (e.point) |p| if (s.tier) |t| if (t.vtable.release) |f| f(t.ptr, p, e.saved);
        }
        many(s.family.ptr, saved[0..n]);
        for (0..n) |_| {
            const e = s.entries.orderedRemove(start);
            s.held -= e.bytes;
            s.gpa.free(e.tokens);
            s.gpa.free(e.last);
            s.gpa.free(e.lineage);
            if (e.media.len > 0) s.gpa.free(e.media);
            s.gpa.destroy(e);
        }
        if (s.x.trace) s.x.drop_ns +|= x1.now() -| t0;
        return true;
    }

    /// Engine-thread scoped borrow. Retired entries remain charged until the final borrower releases.
    pub fn pin(_: *Store, e: *Entry) void {
        std.debug.assert(!e.retired);
        e.pins += 1;
    }

    pub fn unpin(s: *Store, e: *Entry) void {
        std.debug.assert(e.pins > 0);
        e.pins -= 1;
        if (e.pins == 0 and e.retired) s.releaseRetired(e);
    }

    fn releaseRetired(s: *Store, e: *Entry) void {
        s.held -= e.bytes;
        const t0 = if (s.x.trace) x1.now() else 0;
        s.free(e);
        if (s.x.trace) s.x.drop_ns +|= x1.now() -| t0;
    }

    /// The family's free storage kept for later saves.
    pub fn spare(s: *const Store) u64 {
        return if (s.family.vtable.spare) |f| f(s.family.ptr) else 0;
    }

    /// What the budget leaves past the kept states and the spare storage: a family readies storage only inside it.
    pub fn room(s: *const Store) u64 {
        return s.budget -| s.held -| s.restoring_bytes -| s.spare();
    }

    /// Room for `bytes` of new storage beside the kept states, the restores in flight and the spare storage: spare that
    /// is in the way trimmed first, then states evicted (never `skip`; `prompt`'s superseded ones first), trimming
    /// again after each since a dropped state's storage may come back spare. False: the budget cannot hold it.
    fn makeRoom(s: *Store, bytes: u64, skip: ?*Entry, prompt: []const u32) bool {
        // never evict for room that evicting everything evictable could not make
        if (s.restoring_bytes + bytes + (if (skip) |e| e.bytes else 0) > s.budget) return false;
        while (true) {
            const free_ = s.spare();
            if (s.held + s.restoring_bytes + free_ + bytes <= s.budget) return true;
            if (free_ > 0) if (s.family.vtable.trim) |f| {
                f(s.family.ptr, s.budget -| s.held -| s.restoring_bytes -| bytes);
                if (s.spare() < free_) continue;
            };
            s.remove(s.victimBut(if (skip) |e| e.namespace else null, skip, prompt) orelse return false);
            s.counts.evicted += 1;
        }
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or a chunk start for planned families.
    fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or std.mem.indexOfScalar(u32, starts, at) != null);
    }

    /// Whether a kept state still restores for a prompt of `len` tokens: the family's own rule, then the native one
    /// (both only where set: a family without them keeps the old behaviour).
    fn live(s: *const Store, e: *const Entry, len: usize) bool {
        if (s.family.vtable.usable) |f| if (!f(s.family.ptr, e.saved, @intCast(@min(len, std.math.maxInt(u32))))) return false;
        if (s.rules.grid == 0 and s.rules.window == 0) return true;
        if (s.rules.exact_end and e.at == len and s.family.vtable.compatible != null) return true;
        if (s.rules.eligible(e.at, len)) return true;
        // Only the native backend can attest that a completed decoder tail has
        // its short-window rows. A grid/window bypass without that proof stays forbidden.
        if (s.family.vtable.short_eligible) |f|
            return f(s.family.ptr, e.saved, @intCast(@min(len, std.math.maxInt(u32))));
        return false;
    }

    /// Predict a future keep without backend callbacks: no snapshot exists yet. The full scoped lookup at
    /// admission still checks residency, compatibility, and native eligibility; a false prediction only waits.
    pub fn wouldServe(s: *const Store, namespace: Namespace, prompt: []const u32, starts: []const u32, kept_namespace: Namespace, kept: []const u32, at: u32) bool {
        const n = @as(usize, at) + s.rules.lookahead;
        if (!std.meta.eql(namespace, kept_namespace) or at > prompt.len or (at == prompt.len and !s.rules.exact_end) or n > prompt.len or n > kept.len or !s.usable(at, starts)) return false;
        if (s.rules.grid != 0 or s.rules.window != 0) {
            const exact = s.rules.exact_end and at == prompt.len and s.family.vtable.compatible != null;
            if (!exact and !s.rules.eligible(at, prompt.len)) return false;
        }
        return std.mem.eql(u32, prompt[0..n], kept[0..n]);
    }

    /// The longest kept state `prompt` resumes exactly: its tokens a prefix of the prompt, at least one row left.
    pub fn find(s: *Store, prompt: []const u32, starts: []const u32) ?*Entry {
        return s.findScoped(null, prompt, starts);
    }

    pub fn findScoped(s: *Store, namespace: Namespace, prompt: []const u32, starts: []const u32) ?*Entry {
        return s.findMediaScoped(namespace, prompt, starts, &.{});
    }

    /// The short token key must agree with every full image digest before the resumed state, not just its ids.
    fn mediaBefore(media: []const Media, at: u32) []const Media {
        var n: usize = 0;
        while (n < media.len and media[n].start < at) n += 1;
        return media[0..n];
    }

    fn sameMedia(saved: []const Media, requested: []const Media, at: u32) bool {
        const prefix = mediaBefore(requested, at);
        if (saved.len != prefix.len) return false;
        for (saved, prefix) |a, b| if (!a.eql(b)) return false;
        return true;
    }

    pub fn findMediaScoped(s: *Store, namespace: Namespace, prompt: []const u32, starts: []const u32, media: []const Media) ?*Entry {
        var best: ?*Entry = null;
        var alternate: u32 = 0;
        for (s.entries.items) |e| {
            if (!std.meta.eql(e.namespace, namespace) or !sameMedia(e.media, media, e.at)) continue;
            if (e.tokens.len > prompt.len or e.at > prompt.len or (e.at == prompt.len and !s.rules.exact_end) or !s.usable(e.at, starts)) continue;
            if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) { if (s.diagnostics) s.reasons.dead_candidate_skips +|= 1; if (s.adaptive) s.xLost(e, .dead); continue; };
            if (s.family.vtable.compatible) |f| if (!f(s.family.ptr, e.saved, prompt.len)) { if (s.diagnostics) s.reasons.incompatible_candidate_skips +|= 1; continue; };
            if (best != null and e.at <= best.?.at) {
                // Original lookup skips comparing these tokens. They may match:
                // treating them as alternatives keeps the marginal bound conservative.
                if (s.diagnostics) alternate = @max(alternate, e.at);
                continue;
            }
            const n = e.tokens.len;
            if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
            if (s.diagnostics) if (best) |previous| {
                alternate = @max(alternate, previous.at);
            };
            if (!s.live(e, prompt.len)) continue; // stale or ineligible: never shadows a shorter usable state or a point
            best = e;
        }
        if (s.diagnostics) {
            if (best) |e| e.reuse.pending_marginal = e.at -| alternate;
            if (best != null) s.reasons.lookup_candidate +|= 1 else s.reasons.no_eligible_prefix_match +|= 1;
        }
        return best;
    }

    /// What `key` would resume, changing nothing in memory: the best usable kept state, and the tier's longer point.
    pub fn probe(s: *Store, key: []const u32, starts: []const u32) Probe {
        return s.probeMedia(key, starts, false, &.{});
    }

    /// `probe` for a prompt with pictures: `images` (it holds some) and each one's identity go to the tier's lookup
    /// (Want.images, Want.media), so a point is found only for the same pictures in the same places.
    pub fn probeMedia(s: *Store, key: []const u32, starts: []const u32, images: bool, media: []const Media) Probe {
        const m = s.findMediaScoped(null, key, starts, media);
        const t = s.tier orelse return .{ .memory = m };
        const have: u32 = if (m) |e| e.at else 0;
        s.stats.disk_lookups += 1;
        const max_at = s.rules.maxAt(key.len) orelse return .{ .memory = m };
        if (max_at <= have) return .{ .memory = m };
        const want: Want = .{ .key = key, .beyond = have, .max_at = max_at, .grid = s.rules.grid, .lookahead = s.rules.lookahead, .images = images, .media = media };
        const p = t.vtable.lookup(t.ptr, &want) orelse return .{ .memory = m };
        if (!s.pointFits(p, key, starts, have)) return .{ .memory = m };
        s.stats.disk_hits += 1;
        return .{ .memory = m, .durable = p };
    }

    /// A durable answer checked again: past the memory hit, eligible here, its ids inside the key, inside the budget.
    fn pointFits(s: *const Store, p: Point, key: []const u32, starts: []const u32, have: u32) bool {
        if (p.at <= have or @as(u64, p.at) + s.rules.lookahead != p.tokens or p.tokens > key.len) return false;
        if (!s.rules.eligible(p.at, key.len) or !s.usable(p.at, starts)) return false;
        return s.pointBytes(p) <= s.budget;
    }

    fn pointBytes(s: *const Store, p: Point) u64 {
        return if (p.bytes > 0) p.bytes else s.family.vtable.bytes(s.family.ptr, p.at);
    }

    /// A restore of `p` (found by `probe` for `key`): the job already in flight for it, or a new one. A new job's
    /// destination is charged to the budget from here until it ends (room made first, as for a keep; nobody holds a
    /// looked-up entry at admission, so any state may go): error.NoRoom when the budget cannot hold it.
    pub fn restoreBegin(s: *Store, p: Point, key: []const u32) !*Restore {
        return s.restoreBeginMedia(p, key, &.{});
    }

    pub fn restoreBeginMedia(s: *Store, p: Point, key: []const u32, media: []const Media) !*Restore {
        for (s.restoring.items) |r| if (r.point.id == p.id) {
            s.stats.restores_shared += 1;
            return r;
        };
        const t = s.tier orelse return error.NoTier;
        if (p.tokens > key.len) return error.BadPoint;
        const bytes = s.pointBytes(p);
        if (!s.makeRoom(bytes, null, key)) {
            s.stats.restores_refused += 1;
            return error.NoRoom;
        }
        const r = try s.gpa.create(Restore);
        errdefer s.gpa.destroy(r);
        const k = try s.gpa.dupe(u32, key[0..p.tokens]);
        errdefer s.gpa.free(k);
        const prefix = mediaBefore(media, p.at);
        const pictures: []const Media = if (prefix.len > 0) try s.gpa.dupe(Media, prefix) else &.{};
        errdefer if (pictures.len > 0) s.gpa.free(pictures);
        try s.restoring.ensureUnusedCapacity(s.gpa, 1);
        const job = try t.vtable.begin(t.ptr, &p);
        r.* = .{ .point = p, .job = job, .key = k, .media = pictures, .bytes = bytes };
        s.restoring.appendAssumeCapacity(r);
        s.restoring_bytes += bytes;
        s.stats.restores_begun += 1;
        return r;
    }

    /// Progress `r` by at most `budget_ns`; once it is no longer pending `r` is freed. An installed state becomes a
    /// shared or restored memory entry that the next lookup of its prompt finds (adopted).
    pub fn restorePoll(s: *Store, r: *Restore, budget_ns: u64) Restored {
        const t = s.tier.?;
        switch (t.vtable.poll(t.ptr, r.job, budget_ns)) {
            .pending => |p| {
                r.done = p.done;
                r.total = p.total;
                return .pending;
            },
            .failed => |err| {
                s.stats.restores_failed += 1;
                note("restoring {d} tokens from disk failed ({s}); its prompt prefills", .{ r.point.at, @errorName(err) });
                s.endRestore(r);
                return .failed;
            },
            .ready => |rd| {
                const ok = s.adopt(r, rd.saved, rd.bytes, rd.lineage);
                s.endRestore(r);
                return if (ok) .adopted else .dropped;
            },
        }
    }

    /// Stop `r` (its last waiter went, or shutdown): the tier releases its reservations; `r` is freed.
    pub fn restoreCancel(s: *Store, r: *Restore) void {
        if (s.tier) |t| t.vtable.cancel(t.ptr, r.job);
        s.stats.restores_cancelled += 1;
        s.endRestore(r);
    }

    fn endRestore(s: *Store, r: *Restore) void {
        if (std.mem.indexOfScalar(*Restore, s.restoring.items, r)) |i| {
            _ = s.restoring.orderedRemove(i);
            s.restoring_bytes -= r.bytes;
        }
        s.gpa.free(r.key);
        if (r.media.len > 0) s.gpa.free(r.media);
        s.gpa.destroy(r);
    }

    /// An installed restore as a memory entry, room made as for any keep; false: dropped (no room, or memory had it).
    /// Its reservation moves from the restores' charge to the kept states'. A memory twin of the same point that is
    /// no longer usable (the family staled it: KV sharding's placements give a kept state's rows away, an overwrite) is
    /// what made the lookup go to the tier: it is replaced, never a reason to drop the restore and recompute.
    fn adopt(s: *Store, r: *Restore, saved: Saved, bytes0: u64, lineage: []const Fork) bool {
        const bytes = if (bytes0 > 0) bytes0 else s.pointBytes(r.point);
        s.restoring_bytes -= r.bytes; // (zeroed: endRestore then releases nothing more)
        r.bytes = 0;
        for (s.entries.items, 0..) |e, i| if (e.at == r.point.at and sameMedia(e.media, r.media, e.at) and std.mem.eql(u32, e.tokens, r.key)) {
            if (s.live(e, r.key.len)) return s.unadopted(saved); // a usable copy came meanwhile: it wins
            s.remove(i); // the stale twin goes (its storage back to the family), the restore takes its place
            break;
        };
        if (!s.makeRoom(bytes, null, r.key)) return s.unadopted(saved);
        const e = s.gpa.create(Entry) catch return s.unadopted(saved);
        const tokens = s.gpa.dupe(u32, r.key) catch {
            s.gpa.destroy(e);
            return s.unadopted(saved);
        };
        const last = s.gpa.dupe(u32, r.key) catch {
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.unadopted(saved);
        };
        const lin = s.gpa.dupe(Fork, lineage) catch {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.unadopted(saved);
        };
        const pictures: []const Media = if (r.media.len > 0) s.gpa.dupe(Media, r.media) catch {
            s.gpa.free(lin);
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.unadopted(saved);
        } else &.{};
        s.clock += 1;
        e.* = .{ .tokens = tokens, .at = r.point.at, .saved = saved, .bytes = bytes, .born = @intCast(r.key.len), .used = s.clock, .last = last, .shared = r.point.shared, .group = r.point.group, .point = r.point.id, .images = pictures.len > 0, .media = pictures, .restored = true, .lineage = lin };
        s.entries.append(s.gpa, e) catch {
            e.point = null; // the point stays the tier's; only the memory copy goes
            s.free(e);
            s.stats.restores_dropped += 1;
            return false;
        };
        s.held += bytes;
        if (s.x.enabled or s.x.trace) {
            // A disk copy has no in-memory X1/X3 identity. Seed the same
            // terminal role, identity and history a fresh kept end receives;
            // only a real resumedRequest may grant it continuation credit.
            s.x.request_serial +|= 1;
            s.x.serial +|= 1;
            const origin = s.x.request_serial;
            const terminal = s.rules.exact_end and @as(usize, e.at) == r.key.len;
            e.x = .{ .id = s.x.serial, .key = sharedKey(null, e.tokens),
                .cohort = origin, .created_request = origin,
                .role = if (terminal) .prompt_end else .auxiliary };
            if (s.adaptive and terminal and !e.shared) {
                const namespace_id = sharedKey(e.namespace, &.{});
                e.x.history = s.history.begin(namespace_id, origin);
                e.x.history_request = origin;
            }
            s.adaptiveHint(e);
        }
        s.stats.disk_hits_committed += 1;
        s.stats.restored_tokens += r.point.at;
        if (!@import("builtin").is_test) std.log.info("prompt cache: restored {d} tokens from disk", .{r.point.at});
        return true;
    }

    fn unadopted(s: *Store, saved: Saved) bool {
        s.family.vtable.drop(s.family.ptr, saved);
        s.stats.restores_dropped += 1;
        return false;
    }

    /// Writes staged for at most `budget_ns` (engine thread).
    pub fn pump(s: *Store, budget_ns: u64) void {
        const t = s.tier orelse return;
        if (t.vtable.pump) |f| f(t.ptr, budget_ns);
    }

    /// Completion-only maintenance; never substitute poll(0) or pump(0) for an absent callback.
    pub fn maintain(s: *Store) void {
        const t = s.tier orelse return;
        if (t.vtable.maintain) |f| f(t.ptr);
    }

    /// Whether the tier has restores in flight or writes to stage (the host keeps servicing it).
    pub fn busy(s: *const Store) bool {
        const t = s.tier orelse return false;
        if (s.restoring.items.len > 0) return true;
        return if (t.vtable.busy) |f| f(t.ptr) else false;
    }

    /// Clean shutdown, engine thread: every restore stopped, writes drained within `budget_ns`; the points left
    /// unpublished at the deadline (counted, never published to make the count).
    pub fn flush(s: *Store, budget_ns: u64) u64 {
        const t = s.tier orelse return 0;
        while (s.restoring.items.len > 0) s.restoreCancel(s.restoring.items[s.restoring.items.len - 1]);
        const left = t.vtable.flush(t.ptr, budget_ns);
        s.stats.flushes += 1;
        if (left > 0) {
            s.stats.flush_timeouts += 1;
            s.stats.flush_unpublished += left;
            note("{d} kept states were still unpublished at the flush deadline; a restart misses them", .{left});
        }
        return left;
    }

    /// Durable point `id` is unusable (the family's all-rank abort, or its install failed): the tier drops it.
    pub fn forgetPoint(s: *Store, id: u64) void {
        const t = s.tier orelse return;
        for (s.entries.items) |e| if (e.point == id) {
            e.point = null;
        };
        t.vtable.forget(t.ptr, id);
    }

    /// Each rank's tier counters, when a tier is set.
    pub fn tierCounts(s: *const Store) ?RankCounts {
        const t = s.tier orelse return null;
        var out: RankCounts = .{};
        out.n = @intCast(@min(t.vtable.counts(t.ptr, &out.ranks), max_ranks));
        return out;
    }

    /// The lineage of a pass that resumes `from` (null: a fresh prefill) and computes its own rows from there: `from`'s
    /// forked at its position, or a new one (gpa-owned; empty without a tier, whose block names are its only use).
    pub fn passLineage(s: *Store, from: ?*const Entry) ![]Fork {
        if (s.tier == null) return &.{};
        s.next_lineage += 1;
        const own: Fork = .{ .name = s.next_lineage, .at = if (from) |e| e.at else 0 };
        const base: []const Fork = if (from) |e| e.lineage else &.{};
        if (base.len == 0) return s.gpa.dupe(Fork, &.{.{ .name = own.name, .at = 0 }});
        const out = try s.gpa.alloc(Fork, base.len + 1);
        @memcpy(out[0..base.len], base);
        out[base.len] = own;
        return out;
    }

    /// Before a prompt pass: restore the longest state `prompt` resumes (from 0: none or a failed copy) and plan its marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque) !Plan {
        const l = try s.lookup(a, prompt, history_len, shared, starts);
        const e = l.entry orelse return .{ .marks = l.marks };
        const ok = if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| true else |err| blk: {
            note("restoring {d} tokens failed ({s}); prefilling from the start", .{ e.at, @errorName(err) });
            break :blk false;
        };
        const from = if (ok) e.at else 0;
        s.resumed(e, prompt, ok);
        s.reserve(null, prompt, null, l.marks); // restored: the state it resumed may go too (a peer restores before it drops)
        return .{ .from = from, .marks = l.marks };
    }

    /// begin without the restore, for backends that restore inside their own prompt pass and then call `resumed`.
    pub fn lookup(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32) !Lookup {
        return s.lookupScoped(null, a, prompt, history_len, shared, starts);
    }

    pub fn lookupScoped(s: *Store, namespace: Namespace, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32) !Lookup {
        return s.lookupMediaScoped(namespace, a, prompt, history_len, shared, starts, false, &.{});
    }

    /// Image requests have no terminal snapshot in the native lane; account
    /// only for checkpoints the lane can actually keep (images). A state is reused only when the full identities of
    /// the pictures before its position match the request's (media; empty for a text request).
    pub fn lookupMediaScoped(s: *Store, namespace: Namespace, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, images: bool, media: []const Media) !Lookup {
        const have = s.findMediaScoped(namespace, prompt, starts, media);
        // Learned files use token-only keys. Salted requests neither read nor
        // write those files, preventing cross-namespace reuse after restart.
        const e = if (namespace == null) s.recall(prompt, starts, have, media) else have;
        if (e == null) s.counts.misses += 1;
        s.x.request_serial +|= 1;
        // Fork cuts use namespace-scoped, physically compatible evidence; the
        // upstream mark normalizer must not replace that guarded search.
        const raw = try s.marksScoped(namespace, a, prompt, if (e) |x| x.at else 0,
            history_len, shared, starts, if (e) |x| x.last else &.{});
        const marks_ = try s.fitting(a, try s.actualMarks(a, prompt, e, raw));
        errdefer a.free(marks_); // the caller owns marks_ only on success
        var automatic: [2]u32 = undefined;
        const auto_n = if (s.family.vtable.automatic_marks) |f| f(s.family.ptr, if (e) |x| x.saved else null, prompt.len, &automatic) else 0;
        var request: RequestState = .{ .id = s.x.request_serial, .cohort = s.x.request_serial,
            .namespace_id = sharedKey(namespace, &.{}), .input_len = @intCast(prompt.len),
            .encoder_at = if (auto_n == 2) automatic[0] else 0 };
        if (s.adaptive) request.history = s.history.begin(request.namespace_id, request.id);
        s.x.push(.{ .kind = .request, .request = request.id, .cohort = request.cohort,
            .namespace_id = request.namespace_id, .prompt = request.input_len,
            .at = if (e) |x| x.at else 0, .state = if (e) |x| x.x.id else 0 });
        s.xGhostReturns(namespace, prompt, if (e) |x| x.at else 0, &request);
        for (shared) |w| {
            const raw_at = if (s.rules.planned) floorStart(starts, w) else w;
            const at = if (s.normalize_marks and s.family.vtable.map_mark != null)
                s.family.vtable.map_mark.?(s.family.ptr, if (e) |x| x.saved else null, prompt.len, raw_at)
            else if (s.rules.planned) raw_at else s.rules.floorGrid(w);
            const planned_mark = if (s.normalize_marks and s.family.vtable.map_mark != null) at else raw_at;
            if (at > 0 and @as(usize, at) + s.rules.lookahead <= prompt.len and
                std.mem.indexOfScalar(u32, marks_, planned_mark) != null)
                s.noteShared(namespace, prompt[0 .. at + s.rules.lookahead]);
        }
        if (s.normalize_marks) {
            const reservation = try s.reservationMarks(a, prompt, e, marks_, images);
            defer a.free(reservation);
            s.reserve(namespace, prompt, e, reservation);
        } else s.reserve(namespace, prompt, e, marks_);
        return .{ .entry = e, .marks = marks_, .request = request };
    }

    /// Room for every state the pass keeps, made before it: a peer that mirrors the keeps hears of these evictions with the request.
    fn reserve(s: *Store, namespace: Namespace, prompt: []const u32, from: ?*Entry, marks_: []const u32) void {
        var need: u64 = 0;
        for (marks_, 0..) |at, k| {
            if (s.normalize_marks and std.mem.indexOfScalar(u32, marks_[0..k], at) != null) continue;
            var bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (s.normalize_marks) for (s.entries.items) |e| {
                if (!std.meta.eql(e.namespace, namespace) or e.at != at or e.tokens.len > prompt.len or !std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) continue;
                if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) { continue; };
                const compatible = if (s.family.vtable.compatible) |f| f(s.family.ptr, e.saved, prompt.len) else true;
                // Only a terminal request upgrades an incompatible encoder snapshot.
                // A kept key otherwise deduplicates exactly as keepScoped does.
                if (compatible or !(s.rules.exact_end and at == prompt.len)) {
                    bytes = 0;
                    s.x.compatible_reservations +|= 1;
                } else bytes -|= e.bytes;
                break;
            };
            need +|= bytes;
        }
        while (s.held + s.restoring_bytes + need > s.budget) {
            if (!s.evictBudget(namespace, from, prompt)) return;
        }
    }

    /// A kept state a peer could not resume goes, so later prompts do not ask for it again.
    pub fn forget(s: *Store, prompt: []const u32, at: u32) void {
        for (s.entries.items, 0..) |e, i| if (e.namespace == null and e.at == at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) {
            s.counts.failed += 1;
            if (s.diagnostics) s.reasons.restore_failed +|= 1;
            s.x.last_reason = .restore_failed;
            return s.remove(i);
        };
    }

    fn sharedKey(namespace: Namespace, tokens: []const u32) u64 {
        var hash = std.hash.Wyhash.init(0);
        if (namespace) |digest| hash.update(&digest);
        hash.update(std.mem.sliceAsBytes(tokens));
        return hash.final();
    }

    fn noteShared(s: *Store, namespace: Namespace, tokens: []const u32) void {
        const k = sharedKey(namespace, tokens);
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, k) != null) return;
        if (s.shared_keys.items.len == SHARED_KEYS) _ = s.shared_keys.orderedRemove(0);
        s.shared_keys.append(s.gpa, k) catch {};
    }

    /// The marks whose state the budget can hold (a new slice in `a`); the rest are refused now, so no pass, nor a peer, copies them.
    fn fitting(s: *Store, a: Allocator, marks_: []const u32) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        for (marks_) |at| {
            const bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (bytes <= s.budget) try out.append(a, at) else {
                s.counts.refused += 1;
                note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            }
        }
        return out.toOwnedSlice(a);
    }

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (dropped: the pass ran from 0).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        const i = std.mem.indexOfScalar(*Entry, s.entries.items, e) orelse return;
        if (!ok) {
            s.counts.failed += 1;
            if (s.diagnostics) s.reasons.restore_failed +|= 1;
            s.x.last_reason = .restore_failed;
            return s.remove(i);
        }
        if (s.diagnostics) s.reasons.restore_ok +|= 1;
        s.clock += 1;
        e.used = s.clock;
        s.counts.hits += 1;
        if (s.diagnostics) {
            e.reuse.restores +|= 1;
            e.reuse.last_restore = s.clock;
            e.reuse.restored_tokens +|= e.at;
            e.reuse.marginal_lower_bound_tokens +|= e.reuse.pending_marginal;
            s.audit.restored_tokens +|= e.at;
            s.audit.marginal_lower_bound_tokens +|= e.reuse.pending_marginal;
            e.reuse.pending_marginal = 0;
        }
        if (e.point) |p| if (s.tier) |t| if (t.vtable.touch) |f| f(t.ptr, p);
        const last = s.gpa.dupe(u32, prompt) catch return;
        s.gpa.free(e.last);
        e.last = last;
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        return s.marksScoped(null, a, prompt, from, history_len, shared, starts, previous);
    }

    fn marksScoped(s: *const Store, namespace: Namespace, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        if (prompt.len < s.rules.min_prompt) return &.{};
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        try want.append(a, history_len);
        var stable_at: u32 = 0;
        if (previous.len > 0) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (s.rules.fork) {
                stable_at = s.forkCut(forks.cut(previous, prompt, s.rules.lookahead, history_len), prompt.len, previous.len, starts);
                if (stable_at > 0) try want.append(a, stable_at);
            } else if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        // Evidence is inspected before reserve can evict it. No extra state copies outside the existing budget.
        var fork_at: u32 = 0;
        if (s.rules.fork and previous.len == 0) for (s.entries.items) |e| {
            if (!std.meta.eql(e.namespace, namespace) or e.images or e.last.len == 0) continue;
            if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) continue;
            if (s.family.vtable.compatible) |f| if (!f(s.family.ptr, e.saved, e.born)) continue;
            if (!s.live(e, e.born)) continue;
            const cut = forks.cut(e.last, prompt, s.rules.lookahead, history_len);
            const at = s.forkCut(cut, prompt.len, e.last.len, starts);
            fork_at = @max(fork_at, at);
        };
        const fork_index = want.items.len;
        if (fork_at > from) try want.append(a, fork_at);
        try want.appendSlice(a, shared);
        for (want.items, 0..) |w, k| {
            const at = if (s.rules.planned) floorStart(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at >= prompt.len or !s.usable(at, starts)) continue;
            if (k > 0 and !(fork_at > from and k == fork_index) and !(s.rules.fork and stable_at > 0 and at == stable_at)) { // the history's mark always; the others only away from it, each other and the resume point
                if (at - from < s.rules.min_gap) continue;
                const near = for (out.items) |o| {
                    if (@max(o, at) - @min(o, at) < s.rules.min_gap) break true;
                } else false;
                if (near) continue;
            }
            if (std.mem.indexOfScalar(u32, out.items, at) == null) try out.append(a, at);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// A fork must be restorable on both sides, not merely a spill grid point. Planned kernels additionally
    /// require an actual chunk start; flooring to a start and then a grid could produce an invalid start.
    fn forkCut(s: *const Store, cut: u32, len: usize, prior_len: usize, starts: []const u32) u32 {
        if (cut == 0) return 0;
        const hi = @min(cut, @min(s.rules.maxAt(len) orelse return 0, s.rules.maxAt(prior_len) orelse return 0));
        if (!s.rules.planned) return s.rules.floorGrid(hi);
        var best: u32 = 0;
        for (starts) |at| if (at <= hi and at > best and s.rules.eligible(at, len) and s.rules.eligible(at, prior_len)) {
            best = at;
        };
        return best;
    }

    /// The prompt pass stands at `at`: keep its state for `prompt`, evicting to fit; refused (counted) past the budget.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32) bool {
        return s.keepWithScoped(null, prompt, at, owner, starts, .{});
    }

    pub fn keepScoped(s: *Store, namespace: Namespace, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32) bool {
        return s.keepWithRequestScoped(namespace, prompt, at, owner, starts, .{}, null);
    }

    pub fn keepWith(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, info: KeepInfo) bool {
        return s.keepWithScoped(null, prompt, at, owner, starts, info);
    }

    pub fn keepWithScoped(s: *Store, namespace: Namespace, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, info: KeepInfo) bool {
        return s.keepWithRequestScoped(namespace, prompt, at, owner, starts, info, null);
    }

    pub fn keepRequestScoped(s: *Store, namespace: Namespace, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, request: ?*const RequestState) bool {
        return s.keepWithRequestScoped(namespace, prompt, at, owner, starts, .{}, request);
    }

    pub fn keepWithRequestScoped(s: *Store, namespace: Namespace, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, info: KeepInfo, request: ?*const RequestState) bool {
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len) return false;
        s.clock += 1;
        const key = sharedKey(namespace, prompt[0..n]);
        var shared = std.mem.indexOfScalar(u64, s.shared_keys.items, key) != null;
        // A stale physical reservation must be replaced, never refreshed by token-key deduplication.
        var old_i: usize = 0;
        while (old_i < s.entries.items.len) {
            const old = s.entries.items[old_i];
            if (s.family.vtable.live) |f| if (!f(s.family.ptr, old.saved)) {
                if (s.removeDeadRun(old_i)) continue;
                s.x.last_reason = .dead;
                s.audit.dead_at_keep +|= 1;
                s.noteDead(old);
                s.remove(old_i);
                continue;
            };
            // Encoder-only and complete terminal snapshots can share a token key.
            // Upgrade only at the exact request end; never downgrade a complete state
            // when another longer request keeps an encoder checkpoint at that key.
            if (s.rules.exact_end and at == prompt.len and std.meta.eql(old.namespace, namespace) and old.at == at and sameMedia(old.media, info.media, at) and std.mem.eql(u32, old.tokens, prompt[0..n])) {
                if (s.family.vtable.compatible) |f| if (!f(s.family.ptr, old.saved, prompt.len)) {
                    s.x.last_reason = .upgrade;
                    s.audit.upgrades +|= 1;
                    shared = shared or old.shared;
                    s.remove(old_i);
                    continue;
                };
            }
            old_i += 1;
        }
        for (s.entries.items) |e| if (std.meta.eql(e.namespace, namespace) and e.at == at and sameMedia(e.media, info.media, at) and std.mem.eql(u32, e.tokens, prompt[0..n])) {
            if (s.diagnostics) {
                e.reuse.dedups +|= 1;
                s.audit.dedups +|= 1;
            }
            e.used = s.clock; // the same state again: no copy
            if (shared and !e.shared) if (e.point) |p| if (s.tier) |t| if (t.vtable.share) |f| { // now a shared prefix, on disk too
                f(t.ptr, p);
                s.stats.shared_promoted += 1;
            };
            e.shared = e.shared or shared;
            if (e.group == 0) e.group = info.group;
            s.xDedup(e, request);
            return true;
        };
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes > s.budget) {
            s.counts.refused += 1;
            note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            return false;
        }
        while (s.entries.items.len > 0) { // room: spare storage this save can take, else new storage inside the budget
            if (s.family.vtable.reuses) |f| if (f(s.family.ptr, at)) break;
            const free_ = s.spare();
            if (s.held + s.restoring_bytes + free_ + bytes <= s.budget) break;
            if (free_ > 0) {
                if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| s.restoring_bytes -| bytes); // spare that cannot take this state goes before any state
                if (s.spare() < free_) continue;
            }
            if (!s.evictBudget(namespace, null, prompt)) break; // storage may return spare

        }
        if (s.entries.items.len == 0) if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| s.restoring_bytes -| bytes);
        if (bytes > s.budget -| s.held -| s.restoring_bytes -| s.spare()) { // pinned and retired states remain charged even without restores
            const reuses = if (s.family.vtable.reuses) |f| f(s.family.ptr, at) else false;
            if (!reuses) {
                s.counts.refused += 1;
                note("kept nothing at {d} tokens: held storage leaves insufficient room in the {d} MiB budget", .{ at, s.budget >> 20 });
                return false;
            }
        }
        const e = s.gpa.create(Entry) catch return s.fail(at, error.OutOfMemory);
        const tokens = s.gpa.dupe(u32, prompt[0..n]) catch {
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const last = s.gpa.dupe(u32, prompt) catch {
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const save_t0 = if (s.x.trace) x1.now() else 0;
        const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, err);
        };
        const lin = s.gpa.dupe(Fork, info.lineage) catch {
            s.family.vtable.drop(s.family.ptr, saved);
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        if (s.x.trace) s.x.save_ns +|= x1.now() -| save_t0;
        const prefix = mediaBefore(info.media, at);
        const pictures: []const Media = if (prefix.len > 0) s.gpa.dupe(Media, prefix) catch {
            s.gpa.free(lin);
            s.family.vtable.drop(s.family.ptr, saved);
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        } else &.{};
        const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
        e.* = .{ .namespace = namespace, .tokens = tokens, .at = at, .saved = saved, .bytes = charged, .born = @intCast(prompt.len), .used = s.clock, .last = last, .shared = shared, .group = info.group, .images = info.images, .media = pictures, .lineage = lin };
        s.entries.append(s.gpa, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
        s.xSaved(e, key, request);
        s.held += charged;
        s.counts.kept += 1;
        if (s.tier) |t| { // write at keep: the tier stages the state's bytes between rounds
            e.point = t.vtable.keep(t.ptr, &.{ .saved = saved, .key = tokens, .at = at, .bytes = charged, .group = info.group, .shared = shared, .images = info.images, .media = info.media, .raw = info.raw, .parent = info.parent, .fork = info.fork, .lineage = lin });
            if (e.point != null) s.stats.writes_enqueued += 1;
        }
        if (shared and !e.images) s.learn(e, starts); // learned files have token-only keys, never picture states
        if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| s.restoring_bytes); // spare storage only inside what the budget leaves
        return true;
    }

    /// A shared cut's state written to disk once (--learn): later sessions read it back, after a restart too.
    fn learn(s: *Store, e: *const Entry, starts: []const u32) void {
        if (e.namespace != null) return;
        const im = s.imprint orelse return;
        const write = s.family.vtable.write orelse return;
        const key = imprint.Imprint.keyOf(e.tokens);
        if (im.has(key) or e.bytes > im.cap) return;
        while (!im.fits(e.bytes)) s.unlearn(im, im.victim() orelse return); // the least recently used go first
        write(s.family.ptr, e.saved, im.dir, key) catch |err| return note("learning {d} tokens failed ({s})", .{ e.at, @errorName(err) });
        im.add(key, e.at, e.tokens, starts, e.bytes) catch |err| return note("learning {d} tokens failed ({s})", .{ e.at, @errorName(err) });
        if (!@import("builtin").is_test) std.log.info("prompt cache: learned {d} tokens to disk", .{e.at});
    }

    /// Learned state `key` forgotten: its files (the family's), then its index record.
    fn unlearn(s: *Store, im: *imprint.Imprint, key: u64) void {
        if (s.family.vtable.forget) |f| f(s.family.ptr, im.dir, key);
        im.remove(key) catch |err| note("forgetting a learned state failed ({s})", .{@errorName(err)});
    }

    /// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
    fn recall(s: *Store, prompt: []const u32, starts: []const u32, have: ?*Entry, media: []const Media) ?*Entry {
        const im = s.imprint orelse return have;
        const read = s.family.vtable.read orelse return have;
        const m = im.best(prompt, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
        if (mediaBefore(media, m.at).len > 0) return have; // token-only learned files cannot prove picture identity
        const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
        if (!s.makeRoom(bytes, have, prompt)) return have;
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = @intCast(prompt.len), .used = s.clock, .last = &.{}, .shared = true };
        e.tokens = s.gpa.dupe(u32, m.tokens) catch return s.drop(e, have, false);
        e.last = s.gpa.dupe(u32, prompt) catch return s.drop(e, have, false);
        const key = m.key;
        e.saved = read(s.family.ptr, im.dir, key, m.at) catch |err| {
            note("reading the learned {d} tokens failed ({s}); forgotten, a later pass learns them again", .{ e.at, @errorName(err) });
            s.unlearn(im, key);
            return s.drop(e, have, false);
        };
        im.touch(key);
        s.entries.append(s.gpa, e) catch return s.drop(e, have, true);
        s.held += bytes;
        if (!@import("builtin").is_test) std.log.info("prompt cache: read {d} learned tokens from disk", .{m.at});
        return e;
    }

    /// A half-made recalled entry undone (`saved` once read); the lookup goes on with `have`.
    fn drop(s: *Store, e: *Entry, have: ?*Entry, saved: bool) ?*Entry {
        if (saved) s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
        return have;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the store holds.
    pub fn report(s: *Store, prompt: usize, from: u32, kept: u64) void {
        s.flushX1();
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed });
        if (s.diagnostics) std.log.info("prompt cache k2a: dead_pressure {d}, dead_keep {d}, valid_budget {d}, upgrades {d}, dedups {d}, restored_tokens {d}, marginal_lower_bound_tokens {d}, shadow_decisions {d}, shadow_disagreements {d}, shadow_eligible {d}; image_bypass {d}, image_tokens {d}, lookup_candidate {d}, no_eligible_match {d}, dead_skips {d}, incompatible_skips {d}, restore_ok {d}, restore_failed {d}, physical_overlap {d}, placement_pressure {d}, physical_unknown {d}, snapshot_budget {d}, reset_entries {d}", .{ s.audit.dead_at_reserve, s.audit.dead_at_keep, s.audit.valid_budget_evictions, s.audit.upgrades, s.audit.dedups, s.audit.restored_tokens, s.audit.marginal_lower_bound_tokens, s.audit.shadow_decisions, s.audit.shadow_disagreements, s.audit.shadow_eligible, s.reasons.image_bypass, s.reasons.image_bypass_prompt_tokens, s.reasons.lookup_candidate, s.reasons.no_eligible_prefix_match, s.reasons.dead_candidate_skips, s.reasons.incompatible_candidate_skips, s.reasons.restore_ok, s.reasons.restore_failed, s.reasons.physical_overlap, s.reasons.placement_pressure, s.reasons.physical_unknown, s.reasons.snapshot_budget, s.reasons.reset_entries });
    }

    fn fail(s: *Store, at: u32, err: anyerror) bool {
        defer s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return false;
    }

    /// The entry to free first, never `skip` (a pass's resume): one a later prompt extends (never a shared cut), oldest first; else the oldest.
    fn victimBut(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) ?usize {
        s.x.last_reason = .none;
        s.x.last_baseline = 0;
        if (s.x.enabled) return s.xVictim(namespace, skip, prompt);
        return s.legacyVictimBut(namespace, skip, prompt);
    }

    fn legacyVictimBut(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) ?usize {
        s.victim_dead = false;
        var best: ?usize = null;
        var shadow: ?usize = null;
        var eligible: u64 = 0;
        // Integrate physical liveness into the existing pressure-only pass.
        // Never drop the borrowed resume entry; no new scan on the decode path.
        for (s.entries.items, 0..) |e, i| {
            if (e == skip or e.pins > 0) continue;
            if (s.prune_dead) if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) {
                s.victim_dead = true;
                return i;
            };
            if (e.shared) continue;
            const now = std.meta.eql(e.namespace, namespace) and prompt.len > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens);
            const moved_on = now or for (s.entries.items) |o| {
                if (o != e and std.meta.eql(o.namespace, e.namespace) and o.born > e.born and o.tokens.len > e.tokens.len and std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) break true;
            } else false;
            if (moved_on) {
                if (best == null or e.used < s.entries.items[best.?].used) best = i;
                if (s.diagnostics and s.prune_dead) {
                    eligible += 1;
                    if (shadow == null or s.shadowLess(e, s.entries.items[shadow.?])) shadow = i;
                }
            }
        }
        if (best) |i| { s.observeShadow(i, shadow, eligible); return if (s.reuse_eviction and shadow != null) shadow.? else i; }
        var oldest: ?usize = null;
        shadow = null;
        eligible = 0;
        for (s.entries.items, 0..) |e, i| if (e != skip and e.pins == 0) {
            if (oldest == null or e.used < s.entries.items[oldest.?].used) oldest = i;
            if (s.diagnostics and s.prune_dead) {
                eligible += 1;
                if (shadow == null or s.shadowLess(e, s.entries.items[shadow.?])) shadow = i;
            }
        };
        if (oldest) |i| s.observeShadow(i, shadow, eligible);
        return if (s.reuse_eviction and shadow != null) shadow else oldest;
    }

    fn evictBudget(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) bool {
        const i = s.victimBut(namespace, skip, prompt) orelse return false;
        if (s.victim_dead) s.audit.dead_at_reserve +|= 1 else s.audit.valid_budget_evictions +|= 1;
        if (s.victim_dead) s.noteDead(s.entries.items[i]) else if (s.diagnostics) { s.reasons.snapshot_budget +|= 1; }
        if (s.x.last_reason == .none) s.x.last_reason = if (s.victim_dead) .dead else .legacy_fallback;
        s.remove(i); // same owner thread, family drop and TP3 ordering as before
        s.counts.evicted += 1; // preserve legacy eviction semantics
        return true;
    }

    pub fn noteImageBypass(s: *Store, tokens: usize) void {
        if (!s.diagnostics) return;
        s.reasons.image_bypass +|= 1;
        s.reasons.image_bypass_prompt_tokens +|= tokens;
    }

    fn noteDead(s: *Store, e: *const Entry) void {
        if (!s.diagnostics) return;
        const reason = if (s.family.vtable.dead_reason) |f| f(s.family.ptr, e.saved) else 0;
        switch (reason) {
            1 => s.reasons.physical_overlap +|= 1,
            2 => s.reasons.placement_pressure +|= 1,
            else => s.reasons.physical_unknown +|= 1,
        }
    }

    fn observeShadow(s: *Store, actual: usize, shadow: ?usize, eligible: u64) void {
        if (!s.diagnostics or !s.prune_dead or shadow == null) return;
        s.audit.shadow_decisions +|= 1;
        s.audit.shadow_eligible +|= eligible;
        if (actual != shadow.?) s.audit.shadow_disagreements +|= 1;
    }

    /// Diagnostic hypothesis ONLY, within the same superseded/fallback tier.
    /// Shared anchors last; fresh entries get 64 store-event grace; real reuse
    /// is capped at 8 observations and marginal credit at 128 chunks. Ties LRU.
    fn shadowLess(s: *const Store, a: *const Entry, b: *const Entry) bool {
        if (a.shared != b.shared) return !a.shared;
        const av = s.shadowValue(a);
        const bv = s.shadowValue(b);
        return av < bv or (av == bv and a.used < b.used);
    }

    fn shadowValue(s: *const Store, e: *const Entry) u64 {
        if (e.reuse.restores == 0) return if (s.clock -| e.used < 64) 1 else 0;
        const marginal = e.reuse.marginal_lower_bound_tokens / e.reuse.restores;
        const hits: u64 = @min(e.reuse.restores, 8);
        const chunks: u64 = @min(marginal / 2048, 128);
        const value = hits * (1 + chunks);
        if (!s.reuse_eviction) return value; // baseline diagnostic unchanged
        const age = s.clock -| e.reuse.last_restore;
        // Event-age decay, not a wall-clock TTL. Never permanently pin a
        // historically popular state; only compare within the original tier.
        return if (age >= 512) 0 else value / (1 + age / 64);
    }

    /// Normalize BEFORE fitting, shared registration and reserve. Automatic
    /// callbacks are included once; the native GPU callback path is unchanged.
    fn actualMarks(s: *Store, a: Allocator, prompt: []const u32, e: ?*Entry, raw: []const u32) ![]const u32 {
        const map = s.family.vtable.map_mark orelse return raw;
        if (!s.normalize_marks) return raw;
        const t0 = if (s.x.trace) x1.now() else 0;
        defer if (s.x.trace) { s.x.plan_ns +|= x1.now() -| t0; };
        defer a.free(raw);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        for (raw) |wanted| {
            const at = map(s.family.ptr, if (e) |v| v.saved else null, prompt.len, wanted);
            s.x.push(.{ .kind = .plan, .request = s.x.request_serial, .desired = wanted, .effective = at });
            if (at == 0) { s.x.omitted +|= 1; continue; }
            if (at != wanted) s.x.mapped +|= 1;
            if (std.mem.indexOfScalar(u32, out.items, at) != null) { s.x.duplicate_marks +|= 1; continue; }
            try out.append(a, at);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// Automatic checkpoints belong in accounting, not in the optional marks
    /// sent to the GPU. This also preserves callbacks if a restore later fails.
    fn reservationMarks(s: *Store, a: Allocator, prompt: []const u32, e: ?*Entry, marks_: []const u32, images: bool) ![]const u32 {
        std.debug.assert(s.normalize_marks);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a); // including a failed toOwnedSlice after the fitted marks were copied
        try out.appendSlice(a, marks_);
        if (s.family.vtable.automatic_marks) |f| {
            var automatic: [2]u32 = undefined;
            const n = f(s.family.ptr, if (e) |v| v.saved else null, prompt.len, &automatic);
            for (automatic[0..n]) |at| {
                if (at == 0 or (images and at == prompt.len)) continue; // neither point is saved by lanes
                if (std.mem.indexOfScalar(u32, out.items, at) != null) { s.x.duplicate_marks +|= 1; continue; }
                try out.append(a, at);
            }
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// Capture old tail evidence BEFORE resumed updates e.last. Token comparison
    /// is bounded by the family's window; the earlier prefix was checked by find.
    pub fn resumedRequest(s: *Store, e: *Entry, prompt: []const u32, ok: bool, r: *RequestState) void {
        const present = std.mem.indexOfScalar(*Entry, s.entries.items, e) != null;
        if (!present) return;
        const continuation = ok and x1.tailEligible(e.x.role, e.shared, e.at, e.last.len, prompt.len, s.family.vtable.tail_window) and
            std.mem.eql(u32, e.last[e.at..], prompt[e.at..e.last.len]);
        const id = e.x.id;
        const cohort = e.x.cohort;
        const parent_request = e.x.created_request;
        const at = e.at;
        s.resumed(e, prompt, ok);
        if (!ok) return; // e may have been freed
        e.x.last_return = r.id;
        r.restored_at = at;
        r.parent = id;
        if (s.adaptive and !e.shared and e.x.history != 0 and (continuation or at == r.input_len)) {
            if (!r.history_observed) r.history = s.history.returned(r.namespace_id, r.id, e.x.history, e.x.history_request, false, at == r.input_len);
            r.history_observed = true;
            e.x.history = r.history;
            e.x.history_request = r.id;
            s.adaptiveHint(e);
        }
        if (continuation) {
            r.continuation = true;
            r.parent_request = parent_request;
            r.cohort = if (cohort != 0) cohort else r.id;
            s.x.continuations +|= 1;
        }
        s.x.push(.{ .kind = .restore, .request = r.id, .namespace_id = r.namespace_id,
            .state = id, .cohort = r.cohort, .parent = r.parent, .role = e.x.role,
            .at = at, .prompt = @intCast(prompt.len), .effective = @intFromBool(continuation) });
    }

    fn xSaved(s: *Store, e: *Entry, key: u64, request: ?*const RequestState) void {
        if (!s.x.enabled and !s.x.trace) return;
        s.x.serial +|= 1;
        const r: RequestState = if (request) |v| v.* else .{};
        const role = r.role(e.at);
        const tail = role == .encoder_tail or role == .prompt_end or role == .decode_end;
        e.x = .{ .id = s.x.serial, .key = key, .cohort = r.cohort, .parent = r.parent,
            .created_request = r.id, .role = role, .continuation = tail and r.continuation,
            .last_return = if (tail and r.continuation) s.x.request_serial else 0 };
        if (s.adaptive and tail and !e.shared) { e.x.history = r.history; e.x.history_request = r.id; }
        s.adaptiveHint(e);
        s.xHint(e, r);
        // Only the same request or a verified old request tail can establish
        // succession. Cohort equality by itself is not proof (forks may share it).
        for (s.entries.items) |old| {
            if (old == e or old.at >= e.at or !std.meta.eql(old.namespace, e.namespace)) continue;
            if (old.x.created_request == 0 or !(old.x.created_request == r.id or
                (r.continuation and old.x.created_request == r.parent_request))) continue;
            if (old.x.role == .auxiliary or (old.x.role == .encoder_tail and role == .encoder_tail)) old.x.successor = e.x.id;
            // Retire old historical ends only after an actually saved, verified
            // next-turn prompt end exists. A decode tail alone cannot replace
            // its own exact input prompt. Encoder fallback remains separate.
            if (r.continuation and old.x.created_request == r.parent_request and role == .prompt_end and
                (old.x.role == .prompt_end or old.x.role == .decode_end)) old.x.successor = e.x.id;
        }
        s.x.push(.{ .kind = .save, .request = r.id, .namespace_id = sharedKey(e.namespace, &.{}),
            .state = e.x.id, .cohort = e.x.cohort, .parent = e.x.parent, .role = role,
            .at = e.at, .prompt = r.input_len, .bytes = e.bytes });
    }

    fn xDedup(s: *Store, e: *Entry, request: ?*const RequestState) void {
        if (s.adaptive and e.shared) { e.x.history = 0; e.x.history_request = 0; }
        const r = request orelse {
            if (e.shared) s.adaptiveHint(e);
            return;
        };
        const role = r.role(e.at);
        if (@intFromEnum(role) > @intFromEnum(e.x.role)) e.x.role = role;
        if (r.continuation and role != .auxiliary and !e.shared) {
            e.x.continuation = true;
            e.x.last_return = r.id;
        }
        if (s.adaptive and !e.shared and role != .auxiliary and role != .unknown) { e.x.history = r.history; e.x.history_request = r.id; }
        if (s.adaptive and e.shared) { e.x.history = 0; e.x.history_request = 0; }
        s.adaptiveHint(e);
        s.xHint(e, r.*);
        s.x.push(.{ .kind = .dedup, .request = r.id, .namespace_id = r.namespace_id,
            .state = e.x.id, .cohort = e.x.cohort, .parent = r.parent, .role = e.x.role, .at = e.at });
    }

    fn adaptiveHint(s: *Store, e: *const Entry) void {
        if (!s.adaptive) return;
        const value = if (e.shared or e.x.history == 0) 0 else s.history.density(sharedKey(e.namespace, &.{}), e.x.history, s.x.request_serial, e.at);
        if (s.family.vtable.adaptive_hint) |f| f(s.family.ptr, e.saved, value);
    }

    fn xHint(s: *Store, e: *const Entry, r: RequestState) void {
        // Shared headers, auxiliary marks and mere length are not evidence that
        // this physical tail belongs to a continuing conversation.
        if (!s.x.enabled or !r.continuation or e.shared) return;
        if (r.role(e.at) == .auxiliary or r.role(e.at) == .unknown) return;
        if (s.family.vtable.continuation_hint) |f| f(s.family.ptr, e.saved, e.x.id);
    }

    fn xRemoved(s: *Store, e: *Entry) void {
        if (!s.x.enabled and !s.x.trace) return;
        const namespace_id = sharedKey(e.namespace, &.{});
        s.x.push(.{ .kind = .drop, .request = s.x.request_serial, .namespace_id = namespace_id,
            .state = e.x.id, .cohort = e.x.cohort, .parent = e.x.parent, .role = e.x.role,
            .at = e.at, .bytes = e.bytes, .reason = s.x.last_reason, .baseline = s.x.last_baseline });
        s.xLost(e, s.x.last_reason);
    }

    // Numeric evidence only; no removal, restore or new token hashing.
    fn xLost(s: *Store, e: *Entry, reason: x1.Reason) void {
        // A failed restore is not evidence that an evicted prefix existed.
        if (e.x.history_lost or reason == .reset or reason == .upgrade or reason == .restore_failed) return;
        e.x.history_lost = true;
        const namespace_id = sharedKey(e.namespace, &.{});
        if (s.adaptive and !e.shared and e.x.history != 0)
            s.history.evicted(namespace_id, e.x.history, s.x.request_serial);
        if (e.x.id != 0 and e.tokens.len == e.at) {
            const range = if (s.family.vtable.ghost_range) |f| f(s.family.ptr, e.saved) else [2]u32{0,0};
            s.x.ghosts[s.x.ghost_next] = .{ .key = e.x.key, .namespace_id = namespace_id,
                .state = e.x.id, .cohort = e.x.cohort, .at = e.at,
                .history = if (e.shared) 0 else e.x.history, .history_request = e.x.history_request,
                .last_token = e.tokens[e.tokens.len - 1], .reason = reason,
                .compatibility_known = s.family.vtable.ghost_range != null,
                .exact_len = if (s.rules.exact_end) range[0] else 0, .min_len = range[1] };
            s.x.ghost_next = (s.x.ghost_next + 1) % x1.ghost_capacity;
        }
    }

    /// At most four hashes / 524288 tokens per request, not per decode step.
    /// Ghost matches are diagnostic ONLY; all restores still compare real tokens.
    fn xGhostReturns(s: *Store, namespace: Namespace, prompt: []const u32, from: u32, r: *RequestState) void {
        if (!s.x.trace and !s.adaptive) return;
        var hashes: usize = 0;
        var tokens: usize = 0;
        for (0..x1.ghost_capacity) |off| {
            const i = (s.x.ghost_next + x1.ghost_capacity - 1 - off) % x1.ghost_capacity;
            const g = &s.x.ghosts[i];
            if (g.state == 0 or g.namespace_id != r.namespace_id or g.at <= from or g.at > prompt.len or prompt[g.at - 1] != g.last_token) continue;
            if (g.compatibility_known and prompt.len != g.exact_len and prompt.len < g.min_len) {
                s.x.ghost_incompatible_skips +|= 1;
                continue;
            }
            if (hashes >= 4 or tokens + g.at > 524288) { s.x.ghost_hash_skips +|= 1; continue; }
            hashes += 1;
            tokens += g.at;
            if (sharedKey(namespace, prompt[0..g.at]) != g.key) continue;
            if (s.adaptive and g.history != 0) {
                r.history = s.history.returned(r.namespace_id, r.id, g.history, g.history_request, true, false);
                r.history_observed = true;
            }
            s.x.ghost_returns +|= 1;
            s.x.push(.{ .kind = .ghost_return, .request = r.id, .namespace_id = r.namespace_id,
                .state = g.state, .cohort = g.cohort, .at = g.at, .effective = from, .prompt = r.input_len, .reason = g.reason });
            g.state = 0;
            // One known lost prefix per request. Does not claim all misses.
            break;
        }
    }

    /// One bounded pressure scan. No wall-clock TTL and no size-only pinning.
    fn xVictim(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) ?usize {
        const t0 = x1.now();
        defer s.x.choose_ns +|= x1.now() -| t0;
        s.victim_dead = false;
        s.x.choices +|= 1;
        if (s.entries.items.len > x1.max_entries) return s.xFallback(namespace, skip, prompt);
        const Ref = struct { id: u64, index: usize };
        var refs: [x1.max_entries]Ref = undefined;
        for (s.entries.items, 0..) |e, i| refs[i] = .{ .id = e.x.id, .index = i };
        const order = struct { fn less(_: void, a: Ref, b: Ref) bool { return a.id < b.id; } }.less;
        std.mem.sort(Ref, refs[0..s.entries.items.len], {}, order);
        var best: ?x1.Candidate = null;
        for (s.entries.items, 0..) |e, i| {
            if (e == skip or e.pins > 0) continue;
            if (s.prune_dead) if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) {
                s.victim_dead = true;
                s.x.last_reason = .dead;
                return i;
            };
            var superseded = false;
            // An explicit successor was established only by a verified request.
            // Historical terminal ends retain a separate priority from auxiliaries.
            if (!e.shared and e.x.successor != 0) {
                var lo: usize = 0;
                var hi = s.entries.items.len;
                while (lo < hi) {
                    const mid = lo + (hi - lo) / 2;
                    if (refs[mid].id < e.x.successor) lo = mid + 1 else hi = mid;
                }
                if (lo < s.entries.items.len and refs[lo].id == e.x.successor) {
                    const o = s.entries.items[refs[lo].index];
                    superseded = if (s.family.vtable.live) |f| f(s.family.ptr, o.saved) else true;
                }
            }
            if (i % 32 == 0 and x1.now() -| t0 > 2 * std.time.ns_per_ms) return s.xFallback(namespace, skip, prompt);
            const terminal = e.x.role == .prompt_end or e.x.role == .decode_end;
            const candidate = x1.candidate(&s.x, e.x, e.reuse.restores, e.at, e.used, i, superseded and !terminal, superseded and terminal);
            if (best == null or x1.less(candidate, best.?)) best = candidate;
        }
        const chosen = best orelse return null;
        // X2: no observational legacy scan on the engine thread. The actual
        // legacy fallback above remains available for bounded-selection failure.
        s.x.last_reason = chosen.reason;
        return chosen.index;
    }

    fn xFallback(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) ?usize {
        s.x.fallbacks +|= 1;
        s.x.last_reason = .legacy_fallback;
        const victim = s.legacyVictimBut(namespace, skip, prompt);
        if (s.victim_dead) s.x.last_reason = .dead;
        return victim;
    }

    pub fn completedRequest(s: *Store, r: RequestState, cached: u32, elapsed_ns: u64, prefill_ns: u64, outcome: u8) void {
        if (r.id == 0) return; // includes image bypass and queued cancellations
        s.x.push(.{ .kind = .finished, .request = r.id, .namespace_id = r.namespace_id,
            .cohort = r.cohort, .parent = r.parent, .prompt = r.input_len, .at = cached,
            .elapsed_ns = elapsed_ns, .prefill_ns = prefill_ns, .outcome = outcome });
        if (s.adaptive) {
            s.history.completed(r.namespace_id, r.history, r.id, r.input_len, cached, prefill_ns, outcome == 1);
            if (s.first_continuation and outcome == 1) if (s.history.get(r.namespace_id, r.history)) |h| {
                if (h.returns == 0) if (s.family.vtable.first_hint) |hint| {
                    for (s.entries.items) |e| if (!e.shared and e.x.created_request == r.id and
                        (e.x.role == .prompt_end or e.x.role == .decode_end)) hint(s.family.ptr, e.saved);
                };
            };
            // Update only this request's saved endpoints, once at completion.
            for (s.entries.items) |e| if (e.x.created_request == r.id) { s.adaptiveHint(e); };
            if (!@import("builtin").is_test and s.x.trace) std.log.info("checkpoint_x3: request={d} namespace={d} history={d} density_us={d} returns={d} forks={d} ghosts={d} cost_samples={d} replaced={d}", .{
                r.id, r.namespace_id, r.history, s.history.density(r.namespace_id, r.history, s.x.request_serial, r.input_len),
                s.history.counters.returns, s.history.counters.forks, s.history.counters.ghosts,
                s.history.counters.cost_samples, s.history.counters.replaced,
            });
        }
        s.flushX1();
    }

    pub fn flushX1(s: *Store) void {
        if (@import("builtin").is_test or !s.x.trace) return;
        // Bounded, numeric metadata, flushed only at prefill/completion boundaries.
        for (0..s.x.event_count) |off| {
            const e = s.x.events[(s.x.event_start + off) % x1.trace_capacity];
            std.log.info("checkpoint_x1: {{\"kind\":\"{s}\",\"reason\":\"{s}\",\"role\":\"{s}\",\"request\":{d},\"namespace\":{d},\"state\":{d},\"cohort\":{d},\"parent\":{d},\"at\":{d},\"prompt\":{d},\"desired\":{d},\"effective\":{d},\"bytes\":{d},\"baseline\":{d},\"elapsed_ns\":{d},\"prefill_ns\":{d},\"outcome\":{d},\"event_ns\":{d}}}", .{ @tagName(e.kind), @tagName(e.reason), @tagName(e.role), e.request, e.namespace_id, e.state, e.cohort, e.parent, e.at, e.prompt, e.desired, e.effective, e.bytes, e.baseline, e.elapsed_ns, e.prefill_ns, e.outcome, e.event_ns });
        }
        s.x.event_count = 0;
        s.x.event_start = 0;
        std.log.info("checkpoint_x1 totals: choices {d}, sampled {d}, different {d}, fallback {d}, mapped {d}, omitted {d}, duplicate_marks {d}, compatible_reservations {d}, continuations {d}, ghost_returns {d}, ghost_hash_skips {d}, ghost_incompatible_skips {d}, overwritten {d}, choose_ns {d}, plan_ns {d}, save_ns {d}, drop_ns {d}", .{ s.x.choices, s.x.sampled_choices, s.x.sampled_different, s.x.fallbacks, s.x.mapped, s.x.omitted, s.x.duplicate_marks, s.x.compatible_reservations, s.x.continuations, s.x.ghost_returns, s.x.ghost_hash_skips, s.x.ghost_incompatible_skips, s.x.overwritten, s.x.choose_ns, s.x.plan_ns, s.x.save_ns, s.x.drop_ns });
    }

    // -- 0111: continuation ---------------------------------------------------------------------------------------

    /// The conversation a request belongs to, decided at its lookup (`hit`: the state it resumes, null: none). A hit on
    /// one of a conversation's own turn states keeps that conversation, when the prompt shows it is that conversation
    /// (`ownHit`); otherwise (a shared prefix, a state read back from disk, an own state of another conversation, or
    /// nothing) the most recently used own state the prompt repeats past its conversation's first prompt and the first
    /// chat_match reply tokens names it (`matchable`: no pictures and drafted, as the source requires); else a new
    /// conversation. Deterministic in the entries' use order. Off (group 0) unless `groups`.
    pub fn join(s: *Store, prompt: []const u32, hit: ?*const Entry, matchable: bool) Join {
        if (!s.groups) return .{};
        const source: Join.Source = if (hit) |e| (if (e.shared) .shared else .other) else .nothing;
        if (hit) |e| if (s.ownHit(e, prompt)) return .{ .group = e.group, .source = .other };
        if (matchable) if (s.continuedChat(prompt)) |c| return .{
            .group = c.group,
            .continued = true,
            .prior_len = @intCast(c.tokens.len),
            .back = @intCast(commonPrefix(c.last, prompt)),
            .source = source,
        };
        s.next_group += 1;
        const g = s.next_group;
        s.roots.put(s.gpa, g, @intCast(prompt.len)) catch {};
        if (s.roots.count() > 4 * @max(16, s.entries.items.len)) s.pruneRoots(g); // conversations no kept state is in
        return .{ .group = g, .source = source };
    }

    /// Whether resuming `e` keeps `e`'s conversation: one of its own states (not shared, not read back), and the prompt
    /// repeats its evidence (the last prompt that kept or resumed it) up to its conversation's first prompt and first
    /// chat_match reply tokens, or all of it when shorter. GLM's own states always hold conversation-specific tokens;
    /// a native state on the grid or at last_end can lie wholly inside a system prompt other conversations share, so
    /// the hit alone is no evidence. A conversation whose first prompt was forgotten keeps the hit-alone rule.
    fn ownHit(s: *const Store, e: *const Entry, prompt: []const u32) bool {
        if (e.shared or e.restored or e.group == 0) return false;
        const root = s.roots.get(e.group) orelse return true;
        return commonPrefix(e.last, prompt) >= @min(e.last.len, @as(usize, root) + chat_match);
    }

    /// The own kept state whose conversation `prompt` continues (continued_chat): most recently used first; a later
    /// entry wins a tie of use. Shared, restored, pictured and forgotten conversations' states are passed over.
    pub fn continuedChat(s: *const Store, prompt: []const u32) ?*Entry {
        var best: ?*Entry = null;
        for (s.entries.items) |c| {
            if (c.shared or c.restored or c.images or c.group == 0) continue;
            const root = s.roots.get(c.group) orelse continue;
            const n = @as(usize, root) + chat_match;
            if (c.last.len < n or prompt.len < n or c.last[n - 1] != prompt[n - 1] or !std.mem.eql(u32, c.last[0..n], prompt[0..n])) continue;
            if (best == null or c.used >= best.?.used) best = c;
        }
        return best;
    }

    fn pruneRoots(s: *Store, keep_: u64) void {
        var gone: std.ArrayList(u64) = .empty;
        defer gone.deinit(s.gpa);
        var it = s.roots.keyIterator();
        while (it.next()) |k| {
            if (k.* == keep_) continue;
            const held = for (s.entries.items) |e| {
                if (e.group == k.*) break true;
            } else false;
            if (!held) gone.append(s.gpa, k.*) catch return;
        }
        for (gone.items) |k| _ = s.roots.remove(k);
    }

    /// After a continued request's pass resumed at `cut` of `len` tokens: counted, and a large prefill logged with where
    /// the prompt left the kept state and what it resumed from (only rank 0 runs the host).
    pub fn continued(s: *Store, j: Join, cut: u32, len: usize) void {
        if (!j.continued) return;
        s.stats.chats_continued += 1;
        s.stats.chats_continued_prefill_tokens += len -| cut;
        if (!shouldLog(j, cut, len) or @import("builtin").is_test) return;
        var buf: [512]u8 = undefined;
        std.log.info("{s}", .{continuedLine(&buf, j, cut, len)});
    }
};

/// Whether `continued` logs a continued request's pass (from `cut` of `len` tokens): its prefill is large.
pub fn shouldLog(j: Join, cut: u32, len: usize) bool {
    return j.continued and len -| cut >= chat_log_tokens;
}

/// continued's log line.
pub fn continuedLine(buf: []u8, j: Join, cut: u32, len: usize) []const u8 {
    const source = switch (j.source) {
        .nothing => "nothing",
        .shared => "a shared prefix",
        .other => "a state that is not a turn's own (read back from disk)",
    };
    var why_buf: [256]u8 = undefined;
    const why = (if (j.back < j.prior_len)
        std.fmt.bufPrint(&why_buf, "it went back to token {d}, before that conversation's newest kept state ({d} tokens)", .{ j.back, j.prior_len })
    else
        std.fmt.bufPrint(&why_buf, "its kept state of {d} tokens could not serve it (stale, off the grid or past the replay cut)", .{j.prior_len})) catch "";
    return std.fmt.bufPrint(buf, "prompt cache: conversation {d}: resumed at {d} of {d} tokens from {s}, prefilling {d}; {s}", .{ j.group, cut, len, source, len -| cut, why }) catch buf[0..0];
}

/// The tokens two id lists share from the start.
pub fn commonPrefix(a: []const u32, b: []const u32) usize {
    return std.mem.indexOfDiff(u32, a, b) orelse @min(a.len, b.len);
}

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

/// The chunk start at or before `at` (0 when none).
fn floorStart(starts: []const u32, at: u32) u32 {
    var best: u32 = 0;
    for (starts) |p| if (p <= at and p > best) {
        best = p;
    };
    return best;
}

test {
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_cache_tier_test.zig");
    _ = @import("prompt_cache_fork.zig");
}

// K2a regression harness: no CUDA, no IO; real Store with a fake family.
const K2aFake = struct {
    drops: usize = 0,
    live_calls: usize = 0,
    const State = struct { live: bool = true, compatible: bool = true, hint_state: u64 = 0, hints: u64 = 0, short: bool = false, at: u32 = 0 };
    fn family(f: *K2aFake, liveness: bool) Snapshots {
        return .{ .ptr = f, .vtable = if (liveness) &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop_, .live = live, .compatible = compatible } else &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop_ } };
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!Saved {
        const st = try std.testing.allocator.create(State);
        st.* = .{};
        return st;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: Saved) anyerror!void {}
    fn drop_(p: *anyopaque, saved: Saved) void {
        const f: *K2aFake = @ptrCast(@alignCast(p));
        f.drops += 1;
        std.testing.allocator.destroy(@as(*State, @ptrCast(@alignCast(saved))));
    }
    fn live(p: *anyopaque, saved: Saved) bool {
        const f: *K2aFake = @ptrCast(@alignCast(p));
        f.live_calls += 1;
        return (@as(*State, @ptrCast(@alignCast(saved)))).live;
    }
    fn compatible(_: *anyopaque, saved: Saved, _: usize) bool {
        return (@as(*State, @ptrCast(@alignCast(saved)))).compatible;
    }
    fn state(e: *Entry) *State {
        return @ptrCast(@alignCast(e.saved));
    }
    fn shortEligible(_: *anyopaque, saved: Saved, len: u32) bool {
        const st: *State = @ptrCast(@alignCast(saved));
        return st.short and st.at > 0 and st.at % 2 == 0 and len > st.at and len - st.at < 4;
    }
    fn add(s: *Store, tokens: []const u32, at: u32) !*Entry {
        const a = std.testing.allocator;
        const e = try a.create(Entry);
        const st = try a.create(State);
        st.* = .{ .at = at };
        s.clock += 1;
        e.* = .{ .tokens = try a.dupe(u32, tokens), .at = at, .saved = st, .bytes = 1, .born = @intCast(tokens.len), .used = s.clock, .last = try a.dupe(u32, tokens) };
        try s.entries.append(a, e);
        s.held += 1;
        return e;
    }
};

test "K2a physical dead state goes before valid LRU, including shared and foreign namespace" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 3);
    defer s.deinit();
    const a = try K2aFake.add(&s, &.{1}, 1);
    const dead = try K2aFake.add(&s, &.{2}, 1);
    _ = try K2aFake.add(&s, &.{3}, 1);
    dead.shared = true;
    dead.namespace = namespaceFor("other");
    K2aFake.state(dead).live = false;
    s.reserve(null, &.{9}, null, &.{1});
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    try std.testing.expectEqual(@as(u64, 1), s.audit.dead_at_reserve);
    try std.testing.expectEqual(@as(u64, 0), s.audit.valid_budget_evictions);
    try std.testing.expect(s.entries.items[0] == a);
    try std.testing.expectEqual(@as(u64, 2), s.held);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expectEqual(@as(u64, 0), s.audit.shadow_decisions);
}

test "K2a reserve preserves borrowed resume even when dead, drops other dead exactly once" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 3);
    defer s.deinit();
    const borrowed = try K2aFake.add(&s, &.{1}, 1);
    const dead = try K2aFake.add(&s, &.{2}, 1);
    _ = try K2aFake.add(&s, &.{3}, 1);
    K2aFake.state(borrowed).live = false;
    K2aFake.state(dead).live = false;
    s.reserve(null, &.{9}, borrowed, &.{1});
    try std.testing.expect(s.entries.items[0] == borrowed);
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    // No pressure: no liveness scan and no second drop.
    const calls = f.live_calls;
    s.reserve(null, &.{9}, borrowed, &.{});
    try std.testing.expectEqual(calls, f.live_calls);
    try std.testing.expectEqual(@as(usize, 1), f.drops);
}

test "K2a shadow disagrees but actual valid victim stays LRU; safe off and absent liveness work" {
    for ([_]bool{ true, false }) |liveness| {
        var f: K2aFake = .{};
        var s = Store.init(std.testing.allocator, f.family(liveness), .{}, 3);
        defer s.deinit();
        const a = try K2aFake.add(&s, &.{1}, 1);
        const b = try K2aFake.add(&s, &.{2}, 1);
        _ = try K2aFake.add(&s, &.{3}, 1);
        a.reuse.restores = 8;
        a.reuse.marginal_lower_bound_tokens = 32768;
        try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
        try std.testing.expectEqual(@as(u64, 1), s.audit.shadow_disagreements);
        s.diagnostics = false;
        try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
        try std.testing.expectEqual(@as(u64, 1), s.audit.shadow_disagreements);
        s.reserve(null, &.{9}, null, &.{1});
        try std.testing.expect(s.entries.items[0] == b);
        try std.testing.expectEqual(@as(u64, 1), s.audit.valid_budget_evictions);
    }
}

test "K2a no-dead pressure matches baseline policy across randomized tiers, namespaces and borrowers" {
    var prng = std.Random.DefaultPrng.init(0x4b3241);
    const rng = prng.random();
    for (0..200) |_| {
        var f: K2aFake = .{};
        var s = Store.init(std.testing.allocator, f.family(true), .{}, 8);
        defer s.deinit();
        for (0..8) |j| {
            const e = try K2aFake.add(&s, &.{ @intCast(j % 3), 2, 3 }, @intCast(j % 3 + 1));
            e.shared = rng.boolean();
            e.namespace = if (rng.boolean()) namespaceFor("A") else null;
            e.used = rng.intRangeAtMost(u64, 0, 100);
            e.born = rng.intRangeAtMost(u32, 1, 5);
            e.reuse.restores = rng.intRangeAtMost(u64, 0, 9);
        }
        const skip: ?*Entry = if (rng.boolean()) s.entries.items[rng.uintLessThan(usize, 8)] else null;
        s.prune_dead = false;
        s.diagnostics = false;
        const baseline = s.victimBut(null, skip, &.{ 1, 2, 3, 4 });
        s.prune_dead = true;
        s.diagnostics = true;
        try std.testing.expectEqual(baseline, s.victimBut(null, skip, &.{ 1, 2, 3, 4 }));
    }
}

test "K2a keep dedup is not restore; only successful restore earns tokens and marginal lower bound" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 10);
    defer s.deinit();
    const a = try K2aFake.add(&s, &.{ 1, 2 }, 2);
    const best = try K2aFake.add(&s, &.{ 1, 2, 3, 4 }, 4);
    try std.testing.expect(s.keep(&.{ 1, 2, 3, 4, 5 }, 4, null, &.{}));
    try std.testing.expectEqual(@as(u64, 0), best.reuse.restores);
    try std.testing.expectEqual(@as(u32, 1), best.reuse.dedups);
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 5 }, &.{}) == best);
    s.resumed(best, &.{ 1, 2, 3, 4, 5 }, true);
    try std.testing.expectEqual(@as(u64, 1), best.reuse.restores);
    try std.testing.expectEqual(@as(u64, 4), best.reuse.restored_tokens);
    try std.testing.expectEqual(@as(u64, 2), best.reuse.marginal_lower_bound_tokens);
    try std.testing.expectEqual(s.clock, best.reuse.last_restore);
    s.resumed(a, &.{ 1, 2, 3, 4, 5 }, false);
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    try std.testing.expectEqual(@as(u64, 4), s.audit.restored_tokens);
    try std.testing.expect(@sizeOf(Reuse) <= 64);
}

test "K2a marginal bound stays conservative without comparing skipped prefixes; incompatible is not dead" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 3);
    defer s.deinit();
    const best = try K2aFake.add(&s, &.{ 1, 2, 3, 4 }, 4);
    _ = try K2aFake.add(&s, &.{ 9, 9, 9 }, 3); // skipped: may match, conservatively counts as alternative
    const incompatible = try K2aFake.add(&s, &.{ 1, 2 }, 2);
    K2aFake.state(incompatible).compatible = false;
    incompatible.shared = true; // valid shared anchor must survive while an ordinary victim is available
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 5 }, &.{}) == best);
    try std.testing.expectEqual(@as(u32, 1), best.reuse.pending_marginal);
    s.reserve(null, &.{8}, best, &.{1});
    try std.testing.expectEqual(@as(u64, 0), s.audit.dead_at_reserve);
    try std.testing.expectEqual(@as(u64, 1), s.audit.valid_budget_evictions);
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, incompatible) != null);
}

test "K2a dead cleanup safe off reproduces old LRU and handles multiple dead with insufficient room" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 3);
    defer s.deinit();
    _ = try K2aFake.add(&s, &.{1}, 1);
    const dead = try K2aFake.add(&s, &.{2}, 1);
    const dead2 = try K2aFake.add(&s, &.{3}, 1);
    K2aFake.state(dead).live = false;
    K2aFake.state(dead2).live = false;
    s.prune_dead = false;
    s.reserve(null, &.{9}, null, &.{1});
    try std.testing.expect(s.entries.items[0] == dead);
    s.prune_dead = true;
    s.reserve(null, &.{9}, null, &.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 2), s.audit.dead_at_reserve);
    try std.testing.expectEqual(@as(usize, 3), f.drops);
}

test "rewind lookup chooses the longest legal kept grid prefix" {
    const a = std.testing.allocator;
    var f: K2aFake = .{};
    var s = Store.init(a, f.family(true), .{ .min_prompt = 0, .grid = 2048, .window = 128, .lookahead = 1, .planned = true }, 8);
    defer s.deinit();
    const original = try a.alloc(u32, 6400);
    defer a.free(original);
    for (original, 0..) |*token, i| token.* = @intCast(i);
    const starts = [_]u32{ 0, 2048, 4096, 6144 };
    for (starts[1..]) |at| try std.testing.expect(s.keep(original, at, null, &starts));

    const shorter = original[0..5000];
    const divergent = try a.dupe(u32, original);
    defer a.free(divergent);
    divergent[4096] +%= 1; // lookahead at 4096 makes that checkpoint unsafe
    const extended = try a.alloc(u32, original.len + 2);
    defer a.free(extended);
    @memcpy(extended[0..original.len], original);
    extended[original.len] = 99999;
    extended[original.len + 1] = 100000;
    const prompts = [_][]const u32{ shorter, divergent, extended };
    const expected = [_]u32{ 4096, 2048, 6144 };
    const common = [_]usize{ shorter.len, 4096, original.len };
    for (prompts, expected, common) |prompt, at, common_len| {
        const entry = s.find(prompt, &starts).?;
        try std.testing.expectEqual(at, entry.at);
        try std.testing.expect(entry.at < prompt.len);
        try std.testing.expect(entry.tokens.len <= common_len);
        try std.testing.expectEqualSlices(u32, entry.tokens, prompt[0..entry.tokens.len]);
    }
    // A matching 4096 checkpoint is still unusable when fewer than 128 rows remain.
    try std.testing.expectEqual(@as(u32, 2048), s.find(original[0..4200], &starts).?.at);
    try std.testing.expectEqual(@as(u64, 3), s.counts.kept);
    try std.testing.expectEqual(@as(u64, 0), s.counts.hits); // find alone does not restore
    try std.testing.expectEqual(@as(u64, 3), s.held);
}

test "fork miss learns then hits within namespace and ordinary budget" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var f: K2aFake = .{};
    var s = Store.init(a, f.family(true), .{ .fork = true, .min_prompt = 0, .lookahead = 1 }, 8);
    defer s.deinit();
    const one = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 91, 92, 70, 71 };
    const two = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 93, 94, 70, 71 };
    const three = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 95, 96, 70, 71 };
    const ns = namespaceFor("candidate");
    try std.testing.expect(s.keepScoped(ns, &one, 10, null, &.{}));
    var l = try s.lookupScoped(ns, arena.allocator(), &two, 10, &.{}, &.{});
    try std.testing.expect(l.entry == null);
    try std.testing.expectEqualSlices(u32, &.{ 7, 10 }, l.marks);
    for (l.marks) |at| try std.testing.expect(s.keepScoped(ns, &two, at, null, &.{}));
    l = try s.lookupScoped(ns, arena.allocator(), &three, 10, &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 7), l.entry.?.at);
    try std.testing.expect(s.findScoped(null, &three, &.{}) == null);
    try std.testing.expect(s.held <= s.budget);
}

test "fork evidence excludes dead and stale sources and honors native rules without a tier" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var f: K2aFake = .{};
    var s = Store.init(a, f.family(true), .{ .fork = true, .min_prompt = 0, .grid = 2, .window = 2, .lookahead = 1, .planned = true }, 8);
    defer s.deinit();
    const one = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 91, 92, 70, 71 };
    const two = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 93, 94, 70, 71 };
    const e = try K2aFake.add(&s, one[0..10], 10);
    // last is the full source prompt, not just its checkpoint key.
    a.free(e.last);
    e.last = try a.dupe(u32, &one);
    e.born = one.len;
    const starts = [_]u32{ 0, 4, 6, 10 };
    var l = try s.lookup(a, &two, 10, &.{}, &starts);
    try std.testing.expectEqualSlices(u32, &.{ 6, 10 }, l.marks);
    a.free(l.marks);
    K2aFake.state(e).live = false;
    l = try s.lookup(arena.allocator(), &two, 10, &.{}, &starts);
    try std.testing.expectEqualSlices(u32, &.{10}, l.marks);
    K2aFake.state(e).live = true;
    e.at = 9; // physical state still exists but is off the family's resume grid
    l = try s.lookup(arena.allocator(), &two, 10, &.{}, &starts);
    try std.testing.expectEqualSlices(u32, &.{10}, l.marks);
}

test "fork near history also survives on a hit and image evidence never learns" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var f: K2aFake = .{};
    var s = Store.init(a, f.family(true), .{ .fork = true, .min_prompt = 0, .lookahead = 1 }, 8);
    defer s.deinit();
    const one = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 91, 92, 70, 71 };
    const two = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 93, 94, 70, 71 };
    const e = try K2aFake.add(&s, one[0..3], 2);
    a.free(e.last);
    e.last = try a.dupe(u32, &one);
    e.born = one.len;
    const l = try s.lookup(arena.allocator(), &two, 10, &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 2), l.entry.?.at);
    try std.testing.expectEqualSlices(u32, &.{ 7, 10 }, l.marks);
    s.remove(0);
    const image = try K2aFake.add(&s, one[0..11], 10);
    a.free(image.last);
    image.last = try a.dupe(u32, &one);
    image.born = one.len;
    image.images = true;
    const m = try s.lookup(arena.allocator(), &two, 10, &.{}, &.{});
    try std.testing.expect(m.entry == null);
    try std.testing.expectEqualSlices(u32, &.{10}, m.marks);
}

test "scoped resume pins survive real budget eviction and retired cancellation" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 2);
    defer s.deinit();
    const a = try K2aFake.add(&s, &.{1}, 1);
    s.pin(a);
    s.pin(a); // two admissions borrowing the same snapshot
    _ = try K2aFake.add(&s, &.{2}, 1);
    s.reserve(null, &.{9}, null, &.{1});
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    try std.testing.expectEqual(@as(u32, 1), a.at);
    s.forget(&.{1}, 1); // invalidation cannot free another admission's source
    try std.testing.expectEqual(@as(u64, 1), s.held);
    s.unpin(a); // first cancelled
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    s.unpin(a); // second cancelled before begin
    try std.testing.expectEqual(@as(usize, 2), f.drops);
    try std.testing.expectEqual(@as(u64, 0), s.held);
}

test "terminal native grid composes exact end with family and namespace eligibility" {
    var f: K2aFake = .{};
    var store = Store.init(std.testing.allocator, f.family(true), .{ .exact_end = true, .grid = 4, .window = 2 }, 64);
    defer store.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    const exact = try K2aFake.add(&store, &prompt, 7);
    try std.testing.expect(store.find(&prompt, &.{}) == exact);
    K2aFake.state(exact).compatible = false;
    try std.testing.expect(store.find(&prompt, &.{}) == null);
    K2aFake.state(exact).compatible = true;
    K2aFake.state(exact).live = false;
    try std.testing.expect(store.find(&prompt, &.{}) == null);
    K2aFake.state(exact).live = true;
    exact.namespace = namespaceFor("foreign");
    try std.testing.expect(store.find(&prompt, &.{}) == null);
    exact.namespace = null;
    exact.at = 6; // incomplete off-grid state is not rescued by exact-end support
    try std.testing.expect(store.find(&prompt, &.{}) == null);
}

test "short native reuse requires decoded proof and still respects namespace and liveness" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.short_eligible = K2aFake.shortEligible;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt },
        .{ .exact_end = true, .grid = 2, .window = 4 }, 64);
    defer s.deinit();
    const e = try K2aFake.add(&s, &.{ 1, 2, 3, 4, 5, 6 }, 6);
    const next = &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try std.testing.expect(s.find(next, &.{}) == null);
    K2aFake.state(e).short = true;
    try std.testing.expect(s.find(next, &.{}) == e);
    try std.testing.expect(s.findScoped(namespaceFor("other"), next, &.{}) == null);
    K2aFake.state(e).live = false;
    try std.testing.expect(s.find(next, &.{}) == null);
    K2aFake.state(e).live = true;
    K2aFake.state(e).compatible = false;
    try std.testing.expect(s.find(next, &.{}) == null);
}

// UNRUN: source-only regressions; compilation and execution remain on hold.
test "pinned and retired cache storage refuses an over-budget save without restores" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 1);
    defer s.deinit();
    const old = try K2aFake.add(&s, &.{1}, 1);
    s.pin(old);
    defer s.unpin(old);
    try std.testing.expect(!s.keep(&.{2}, 1, null, &.{}));
    try std.testing.expectEqual(@as(u64, 1), s.held);
    s.forget(&.{1}, 1);
    try std.testing.expect(old.retired);
    try std.testing.expect(!s.keep(&.{2}, 1, null, &.{}));
    try std.testing.expectEqual(@as(u64, 1), s.held);
    try std.testing.expectEqual(@as(usize, 0), f.drops);
}

// UNRUN: both stale physical state and encoder-only terminal upgrades.
test "pinned invalid same-key cache state retires and permits a valid replacement" {
    for ([_]bool{ false, true }) |upgrade| {
        var f: K2aFake = .{};
        var s = Store.init(std.testing.allocator, f.family(true), .{ .exact_end = true }, 2);
        defer s.deinit();
        const old = try K2aFake.add(&s, &.{1}, 1);
        s.pin(old);
        defer s.unpin(old);
        if (upgrade) K2aFake.state(old).compatible = false else K2aFake.state(old).live = false;
        try std.testing.expect(s.keep(&.{1}, 1, null, &.{}));
        try std.testing.expect(old.retired);
        try std.testing.expectEqual(@as(usize, 0), f.drops);
        try std.testing.expectEqual(@as(u64, 2), s.held);
        try std.testing.expect(s.find(&.{1}, &.{}) != old);
        try std.testing.expect(s.find(&.{1}, &.{}) != null);
        try std.testing.expectEqual(@as(u64, 1), s.counts.kept);
    }
}

// UNRUN: retain ordinary pressure eviction when no scoped borrower owns the victim.
test "cache save still evicts an unpinned victim within budget" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 1);
    defer s.deinit();
    _ = try K2aFake.add(&s, &.{1}, 1);
    try std.testing.expect(s.keep(&.{2}, 1, null, &.{}));
    try std.testing.expectEqual(@as(u64, 1), s.held);
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    try std.testing.expectEqualSlices(u32, &.{2}, s.entries.items[0].tokens);
}


test "126b reusable victim policy remains pressure only and preserves borrowed state" {
    const a=std.testing.allocator;
    var fake: K2aFake=.{};
    var s=Store.init(a,fake.family(true),.{},32);
    defer s.deinit();
    s.reuse_eviction=true;
    const t0=[_]u32{1,10,11};const t1=[_]u32{2,20,21};const t2=[_]u32{3,30,31};
    const hot=try K2aFake.add(&s,&t0,1);
    const cold=try K2aFake.add(&s,&t1,1);
    const borrowed=try K2aFake.add(&s,&t2,1);
    s.clock=100;hot.used=90;hot.reuse=.{.restores=4,.last_restore=90,.marginal_lower_bound_tokens=4*2048*64};
    cold.used=95;borrowed.used=0;
    try std.testing.expectEqual(@as(usize,1),s.victimBut(null,borrowed,&.{}).?);
    try std.testing.expect(K2aFake.state(hot).live and K2aFake.state(cold).live and K2aFake.state(borrowed).live);
    // An old popular entry cannot remain pinned forever by historical hits.
    s.clock=700;
    try std.testing.expectEqual(@as(usize,0),s.victimBut(null,borrowed,&.{}).?);
}

test "126b dead state still wins before reusable or shared states" {
    const a=std.testing.allocator;var fake: K2aFake=.{};
    var s=Store.init(a,fake.family(true),.{},32);defer s.deinit();s.reuse_eviction=true;
    const t0=[_]u32{1,10};const t1=[_]u32{2,20};
    const live_=try K2aFake.add(&s,&t0,1);const dead=try K2aFake.add(&s,&t1,1);
    live_.used=0;K2aFake.state(dead).live=false;
    try std.testing.expectEqual(@as(usize,1),s.victimBut(null,null,&.{}).?);
    try std.testing.expect(s.victim_dead);
}

// X1 exercises the real Store with fake snapshots; no inference benchmark.
const X1Grid = struct {
    fn failSave(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!Saved { return error.InjectedSaveFailure; }
    fn map(_: *anyopaque, _: ?Saved, len: usize, at: u32) u32 {
        const effective = at / 2048 * 2048;
        return if (effective < len and effective <= len -| 128) effective else 0;
    }
    fn automatic(_: *anyopaque, _: ?Saved, len: usize, out: *[2]u32) usize {
        out.* = .{ @intCast(std.mem.alignBackward(usize, len -| 128, 2)), @intCast(len) };
        return 2;
    }
};

test "X1 effective marks and shared keys agree; duplicate reservations are removed" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.map_mark = X1Grid.map;
    vt.automatic_marks = X1Grid.automatic;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 20);
    defer s.deinit();
    s.normalize_marks = true;
    s.x.trace = true;
    const tokens = try std.testing.allocator.alloc(u32, 10000);
    defer std.testing.allocator.free(tokens);
    @memset(tokens, 1);
    const l = try s.lookupScoped(null, std.testing.allocator, tokens, 9500, &.{2952, 4488, 5000}, &.{});
    defer std.testing.allocator.free(l.marks);
    try std.testing.expectEqualSlices(u32, &.{2048, 4096, 8192}, l.marks);
    const reservation = try s.reservationMarks(std.testing.allocator, tokens, null, l.marks, false);
    defer std.testing.allocator.free(reservation);
    try std.testing.expectEqualSlices(u32, &.{2048, 4096, 8192, 9872, 10000}, reservation);
    try std.testing.expectEqual(@as(u64, 1), s.x.duplicate_marks);
    try std.testing.expectEqual(@as(usize, 2), s.shared_keys.items.len);
    try std.testing.expect(s.keepRequestScoped(null, tokens, 4096, null, &.{}, &l.request));
    try std.testing.expect(s.entries.items[0].shared);
}

test "X1 compatible saved keys and terminal upgrades reserve no duplicate storage" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{ .exact_end = true }, 3);
    defer s.deinit();
    s.normalize_marks = true;
    const end = try K2aFake.add(&s, &.{1, 2}, 2);
    const other = try K2aFake.add(&s, &.{8, 9}, 2);
    _ = try K2aFake.add(&s, &.{7}, 1);
    s.reserve(null, &.{1, 2}, end, &.{2, 2});
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    K2aFake.state(end).compatible = false;
    s.reserve(null, &.{1, 2}, null, &.{2});
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    try std.testing.expect(s.keep(&.{1, 2}, 2, null, &.{}));
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, other) != null);
    try std.testing.expectEqual(@as(u64, 1), s.audit.upgrades);
    try std.testing.expectEqual(@as(u64, 3), s.held);
}

test "X1 pressure evicts old one-off before active long and short; borrowed dead is preserved" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 4);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true, .serial = 100, .request_serial = 100 };
    const cold = try K2aFake.add(&s, &.{1}, 1);
    cold.x = .{ .id = 1, .role = .prompt_end };
    const long = try K2aFake.add(&s, &.{2}, 1);
    long.x = .{ .id = 2, .role = .prompt_end, .last_return = 100, .continuation = true };
    long.reuse.restores = 2;
    const short = try K2aFake.add(&s, &.{3}, 1);
    short.x = .{ .id = 3, .role = .prompt_end, .last_return = 100 };
    short.reuse.restores = 5;
    const borrowed = try K2aFake.add(&s, &.{4}, 1);
    borrowed.x.id = 4;
    K2aFake.state(borrowed).live = false;
    s.reserve(null, &.{9}, borrowed, &.{1});
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, long) != null);
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, short) != null);
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, borrowed) != null);
    try std.testing.expectEqual(x1.Reason.unused, s.x.last_reason);
}

test "X1 actual tail restore passes bounded credit; shared header and other namespace do not" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.tail_window = 128;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 10);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true };
    var initial: RequestState = .{ .id = 1, .cohort = 1, .input_len = 4 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4}, 4, null, &.{}, &initial));
    const old = s.entries.items[0];
    const l = try s.lookupScoped(null, std.testing.allocator, &.{1, 2, 3, 4, 5, 6}, 0, &.{}, &.{});
    defer std.testing.allocator.free(l.marks);
    var r = l.request;
    s.resumedRequest(old, &.{1, 2, 3, 4, 5, 6}, true, &r);
    try std.testing.expect(r.continuation);
    try std.testing.expectEqual(@as(u64, 1), old.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &r));
    const next = s.entries.items[1];
    try std.testing.expect(next.x.continuation);
    try std.testing.expectEqual(@as(u64, 0), next.reuse.restores);
    try std.testing.expectEqual(old.x.id, next.x.parent);
    try std.testing.expectEqual(next.x.id, old.x.successor);
    // The historical restored end must not out-rank its current descendant
    // merely because only the old object owns the genuine restore counter.
    try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
    try std.testing.expectEqual(x1.Reason.superseded_tail, s.x.last_reason);
    const foreign = try s.lookupScoped(namespaceFor("other"), std.testing.allocator, &.{1, 2, 3, 4, 5, 6, 7}, 0, &.{}, &.{});
    defer std.testing.allocator.free(foreign.marks);
    try std.testing.expect(foreign.entry == null);
    try std.testing.expect(!foreign.request.continuation);
    old.shared = true;
    var header: RequestState = .{ .id = 9, .cohort = 9, .input_len = 7 };
    s.resumedRequest(old, &.{1, 2, 3, 4, 5, 6, 7}, true, &header);
    try std.testing.expect(!header.continuation);
    try std.testing.expectEqual(@as(u64, 9), header.cohort);
}

test "X1 successor priority requires a live replacement; fallback encoder survives terminal save" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{ .exact_end = true }, 10);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true, .request_serial = 1 };
    const r: RequestState = .{ .id = 1, .cohort = 1, .input_len = 6, .encoder_at = 4 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 2, null, &.{}, &r));
    const aux = s.entries.items[0];
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 4, null, &.{}, &r));
    const encoder = s.entries.items[1];
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &r));
    const terminal = s.entries.items[2];
    try std.testing.expectEqual(terminal.x.id, aux.x.successor);
    try std.testing.expectEqual(@as(u64, 0), encoder.x.successor);
    try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
    try std.testing.expectEqual(x1.Reason.superseded_auxiliary, s.x.last_reason);
    K2aFake.state(terminal).live = false;
    try std.testing.expectEqual(@as(?usize, 2), s.victimBut(null, null, &.{9}));
    try std.testing.expectEqual(x1.Reason.dead, s.x.last_reason);
}

test "X1 ghost reports a known evicted return without restoring or crossing namespaces" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{ .exact_end = true }, 1);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true };
    const r: RequestState = .{ .id = 1, .cohort = 1, .input_len = 3 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3}, 3, null, &.{}, &r));
    try std.testing.expect(s.keepRequestScoped(null, &.{8, 9, 10}, 3, null, &.{}, &r));
    const foreign = try s.lookupScoped(namespaceFor("other"), std.testing.allocator, &.{1, 2, 3}, 0, &.{}, &.{});
    defer std.testing.allocator.free(foreign.marks);
    try std.testing.expectEqual(@as(u64, 0), s.x.ghost_returns);
    const local = try s.lookupScoped(null, std.testing.allocator, &.{1, 2, 3}, 0, &.{}, &.{});
    defer std.testing.allocator.free(local.marks);
    try std.testing.expectEqual(@as(u64, 1), s.x.ghost_returns);
    try std.testing.expect(local.entry == null);
    try std.testing.expectEqual(@as(u64, 0), s.counts.hits);
}

test "X1 can disable retention without disabling plan; bounded entries fall back safely" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 600);
    defer s.deinit();
    for (0..513) |i| _ = try K2aFake.add(&s, &.{@intCast(i + 1)}, 1);
    s.x.enabled = true;
    try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{900}));
    try std.testing.expectEqual(@as(u64, 1), s.x.fallbacks);
    s.x.enabled = false;
    s.normalize_marks = true;
    try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{900}));
    try std.testing.expect(s.normalize_marks);
}

test "X1 failed save cannot supersede a usable fallback or invent restore credit" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 10);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true, .request_serial = 1 };
    const r: RequestState = .{ .id = 1, .cohort = 1, .input_len = 4, .encoder_at = 2 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4}, 2, null, &.{}, &r));
    const fallback = s.entries.items[0];
    vt.save = X1Grid.failSave;
    try std.testing.expect(!s.keepRequestScoped(null, &.{1, 2, 3, 4}, 4, null, &.{}, &r));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), fallback.x.successor);
    try std.testing.expectEqual(@as(u64, 0), fallback.reuse.restores);
    try std.testing.expectEqual(@as(u64, 1), s.x.serial);
}

const X2Hints = struct {
    fn hint(_: *anyopaque, saved: Saved, state: u64) void {
        const st: *K2aFake.State = @ptrCast(@alignCast(saved));
        st.hints += 1;
        st.hint_state = state;
    }
};

test "X2 actual keep and dedup forward verified continuation without fake restores" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.tail_window = 128;
    vt.continuation_hint = X2Hints.hint;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 100);
    defer s.deinit();
    s.x.enabled = true;
    const initial: RequestState = .{ .id = 1, .cohort = 1, .input_len = 4 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4}, 4, null, &.{}, &initial));
    const old = s.entries.items[0];
    try std.testing.expectEqual(@as(u64, 0), K2aFake.state(old).hints);
    var r: RequestState = .{ .id = 2, .cohort = 2, .input_len = 6 };
    s.resumedRequest(old, &.{1, 2, 3, 4, 5, 6}, true, &r);
    try std.testing.expect(r.continuation);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &r));
    const tail = s.entries.items[s.entries.items.len-1];
    try std.testing.expectEqual(@as(u64, 1), K2aFake.state(tail).hints);
    try std.testing.expectEqual(tail.x.id, K2aFake.state(tail).hint_state);
    try std.testing.expectEqual(@as(u64, 0), tail.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &r));
    try std.testing.expectEqual(@as(u64, 2), K2aFake.state(tail).hints);
    try std.testing.expectEqual(@as(u64, 0), tail.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6, 7, 8}, 8, null, &.{}, &r));
    const decoded = s.entries.items[s.entries.items.len-1];
    try std.testing.expectEqual(@as(u64, 1), K2aFake.state(decoded).hints);
    try std.testing.expectEqual(@as(u64, 0), decoded.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 2, null, &.{}, &r));
    try std.testing.expectEqual(@as(u64, 0), K2aFake.state(s.entries.items[s.entries.items.len-1]).hints);
}

test "X2 shared header and failed restore cannot create native continuation hints" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.tail_window = 128;
    vt.continuation_hint = X2Hints.hint;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 100);
    defer s.deinit();
    s.x.enabled = true;
    const initial: RequestState = .{ .id = 1, .cohort = 1, .input_len = 4 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4}, 4, null, &.{}, &initial));
    const old = s.entries.items[0];
    old.shared = true;
    var shared: RequestState = .{ .id = 2, .cohort = 2, .input_len = 6 };
    s.resumedRequest(old, &.{1, 2, 3, 4, 5, 6}, true, &shared);
    try std.testing.expect(!shared.continuation);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &shared));
    const tail = s.entries.items[s.entries.items.len-1];
    try std.testing.expectEqual(@as(u64, 0), K2aFake.state(tail).hints);
    var failed: RequestState = .{ .id = 3, .cohort = 3, .input_len = 8 };
    s.resumedRequest(tail, &.{1, 2, 3, 4, 5, 6, 7, 8}, false, &failed);
    try std.testing.expect(!failed.continuation);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6, 7, 8}, 8, null, &.{}, &failed));
    try std.testing.expectEqual(@as(u64, 0), K2aFake.state(s.entries.items[s.entries.items.len-1]).hints);
}

test "X2 no observational legacy scan every32 choices, but real fallback remains" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 4);
    defer s.deinit();
    s.x = .{ .enabled = true, .serial = 100, .request_serial = 100 };
    const cold = try K2aFake.add(&s, &.{1}, 1);
    cold.x.id = 1;
    const hot = try K2aFake.add(&s, &.{2}, 1);
    hot.x = .{ .id = 2, .continuation = true, .last_return = 100 };
    hot.reuse.restores = 3;
    for (0..65) |_| try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
    try std.testing.expectEqual(@as(u64, 0), s.x.sampled_choices);
    try std.testing.expectEqual(@as(u64, 0), s.x.sampled_different);
    try std.testing.expectEqual(@as(u64, 0), s.audit.shadow_decisions);
    try std.testing.expectEqual(@as(u64, 0), s.x.fallbacks);
    // Exercise functional fallback directly; it must still produce a victim.
    try std.testing.expect(s.xFallback(null, null, &.{9}) != null);
    try std.testing.expectEqual(@as(u64, 1), s.x.fallbacks);
}

test "X3 actual cache transfers learned lineage without fake restore counts" {
    const a = std.testing.allocator;
    var k: K2aFake = .{};
    var s = Store.init(a, k.family(true), .{ .planned=false, .exact_end=true }, 32);
    defer s.deinit();
    s.x.enabled = true;
    s.adaptive = true;
    const first = [_]u32{1,2,3,4};
    var r: RequestState = .{ .id=1, .namespace_id=Store.sharedKey(null,&.{}), .input_len=4 };
    s.x.request_serial=1;
    r.history=s.history.begin(r.namespace_id,1);
    try std.testing.expect(s.keepRequestScoped(null,&first,4,null,&.{},&r));
    const e=s.entries.items[0];
    const root=e.x.history;
    var next: RequestState = .{ .id=2, .namespace_id=r.namespace_id, .input_len=5 };
    s.x.request_serial=2;
    next.history=s.history.begin(next.namespace_id,2);
    const longer=[_]u32{1,2,3,4,5};
    s.resumedRequest(e,&longer,true,&next);
    try std.testing.expect(next.continuation);
    try std.testing.expectEqual(root,next.history);
    try std.testing.expectEqual(@as(u64,1),e.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null,&longer,5,null,&.{},&next));
    const tail=s.entries.items[s.entries.items.len-1];
    try std.testing.expectEqual(root,tail.x.history);
    try std.testing.expectEqual(@as(u64,0),tail.reuse.restores);
    s.completedRequest(next,4,1000,500,1);
    try std.testing.expectEqual(@as(u64,2),s.history.get(r.namespace_id,root).?.completed);
}

test "X3 failed restore cannot promote a branch; reset clears learning" {
    const a = std.testing.allocator;
    var k: K2aFake = .{};
    var s=Store.init(a,k.family(true),.{ .planned=false,.exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const ns=Store.sharedKey(null,&.{});
    var r: RequestState=.{ .id=1,.namespace_id=ns,.input_len=4 };
    r.history=s.history.begin(ns,1);
    const tokens=[_]u32{1,2,3,4};
    try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&r));
    var next: RequestState=.{ .id=2,.namespace_id=ns,.input_len=5 };
    next.history=s.history.begin(ns,2);
    const longer=[_]u32{1,2,3,4,5};
    s.resumedRequest(s.entries.items[0],&longer,false,&next);
    try std.testing.expect(!next.continuation and next.history==2);
    try std.testing.expectEqual(@as(u64,0),s.history.counters.returns);
    try s.clear();
    try std.testing.expectEqual(@as(u64,0),s.history.counters.requests);
    try std.testing.expect(s.history.get(ns,1)==null);
}

test "X3 actual ghost learns without tracing, never restores or crosses namespaces" {
    const a=std.testing.allocator;
    var k: K2aFake= .{};
    var s=Store.init(a,k.family(true),.{ .exact_end=true },1);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const ns=Store.sharedKey(null,&.{});
    var first: RequestState=.{ .id=1,.namespace_id=ns,.input_len=3 };
    s.x.request_serial=1;
    first.history=s.history.begin(ns,1);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3},3,null,&.{},&first));
    var second: RequestState=.{ .id=2,.namespace_id=ns,.input_len=3 };
    s.x.request_serial=2;
    second.history=s.history.begin(ns,2);
    try std.testing.expect(s.keepRequestScoped(null,&.{8,9,10},3,null,&.{},&second));
    const foreign=try s.lookupScoped(namespaceFor("other"),a,&.{1,2,3},0,&.{},&.{});
    defer a.free(foreign.marks);
    try std.testing.expectEqual(@as(u64,0),s.history.counters.ghosts);
    const wrong=try s.lookupScoped(null,a,&.{9,2,3},0,&.{},&.{});
    defer a.free(wrong.marks);
    try std.testing.expectEqual(@as(u64,0),s.history.counters.ghosts);
    const local=try s.lookupScoped(null,a,&.{1,2,3},0,&.{},&.{});
    defer a.free(local.marks);
    try std.testing.expectEqual(@as(u64,1),s.history.counters.ghosts);
    try std.testing.expectEqual(first.history,local.request.history);
    try std.testing.expect(local.entry==null);
    try std.testing.expectEqual(@as(u64,0),s.counts.hits);
}

test "X3 long ghost plus short restore is one learned return, not a fork" {
    const a=std.testing.allocator;
    var k: K2aFake= .{};
    var s=Store.init(a,k.family(true),.{ .exact_end=true },8);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const ns=Store.sharedKey(null,&.{});
    var first: RequestState=.{ .id=1,.namespace_id=ns,.input_len=3 };
    s.x.request_serial=1;
    first.history=s.history.begin(ns,1);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3},3,null,&.{},&first));
    var second: RequestState=.{ .id=2,.namespace_id=ns,.input_len=5 };
    s.x.request_serial=2;
    second.history=s.history.begin(ns,2);
    s.resumedRequest(s.entries.items[0],&.{1,2,3,4,5},true,&second);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3,4,5},5,null,&.{},&second));
    s.x.last_reason= .dead;
    s.remove(1); // physical long endpoint gone; short fallback still live
    // Exercise the harder variant: the historical fallback has its own exact
    // old tail, so an ordinary restore would independently qualify as a return.
    a.free(s.entries.items[0].last);
    s.entries.items[0].last=try a.dupe(u32,&.{1,2,3});
    var found=try s.lookupScoped(null,a,&.{1,2,3,4,5,6},0,&.{},&.{});
    defer a.free(found.marks);
    try std.testing.expect(found.entry!=null and found.request.history_observed);
    const before=s.history.counters.returns;
    const history=found.request.history;
    s.resumedRequest(found.entry.?,&.{1,2,3,4,5,6},true,&found.request);
    try std.testing.expect(found.request.continuation);
    try std.testing.expectEqual(history,found.request.history);
    try std.testing.expectEqual(before,s.history.counters.returns);
    try std.testing.expectEqual(@as(u64,0),s.history.counters.forks);
}

// Independent review probes appended only to an owned COPY of the sealed source.
test "X3 fixed review identical sequential replays are not conversation forks" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var s=Store.init(a,f.family(true),.{ .exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const tokens=[_]u32{1,2,3,4};
    var ids: [4]u64=undefined;
    for (0..4) |i| {
        var l=try s.lookupScoped(null,a,&tokens,0,&.{},&.{});
        defer a.free(l.marks);
        if (l.entry) |e| s.resumedRequest(e,&tokens,true,&l.request);
        try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&l.request));
        s.completedRequest(l.request,l.request.restored_at,1000,500,1);
        ids[i]=l.request.history;
    }
    std.debug.print("REVIEW exact_replays history={any} restores={d} returns={d} forks={d}\n",.{ids,s.counts.hits,s.history.counters.returns,s.history.counters.forks});
    try std.testing.expectEqual(@as(u64,0),s.history.counters.forks);
    try std.testing.expectEqual(@as(u64,1),s.entries.items[0].x.created_request);
    try std.testing.expectEqual(@as(u64,4),s.entries.items[0].x.history_request);
}

test "X3 fixed review first return to physically stale prefix learns its lost history" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var s=Store.init(a,f.family(true),.{ .exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const tokens=[_]u32{1,2,3,4};
    const first=try s.lookupScoped(null,a,&tokens,0,&.{},&.{});
    defer a.free(first.marks);
    try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&first.request));
    const previous_history=first.request.history;
    // Native placement invalidates storage; core removal happens later.
    K2aFake.state(s.entries.items[0]).live=false;
    const next_tokens=[_]u32{1,2,3,4,5};
    const next=try s.lookupScoped(null,a,&next_tokens,0,&.{},&.{});
    defer a.free(next.marks);
    try std.testing.expect(next.entry==null);
    const before=s.history.counters.ghosts;
    try std.testing.expect(s.keepRequestScoped(null,&next_tokens,5,null,&.{},&next.request));
    const third=try s.lookupScoped(null,a,&.{1,2,3,4,5,6},0,&.{},&.{});
    defer a.free(third.marks);
    std.debug.print("REVIEW physical_return original={d} next_history={d} observed={} ghosts_before_keep={d} ghosts_next_lookup={d} dead_pruned={d}\n",.{previous_history,next.request.history,next.request.history_observed,before,s.history.counters.ghosts,s.audit.dead_at_keep});
    try std.testing.expect(next.request.history_observed);
    try std.testing.expectEqual(previous_history,next.request.history);
}

const ReviewHint = struct {
    var latest: u32=0;
    var calls: u32=0;
    fn hint(_: *anyopaque,_:Saved,density:u32) void { latest=density;calls+=1; }
};
test "X3 fixed review converting a private checkpoint to shared clears native preference" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var vt=f.family(true).vtable.*;
    vt.adaptive_hint=ReviewHint.hint;
    var s=Store.init(a,.{ .ptr=&f,.vtable=&vt },.{ .exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    ReviewHint.latest=0;ReviewHint.calls=0;
    const tokens=[_]u32{1,2,3,4};
    const first=try s.lookupScoped(null,a,&tokens,0,&.{},&.{});
    defer a.free(first.marks);
    try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&first.request));
    const before=ReviewHint.latest;
    try std.testing.expect(before>0);
    // A later planned shared cut turns this deduplicated physical key shared.
    s.noteShared(null,&tokens);
    try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&first.request));
    try std.testing.expect(s.entries.items[0].shared);
    std.debug.print("REVIEW shared_conversion before_density={d} after_density={d} callback_count={d}\n",.{before,ReviewHint.latest,ReviewHint.calls});
    try std.testing.expectEqual(@as(u32,0),ReviewHint.latest);
    try std.testing.expectEqual(@as(u64,0),s.entries.items[0].x.history);
}

test "X3 replay of an older endpoint remains reuse then growth follows its learning head" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var s=Store.init(a,f.family(true),.{ .exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const inputs=[_][]const u32{&.{1,2,3,4},&.{1,2,3,4,5,6},&.{1,2,3,4},&.{1,2,3,4,9}};
    var root: u64=0;
    for(inputs,0..) |tokens,i| {
        var l=try s.lookupScoped(null,a,tokens,0,&.{},&.{});
        defer a.free(l.marks);
        if(l.entry) |e| s.resumedRequest(e,tokens,true,&l.request);
        try std.testing.expect(s.keepRequestScoped(null,tokens,@intCast(tokens.len),null,&.{},&l.request));
        if(i==0) root=l.request.history;
        try std.testing.expectEqual(root,l.request.history);
    }
    try std.testing.expectEqual(@as(u64,0),s.history.counters.forks);
    try std.testing.expectEqual(@as(u64,3),s.history.counters.returns);
}

test "X3 stale evidence is single use and does not free snapshots during lookup" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var s=Store.init(a,f.family(true),.{ .exact_end=true },32);
    defer s.deinit(); s.x.enabled=true; s.adaptive=true;
    const first=try s.lookupScoped(null,a,&.{1,2,3,4},0,&.{},&.{});
    defer a.free(first.marks);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3,4},4,null,&.{},&first.request));
    K2aFake.state(s.entries.items[0]).live=false;
    const wrong=try s.lookupScoped(null,a,&.{9,2,3,4,5},0,&.{},&.{});
    defer a.free(wrong.marks);
    try std.testing.expect(!wrong.request.history_observed);
    try std.testing.expectEqual(@as(usize,1),s.entries.items.len);
    const good=try s.lookupScoped(null,a,&.{1,2,3,4,5},0,&.{},&.{});
    defer a.free(good.marks);
    try std.testing.expect(good.request.history_observed);
    try std.testing.expectEqual(first.request.history,good.request.history);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3,4,5},5,null,&.{},&good.request));
    const again=try s.lookupScoped(null,a,&.{1,2,3,4,6},0,&.{},&.{});
    defer a.free(again.marks);
    try std.testing.expect(!again.request.history_observed);
    try std.testing.expectEqual(@as(u64,1),s.history.counters.ghosts);
}

test "X3 shared promotion clears native preference even without request metadata" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var vt=f.family(true).vtable.*; vt.adaptive_hint=ReviewHint.hint;
    var s=Store.init(a,.{ .ptr=&f,.vtable=&vt },.{ .exact_end=true },32);
    defer s.deinit(); s.x.enabled=true; s.adaptive=true;
    ReviewHint.latest=0;
    const first=try s.lookupScoped(null,a,&.{1,2,3,4},0,&.{},&.{});
    defer a.free(first.marks);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3,4},4,null,&.{},&first.request));
    try std.testing.expect(ReviewHint.latest>0);
    s.noteShared(null,&.{1,2,3,4});
    try std.testing.expect(s.keepScoped(null,&.{1,2,3,4},4,null,&.{}));
    try std.testing.expectEqual(@as(u32,0),ReviewHint.latest);
    try std.testing.expectEqual(@as(u64,0),s.entries.items[0].x.history);
}


test "X4 diagnostic ghost excludes a length that cannot be natively resumed" {
    var fake:K2aFake=.{};
    var vt=fake.family(true).vtable.*;
    vt.ghost_range=struct {
        fn range(_: *anyopaque,_:Saved) [2]u32 { return .{3,std.math.maxInt(u32)}; }
    }.range;
    var s=Store.init(std.testing.allocator,.{.ptr=&fake,.vtable=&vt},.{.exact_end=true},8);
    defer s.deinit();
    s.x.trace=true;
    const old=try K2aFake.add(&s,&.{1,2,3},3);
    old.x.id=1;old.x.key=Store.sharedKey(null,old.tokens);old.x.role=.prompt_end;
    s.xLost(old,.dead);
    var request:RequestState=.{.id=2,.namespace_id=Store.sharedKey(null,&.{}),.input_len=4};
    s.xGhostReturns(null,&.{1,2,3,4},0,&request);
    try std.testing.expectEqual(@as(u64,0),s.x.ghost_returns);
    try std.testing.expectEqual(@as(u64,1),s.x.ghost_incompatible_skips);
    // It is still a valid diagnostic for a future exact replay, not discarded.
    request.input_len=3;
    s.xGhostReturns(null,&.{1,2,3},0,&request);
    try std.testing.expectEqual(@as(u64,1),s.x.ghost_returns);
}


test "adopted terminal gets X1 X2 and X3 seed before first reuse" {
    const a = std.testing.allocator;
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.tail_window = 128;
    vt.continuation_hint = X2Hints.hint;
    var s = Store.init(a, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 32);
    defer s.deinit();
    s.x.enabled = true;
    s.adaptive = true;
    const saved = try a.create(K2aFake.State);
    saved.* = .{ .at = 4 };
    var key = [_]u32{ 1, 2, 3, 4 };
    var restore: Restore = .{ .point = .{ .id = 20, .at = 4, .tokens = 4 }, .job = 0, .key = &key };
    try std.testing.expect(s.adopt(&restore, saved, 1, &.{}));
    const e = s.entries.items[0];
    try std.testing.expectEqual(x1.Role.prompt_end, e.x.role);
    try std.testing.expect(e.x.id != 0 and e.x.created_request != 0 and e.x.history != 0);
    try std.testing.expectEqual(e.x.created_request, e.x.history_request);
    try std.testing.expectEqual(@as(u64, 0), e.reuse.restores);
    const root = e.x.history;
    var l = try s.lookupScoped(null, a, &.{ 1, 2, 3, 4, 5, 6 }, 0, &.{}, &.{});
    defer a.free(l.marks);
    try std.testing.expect(l.entry == e);
    s.resumedRequest(e, &.{ 1, 2, 3, 4, 5, 6 }, true, &l.request);
    try std.testing.expect(l.request.continuation);
    try std.testing.expectEqual(root, l.request.history);
    try std.testing.expect(s.keepRequestScoped(null, &.{ 1, 2, 3, 4, 5, 6 }, 6, null, &.{}, &l.request));
    const next = s.entries.items[s.entries.items.len - 1];
    try std.testing.expectEqual(root, next.x.history);
    try std.testing.expectEqual(next.x.id, K2aFake.state(next).hint_state);
}

test "failed resume never becomes a physical-loss ghost" {
    const a = std.testing.allocator;
    var f: K2aFake = .{};
    var s = Store.init(a, f.family(true), .{ .exact_end = true }, 32);
    defer s.deinit();
    s.x.enabled = true;
    s.x.trace = true;
    s.adaptive = true;
    const ns = Store.sharedKey(null, &.{});
    var request: RequestState = .{ .id = 1, .cohort = 1, .input_len = 4, .namespace_id = ns };
    request.history = s.history.begin(ns, 1);
    s.x.request_serial = 1;
    try std.testing.expect(s.keepRequestScoped(null, &.{ 1, 2, 3, 4 }, 4, null, &.{}, &request));
    s.x.last_reason = .dead; // previous eviction must not classify this restore
    s.resumed(s.entries.items[0], &.{ 1, 2, 3, 4, 5 }, false);
    try std.testing.expectEqual(x1.Reason.restore_failed, s.x.last_reason);
    try std.testing.expectEqual(x1.Reason.restore_failed, s.x.events[s.x.event_count - 1].reason);
    try std.testing.expectEqual(@as(u64, 0), s.x.ghost_returns);
    for (s.x.ghosts) |ghost| try std.testing.expectEqual(@as(u64, 0), ghost.state);
    try std.testing.expectEqual(@as(u64, 0), s.history.get(ns, request.history).?.last_eviction);
}

test "image reservation excludes absent terminal and frees fitted marks on OOM" {
    const a = std.testing.allocator;
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.automatic_marks = X1Grid.automatic;
    var s = Store.init(a, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 32);
    defer s.deinit();
    s.normalize_marks = true;
    const text = try s.reservationMarks(a, &.{ 1, 2, 3, 4 }, null, &.{}, false);
    defer a.free(text);
    try std.testing.expectEqualSlices(u32, &.{ 4 }, text); // encoder mark is 0 here
    const image = try s.reservationMarks(a, &.{ 1, 2, 3, 4 }, null, &.{}, true);
    defer a.free(image);
    try std.testing.expectEqualSlices(u32, &.{}, image);
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 1, .resize_fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory,
        s.reservationMarks(failing.allocator(), &.{ 1, 2, 3, 4 }, null, &.{ 2 }, false));
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}

test "image cache retains full digest identity even for identical shortened token keys" {
    var f: K2aFake = .{};
    var store = Store.init(std.testing.allocator, f.family(false), .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    // Deliberately collide the shortened ids. The RAM tier must distinguish full picture digests.
    const key = [_]u32{ 1, 2, 0x8000_0001, 0x8000_0002, 0x8000_0003, 5, 6 };
    const a: [32]u8 = @splat(1);
    const b: [32]u8 = @splat(2);
    const first = [_]Media{.{ .start = 2, .len = 3, .digest = a }};
    const other = [_]Media{.{ .start = 2, .len = 3, .digest = b }};
    try std.testing.expect(store.keepWith(&key, 5, null, &.{}, .{ .images = true, .media = &first }));
    const first_entry = store.findMediaScoped(null, &key, &.{}, &first).?;
    try std.testing.expect(store.findMediaScoped(null, &key, &.{}, &other) == null);
    try std.testing.expect(store.find(&key, &.{}) == null);
    try std.testing.expect(store.keepWith(&key, 5, null, &.{}, .{ .images = true, .media = &other }));
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);
    try std.testing.expect(store.findMediaScoped(null, &key, &.{}, &first) == first_entry);
    try std.testing.expect(store.findMediaScoped(null, &key, &.{}, &other) != first_entry);
}
const BatchFake401 = struct {
    var calls: usize = 0;
    var max_count: usize = 0;
    var store: ?*Store = null;
    fn many(ptr:*anyopaque,saved:[]const Saved) void {
        calls+=1;max_count=@max(max_count,saved.len);
        // All bytes remain charged while the backend owns these buffers.
        std.debug.assert(store.?.held >= saved.len);
        for(saved) |p| K2aFake.drop_(ptr,p);
    }
};
test "401a dead batches preserve live entries, budget and exactly-once frees" {
    var f:K2aFake=.{};
    var vt=f.family(true).vtable.*;vt.drop_many=BatchFake401.many;
    var s=Store.init(std.testing.allocator,.{.ptr=&f,.vtable=&vt},.{},128);
    defer s.deinit();s.batch_dead=true;
    BatchFake401.store=&s;defer BatchFake401.store=null;
    BatchFake401.calls=0;BatchFake401.max_count=0;
    const survivor=try K2aFake.add(&s,&.{7},1);
    for(0..35) |_| {const e=try K2aFake.add(&s,&.{9},1);K2aFake.state(e).live=false;}
    while(s.removeDeadRun(1)) {}
    try std.testing.expectEqual(@as(usize,1),s.entries.items.len);
    try std.testing.expect(s.entries.items[0]==survivor);
    try std.testing.expectEqual(@as(usize,35),f.drops);
    try std.testing.expectEqual(@as(usize,3),BatchFake401.calls);
    try std.testing.expectEqual(@as(usize,16),BatchFake401.max_count);
    try std.testing.expectEqual(@as(u64,1),s.held);
    try std.testing.expectEqual(@as(u64,35),s.audit.dead_at_keep);
}
test "401a disabled or unsupported batch release keeps production behavior" {
    var f:K2aFake=.{};
    var s=Store.init(std.testing.allocator,f.family(true),.{},128);defer s.deinit();
    for(0..2) |_| {const e=try K2aFake.add(&s,&.{9},1);K2aFake.state(e).live=false;}
    try std.testing.expect(!s.removeDeadRun(0));s.batch_dead=true;
    try std.testing.expect(!s.removeDeadRun(0));
    try std.testing.expectEqual(@as(usize,0),f.drops);
}

const FirstFake401=struct {
    var calls:usize=0;
    fn hint(_: *anyopaque,_:Saved) void { calls+=1; }
};
test "401a first hint requires success and fresh unshared terminal history" {
    var f:K2aFake=.{};
    var vt=f.family(true).vtable.*;vt.first_hint=FirstFake401.hint;
    var s=Store.init(std.testing.allocator,.{.ptr=&f,.vtable=&vt},.{},128);
    defer s.deinit();s.adaptive=true;s.first_continuation=true;
    _=s.history.begin(1,1);
    const e=try K2aFake.add(&s,&.{1,2},2);
    e.x.created_request=1;e.x.role=.decode_end;
    const r:RequestState=.{.id=1,.namespace_id=1,.history=1,.input_len=2};
    FirstFake401.calls=0;
    s.completedRequest(r,0,0,0,3); // cancelled
    try std.testing.expectEqual(@as(usize,0),FirstFake401.calls);
    s.completedRequest(r,0,0,0,1);
    try std.testing.expectEqual(@as(usize,1),FirstFake401.calls);
    e.shared=true;s.completedRequest(r,0,0,0,1);
    e.shared=false;e.x.role=.auxiliary;s.completedRequest(r,0,0,0,1);
    e.x.role=.decode_end;_=s.history.returned(1,2,1,1,false,false);
    s.completedRequest(r,0,0,0,1);
    try std.testing.expectEqual(@as(usize,1),FirstFake401.calls);
}

test "cache batch dead run stops at scoped borrowers and retains their accounting" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.drop_many = BatchFake401.many;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{}, 128);
    defer s.deinit();
    s.batch_dead = true;
    BatchFake401.store = &s;
    defer BatchFake401.store = null;
    BatchFake401.calls = 0;
    const borrowed = try K2aFake.add(&s, &.{1}, 1);
    K2aFake.state(borrowed).live = false;
    s.pin(borrowed);
    for (0..2) |_| { const e = try K2aFake.add(&s, &.{9}, 1); K2aFake.state(e).live = false; }
    try std.testing.expect(!s.removeDeadRun(0));
    try std.testing.expect(s.removeDeadRun(1));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 1), s.held);
    try std.testing.expectEqual(@as(usize, 2), f.drops);
    s.remove(0);
    try std.testing.expectEqual(@as(u64, 1), s.held);
    s.unpin(borrowed);
    try std.testing.expectEqual(@as(u64, 0), s.held);
    try std.testing.expectEqual(@as(usize, 3), f.drops);
}
