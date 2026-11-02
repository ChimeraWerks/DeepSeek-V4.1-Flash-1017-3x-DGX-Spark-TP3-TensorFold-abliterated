//! Host-only retained-context health. Sticky CUresults match TensorFold v1.1.4.
//! Poisoned entries remain tombstones until process exit: reopening cannot clear a fault.
const std = @import("std");
const abi = @import("abi.zig");

const Entry = struct {
    driver: usize,
    context: abi.Context,
    ordinal: ?c_int = null,
    refs: usize = 1,
    fault: abi.Result = 0,
    next: ?*Entry = null,
    generation: usize = 1,
    admission: std.atomic.Mutex = .unlocked,
    gate: ?*Entry = null,
};
var entries: ?*Entry = null;
var mutex: std.atomic.Mutex = .unlocked;
threadlocal var current_driver: usize = 0;
threadlocal var current_context: abi.Context = null;

fn lock() void {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
}
fn find(driver: usize, context: abi.Context) ?*Entry {
    var cursor = entries;
    while (cursor) |entry| : (cursor = entry.next) {
        if (entry.driver == driver and entry.context == context) return entry;
    }
    return null;
}

pub fn sticky(code: abi.Result) bool {
    return switch (code) {
        214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719 => true,
        else => false,
    };
}

/// No fixed registry capacity. Call before publishing or binding a retained context.
pub fn retain(driver: usize, context: abi.Context) error{OutOfMemory, CudaContextLost}!void {
    lock();
    defer mutex.unlock();
    var shared = entries;
    while (shared) |entry| : (shared = entry.next) {
        if (entry.context == context and entry.fault != 0) return error.CudaContextLost;
    }
    if (find(driver, context)) |entry| {
        if (entry.fault != 0) return error.CudaContextLost;
        if (entry.refs == 0) entry.generation += 1;
        entry.refs += 1;
        return;
    }
    const entry = std.heap.page_allocator.create(Entry) catch return error.OutOfMemory;
    entry.* = .{ .driver = driver, .context = context, .next = entries };
    var peer = entries;
    while (peer) |e| : (peer = e.next) {
        if (e.context == context) { entry.gate = e.gate orelse e; break; }
    }
    entries = entry;
}

/// Record primary-device identity after retain, before any bind. CUDA primary
/// contexts are shared across wrappers, so ordinal poison is process-wide.
pub fn primary(driver: usize, context: abi.Context, ordinal: c_int) void {
    lock();
    defer mutex.unlock();
    if (find(driver, context)) |entry| entry.ordinal = ordinal;
}
pub fn primaryHealthy(ordinal: c_int) bool {
    lock();
    defer mutex.unlock();
    var cursor = entries;
    while (cursor) |entry| : (cursor = entry.next) {
        if (entry.ordinal == ordinal and entry.fault != 0) return false;
    }
    return true;
}

/// Host bookkeeping only; never calls CUDA. Poison tombstones are intentionally retained.
pub fn release(driver: usize, context: abi.Context) void {
    lock();
    defer mutex.unlock();
    var link = &entries;
    while (link.*) |entry| {
        if (entry.driver == driver and entry.context == context) {
            if (entry.refs > 0) entry.refs -= 1;
            return;
        }
        link = &entry.next;
    }
}

pub fn observe(_: usize, context: abi.Context, code: abi.Result) void {
    if (!sticky(code)) return;
    const gate = gateFor(context) orelse return;
    spin(&gate.admission);
    defer gate.admission.unlock();
    publish(context, code);
}
fn publish(context: abi.Context, code: abi.Result) void {
    if (!sticky(code)) return;
    lock();
    defer mutex.unlock();
    var cursor = entries;
    while (cursor) |entry| : (cursor = entry.next) {
        if (entry.context == context and entry.fault == 0) entry.fault = code;
    }
}
fn spin(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) std.Thread.yield() catch {};
}
fn gateFor(context: abi.Context) ?*Entry {
    lock();
    defer mutex.unlock();
    var cursor = entries;
    while (cursor) |e| : (cursor = e.next) {
        if (e.context == context) return e.gate orelse e;
    }
    return null;
}

/// One non-reentrant boundary: check, foreign call, observation, unlock.
/// Registry mutex is never held while waiting for admission or entering a library.
pub const Admission = struct {
    gate: ?*Entry,
    context: abi.Context,
    pub fn observe(self: Admission, code: abi.Result) void { publish(self.context, code); }
    pub fn end(self: Admission) void { if (self.gate) |g| g.admission.unlock(); }
};
pub fn admitCurrent() error{CudaContextLost}!Admission {
    return Owner.borrowCurrent().admit();
}
pub fn admitContext(driver: usize, context: abi.Context) error{CudaContextLost}!Admission {
    lock();
    const e = find(driver, context);
    const o: Owner = if (e) |v| .{ .entry = v, .generation = v.generation } else .{};
    mutex.unlock();
    return o.admitUnbound();
}

/// A poisoned context may still own callbacks/resources in this library. Keep it loaded.
pub fn driverHealthy(driver: usize) bool {
    lock();
    defer mutex.unlock();
    var cursor = entries;
    while (cursor) |entry| : (cursor = entry.next) {
        if (entry.driver == driver and (entry.fault != 0 or entry.refs > 0)) return false;
    }
    return true;
}

/// Pure host read, suitable for readiness even if a peer probe is disabled.
pub fn healthy(driver: usize, context: abi.Context) bool {
    return fault(driver, context) == 0;
}

pub fn fault(driver: usize, context: abi.Context) abi.Result {
    lock();
    defer mutex.unlock();
    return if (find(driver, context)) |entry| entry.fault else 0;
}

/// Publish ONLY after successful cuCtxSetCurrent. A failed bind must not change this.
pub fn setCurrent(driver: usize, context: abi.Context) void {
    current_driver = driver;
    current_context = context;
}
pub fn clearCurrent(driver: usize, context: abi.Context) void {
    if (current_driver == driver and current_context == context) setCurrent(0, null);
}
pub fn observeCurrent(driver: usize, code: abi.Result) void {
    observe(driver, current_context, code);
}
pub fn currentHealthy(_: usize) bool {
    return boundHealthy();
}

/// Library wrappers without a Driver still share the thread's actual context.
/// Compare context identity, not wrapper address, before evaluating any API.
pub fn boundHealthy() bool {
    lock();
    defer mutex.unlock();
    var cursor = entries;
    while (cursor) |entry| : (cursor = entry.next) {
        if (entry.context == current_context and entry.fault != 0) return false;
    }
    return true;
}

pub fn ensureBoundHealthy() error{CudaContextLost}!void {
    if (!boundHealthy()) return error.CudaContextLost;
}

test "library preflight follows context identity across driver wrappers" {
    const context: abi.Context = @ptrFromInt(0xbeef9202);
    try retain(0xbeef9201, context);
    setCurrent(0xbeef9203, context);
    defer clearCurrent(0xbeef9203, context);
    observe(0xbeef9201, context, 700);
    try std.testing.expectError(error.CudaContextLost, ensureBoundHealthy());
    try std.testing.expect(!currentHealthy(0xbeef9203));
}

test "sticky list is exact; recovery codes never poison" {
    for ([_]abi.Result{ 214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719 }) |code| try std.testing.expect(sticky(code));
    for ([_]abi.Result{ 0, 1, 2, 3, 4, 201, 500, 600, 701, 800, 801, 999, 12345 }) |code| try std.testing.expect(!sticky(code));
}

test "first fault survives success, release and reopen; other contexts remain healthy" {
    const driver = 0x9a01;
    const first: abi.Context = @ptrFromInt(0x9a02);
    const other: abi.Context = @ptrFromInt(0x9a03);
    try retain(driver, first);
    try retain(driver, first);
    try retain(driver, other);
    defer release(driver, other);
    for ([_]abi.Result{ 2, 600, 701, 800, 999 }) |code| observe(driver, first, code);
    try std.testing.expect(healthy(driver, first));
    observe(driver, first, 700);
    observe(driver, first, 719);
    observe(driver, first, 0);
    release(driver, first);
    release(driver, first);
    try std.testing.expectEqual(@as(abi.Result, 700), fault(driver, first));
    try std.testing.expectError(error.CudaContextLost, retain(driver, first));
    try std.testing.expect(healthy(driver, other));
}

test "failed bind faults target without publishing it as current" {
    const driver = 0x9b01;
    const old: abi.Context = @ptrFromInt(0x9b02);
    const target: abi.Context = @ptrFromInt(0x9b03);
    try retain(driver, old);
    defer release(driver, old);
    try retain(driver, target);
    defer release(driver, target);
    setCurrent(driver, old);
    defer clearCurrent(driver, old);
    observe(driver, target, 709); // failed cuCtxSetCurrent(target)
    try std.testing.expect(current_context == old);
    try std.testing.expect(currentHealthy(driver));
    try std.testing.expect(!healthy(driver, target));
    observeCurrent(driver + 1, 700);
    try std.testing.expect(!healthy(driver, old));
}

test "shared primary context poison crosses driver wrappers" {
    const ctx: abi.Context = @ptrFromInt(0x9d02);
    try retain(0x9d01, ctx);
    try retain(0x9d03, ctx);
    observe(0x9d01, ctx, 700);
    try std.testing.expect(!healthy(0x9d03, ctx));
    try std.testing.expectError(error.CudaContextLost, retain(0x9d03, ctx));
}

/// Resource identity is captured at construction, never inferred from the later TLS binding.
/// Context registry entries remain retained by the native context owner until resource teardown.
pub const Owner = struct {
    entry: ?*Entry = null,
    generation: usize = 0,
    context_free: bool = false,

    /// Only API families documented as context-independent may opt into this.
    pub fn contextFree() Owner { return .{ .context_free = true }; }
    pub fn borrowCurrent() Owner {
        lock();
        defer mutex.unlock();
        const e = find(current_driver, current_context) orelse return .{};
        return .{ .entry = e, .generation = e.generation };
    }
    pub fn capture() Owner {
        var o = borrowCurrent();
        o = o.retain() catch return .{};
        return o;
    }
    pub fn retain(self: Owner) error{CudaContextLost}!Owner {
        lock();
        defer mutex.unlock();
        if (self.context_free) return self;
        const e = self.entry orelse return error.CudaContextLost;
        if (e.refs == 0 or e.generation != self.generation or e.fault != 0) return error.CudaContextLost;
        e.refs += 1;
        return self;
    }
    pub fn release(self: *Owner) void {
        lock();
        defer mutex.unlock();
        if (self.entry) |e| if (e.generation == self.generation and e.refs > 0) { e.refs -= 1; };
        self.* = .{};
    }
    fn valid(self: Owner, bound: bool) bool {
        if (self.context_free) return true;
        const e = self.entry orelse return false;
        if (e.refs == 0 or e.generation != self.generation or e.fault != 0) return false;
        if (bound and current_context != e.context) return false;
        var cursor = entries;
        while (cursor) |other| : (cursor = other.next) {
            if (other.context == e.context and other.fault != 0) return false;
        }
        return true;
    }
    pub fn ensure(self: Owner) error{CudaContextLost}!void {
        lock();
        defer mutex.unlock();
        if (!self.valid(true)) return error.CudaContextLost;
    }
    pub fn admit(self: Owner) error{CudaContextLost}!Admission { return self.begin(true); }
    pub fn admitUnbound(self: Owner) error{CudaContextLost}!Admission { return self.begin(false); }
    fn begin(self: Owner, bound: bool) error{CudaContextLost}!Admission {
        const g = if (self.entry) |e| e.gate orelse e else null;
        if (g) |gate| spin(&gate.admission);
        lock();
        const ok = self.valid(bound);
        mutex.unlock();
        if (!ok) {
            if (g) |gate| gate.admission.unlock();
            return error.CudaContextLost;
        }
        return .{ .gate = g, .context = if (self.entry) |e| e.context else null };
    }
};

test "poison resource owner rejects healthy foreign current context" {
    const a: abi.Context = @ptrFromInt(0xa901);
    const b: abi.Context = @ptrFromInt(0xa902);
    try retain(0xa900, a);
    try retain(0xa900, b);
    defer release(0xa900, b);
    setCurrent(0xa900, a);
    const owner = Owner.capture();
    observe(0xa900, a, 700);
    setCurrent(0xa900, b);
    defer clearCurrent(0xa900, b);
    try std.testing.expectError(error.CudaContextLost, owner.ensure());
    try ensureBoundHealthy();
}

test "admission serializes poison across wrappers and rejects later calls" {
    const Probe = struct {
        entered: std.atomic.Value(bool) = .init(false),
        attempted: std.atomic.Value(bool) = .init(false),
        blocked: std.atomic.Value(bool) = .init(false),
        context: abi.Context,
        fn poison(p: *@This()) void {
            while (!p.entered.load(.acquire)) std.Thread.yield() catch {};
            const gate = gateFor(p.context).?;
            const got = gate.admission.tryLock();
            if (got) gate.admission.unlock();
            p.blocked.store(!got, .release);
            p.attempted.store(true, .release);
            observe(0xb102, p.context, 700);
        }
    };
    const ctx: abi.Context = @ptrFromInt(0xb101);
    try retain(0xb102, ctx);
    try retain(0xb103, ctx);
    setCurrent(0xb102, ctx);
    defer clearCurrent(0xb102, ctx);
    var owner = Owner.capture();
    defer owner.release();
    var p: Probe = .{ .context = ctx };
    const t = try std.Thread.spawn(.{}, Probe.poison, .{&p});
    const call = try owner.admit();
    p.entered.store(true, .release);
    while (!p.attempted.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(p.blocked.load(.acquire));
    // The admitted call linearizes before publication; observation does not reacquire admission.
    call.observe(0);
    call.end();
    t.join();
    try std.testing.expectError(error.CudaContextLost, owner.admit());
    try std.testing.expectError(error.CudaContextLost, admitContext(0xb103, ctx));
}

test "owner reference survives context release and stale generation fails closed" {
    const ctx: abi.Context = @ptrFromInt(0xb201);
    try retain(0xb202, ctx);
    setCurrent(0xb202, ctx);
    defer clearCurrent(0xb202, ctx);
    var owner = Owner.capture();
    const stale = owner;
    release(0xb202, ctx);
    try owner.ensure();
    try std.testing.expect(!driverHealthy(0xb202));
    owner.release();
    try std.testing.expectError(error.CudaContextLost, stale.ensure());
    try retain(0xb202, ctx);
    defer release(0xb202, ctx);
    try std.testing.expectError(error.CudaContextLost, stale.ensure());
    try std.testing.expectError(error.CudaContextLost, (Owner{}).ensure());
    const free = Owner.contextFree();
    try free.ensure();
}

/// A primary reference cannot be released while dependent resources still use it.
pub fn soleReference(driver: usize, context: abi.Context) bool {
    lock();
    defer mutex.unlock();
    const e = find(driver, context) orelse return false;
    return e.refs == 1;
}

/// Cleanup disposition propagates through legacy void destructors. A parent
/// samples this monotonic thread-local epoch and must retain its allocation if
/// any descendant reports uncertain release. This is not context health.
threadlocal var cleanup_epoch: usize = 0;
pub fn cleanupMark() usize { return cleanup_epoch; }
pub fn cleanupRetained() void { cleanup_epoch +%= 1; }
pub fn cleanupReleased(mark: usize) bool { return cleanup_epoch == mark; }

const RetainedAllocation = struct { next: ?*RetainedAllocation, ptr: *anyopaque };
var retained_mutex: std.atomic.Mutex = .unlocked;
var retained_allocations: ?*RetainedAllocation = null;
/// Reserve custody before acquiring children so failure unwind never allocates.
pub const Custody = struct {
    record: *RetainedAllocation,
    pub fn init(ptr: *anyopaque) error{OutOfMemory}!Custody {
        const record = std.heap.page_allocator.create(RetainedAllocation) catch return error.OutOfMemory;
        record.* = .{ .next = null, .ptr = ptr };
        return .{ .record = record };
    }
    pub fn release(self: Custody) void { std.heap.page_allocator.destroy(self.record); }
    pub fn retain(self: Custody) void {
        while (!retained_mutex.tryLock()) std.Thread.yield() catch {}; defer retained_mutex.unlock();
        self.record.next = retained_allocations;
        retained_allocations = self.record;
    }
};

/// Host qualification drains only the exact record it observed, never production custody.
pub fn releaseTestCustody(ptr: *anyopaque) void {
    if (!@import("builtin").is_test) @compileError("test custody only");
    while (!retained_mutex.tryLock()) std.Thread.yield() catch {};
    defer retained_mutex.unlock();
    var at = &retained_allocations;
    while (at.*) |record| {
        if (record.ptr == ptr) {
            at.* = record.next;
            std.heap.page_allocator.destroy(record);
            return;
        }
        at = &record.next;
    }
    unreachable;
}

/// Observe only for host qualification; production cannot drain retained ledgers.
pub fn latestTestCustody() ?*anyopaque {
    if (!@import("builtin").is_test) @compileError("test custody only");
    while (!retained_mutex.tryLock()) std.Thread.yield() catch {};
    defer retained_mutex.unlock();
    return if (retained_allocations) |record| record.ptr else null;
}
