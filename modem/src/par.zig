//! Batch parallelism: items 0..batch-1 handed out to worker threads through
//! an atomic counter, one workspace per thread.

const std = @import("std");

pub const max_threads = 64;

pub fn threadCount(requested: u32) u32 {
    const cpus: u32 = @intCast(std.Thread.getCpuCount() catch 1);
    return @max(1, @min(max_threads, if (requested == 0) cpus else requested));
}

/// func(ctx, &ws[t], i) for every i < batch, on min(ws.len, batch) threads.
pub fn run(comptime W: type, ws: []W, batch: u32, ctx: anytype, comptime func: fn (@TypeOf(ctx), *W, usize) void) void {
    const T = @min(@as(u32, @intCast(ws.len)), batch);
    if (T == 0) return;
    var next = std.atomic.Value(u32).init(0);
    const Worker = struct {
        fn work(c: @TypeOf(ctx), w: *W, counter: *std.atomic.Value(u32), total: u32) void {
            while (true) {
                const i = counter.fetchAdd(1, .monotonic);
                if (i >= total) return;
                func(c, w, i);
            }
        }
    };
    var handles: [max_threads]?std.Thread = @splat(null);
    for (1..T) |t| {
        // a thread that fails to start leaves its items to the others
        handles[t] = std.Thread.spawn(.{}, Worker.work, .{ ctx, &ws[t], &next, batch }) catch null;
    }
    Worker.work(ctx, &ws[0], &next, batch);
    for (handles[1..T]) |h| if (h) |th| th.join();
}
