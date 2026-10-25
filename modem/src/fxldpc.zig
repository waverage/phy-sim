//! Layered offset min-sum decoder for the 5G LDPC codes of ldpc.zig (M4),
//! the hardware decoder's algorithm. Row-layered schedule: each base row
//! reads the a-posteriori values (app) of its columns through the cyclic
//! shifts, subtracts its old check-to-variable messages, computes new ones
//! with min-sum minus an offset, and writes app back, so later rows in the
//! same iteration already see the update (about twice the convergence speed
//! of flooding).
//!
//! Generic over the number type. i32 is the fixed-point decoder: channel
//! LLRs in llr_bits, app in app_bits and messages in msg_bits, all
//! symmetric-saturated, the offset an integer number of LSBs. f32 runs the
//! same schedule without saturation or rounding, the floating-point
//! reference of the same algorithm. Bit exact against harness/twin.py
//! (ldpc_decode) in fixed point.

const std = @import("std");
const ldpc = @import("ldpc.zig");
const fx = @import("fx.zig");
const Code = ldpc.Code;

pub const Params = extern struct {
    llr_bits: u32,
    app_bits: u32,
    msg_bits: u32,
    offset: f32, // in LLR LSBs (integer in fixed point)
    num_iter: u32,
    early_stop: u32,
};

const L = 8;

pub fn Decoder(comptime T: type) type {
    const fixed = @typeInfo(T) == .int;
    const V = @Vector(L, T);
    return struct {
        const Self = @This();
        app: []T,
        c2v: []T,
        t: []T, // one row's app - c2v, [max_row_degree][zp]
        hard: []u8,

        pub fn init(alloc: std.mem.Allocator) !Self {
            return .{
                .app = try alloc.alloc(T, ldpc.max_cols * ldpc.max_z),
                .c2v = try alloc.alloc(T, ldpc.max_entries * ldpc.max_z),
                .t = try alloc.alloc(T, ldpc.max_row_degree * ldpc.max_z),
                .hard = try alloc.alloc(u8, ldpc.max_cols * ldpc.max_z),
            };
        }

        pub fn deinit(d: *Self, alloc: std.mem.Allocator) void {
            alloc.free(d.app);
            alloc.free(d.c2v);
            alloc.free(d.t);
            alloc.free(d.hard);
        }

        inline fn lim(bits: u32) T {
            return if (fixed) @intCast((@as(i64, 1) << @intCast(bits - 1)) - 1) else std.math.floatMax(T);
        }

        inline fn clampS(x: T, hi: T) T {
            return if (fixed) @max(-hi, @min(hi, x)) else x;
        }

        inline fn clampV(x: V, hi: T) V {
            return if (fixed) @max(@as(V, @splat(-hi)), @min(@as(V, @splat(hi)), x)) else x;
        }

        inline fn absV(x: V) V {
            return @select(T, x < @as(V, @splat(0)), -x, x);
        }

        /// logits: n channel LLRs, log p(1) / p(0) in LSB units (fixed:
        /// integers; saturated to llr_bits here). out: k bits. Returns the
        /// iterations run.
        pub fn decode(d: *Self, c: *const Code, logits: []const T, p: Params, out: []u8) u32 {
            const z = c.z;
            const zp = c.zp;
            const app_max = lim(p.app_bits);
            const msg_max = lim(p.msg_bits);
            const llr_max = lim(p.llr_bits);
            const off: T = if (fixed) @intFromFloat(p.offset) else p.offset;

            const app = d.app[0 .. c.dec_cols * zp];
            @memset(app, 0);
            const n_sys = c.k - 2 * z;
            for (0..c.n) |t| {
                const j = if (t < n_sys) 2 * z + t else c.k_ldpc + (t - n_sys);
                app[j / z * zp + j % z] = -clampS(logits[t], llr_max);
            }
            // filler bits are known zeros
            const filler: T = if (fixed) app_max else 1e6;
            for (c.k..c.k_ldpc) |j| app[j / z * zp + j % z] = filler;
            @memset(d.c2v[0 .. c.dec_entries * zp], 0);

            var it: u32 = 0;
            while (it < p.num_iter) {
                for (0..c.dec_rows) |row| d.layer(c, row, app_max, msg_max, off);
                it += 1;
                if (p.early_stop != 0 and it < p.num_iter and d.syndromeZero(c)) break;
            }
            for (0..c.k) |j| out[j] = @intFromBool(app[j / z * zp + j % z] <= 0);
            return it;
        }

        fn layer(d: *Self, c: *const Code, row: usize, app_max: T, msg_max: T, off: T) void {
            const z = c.z;
            const zp = c.zp;
            const ents = c.entries[c.row_start[row]..c.row_start[row + 1]];
            const e0: usize = c.row_start[row];
            // gather: t_j[i] = app[col][(i + s) mod z] - c2v_j[i]
            for (ents, 0..) |e, j| {
                const tb = d.t[j * zp ..][0..zp];
                const src = d.app[@as(usize, e.col) * zp ..][0..z];
                const old = d.c2v[(e0 + j) * zp ..][0..zp];
                const s = e.shift;
                for (0..z - s) |i| tb[i] = clampS(src[i + s] - old[i], app_max);
                for (z - s..z) |i| tb[i] = clampS(src[i + s - z] - old[i], app_max);
                @memset(tb[z..zp], 0);
            }
            const zero: V = @splat(0);
            const big: V = @splat(if (fixed) std.math.maxInt(T) else std.math.floatMax(T));
            var lo: usize = 0;
            while (lo < zp) : (lo += L) {
                var min1 = big;
                var min2 = big;
                var idx: @Vector(L, u32) = @splat(0);
                var par: @Vector(L, bool) = @splat(false);
                for (0..ents.len) |j| {
                    const v: V = d.t[j * zp + lo ..][0..L].*;
                    const m = @min(absV(v), @as(V, @splat(msg_max)));
                    const lt1 = m < min1;
                    min2 = @select(T, lt1, min1, @min(min2, m));
                    min1 = @select(T, lt1, m, min1);
                    idx = @select(u32, lt1, @as(@Vector(L, u32), @splat(@intCast(j))), idx);
                    par = par != (v < zero);
                }
                const m1 = @max(min1 - @as(V, @splat(off)), zero);
                const m2 = @max(min2 - @as(V, @splat(off)), zero);
                for (0..ents.len) |j| {
                    const tv = d.t[j * zp + lo ..][0..L];
                    const v: V = tv.*;
                    const mag = @select(T, idx == @as(@Vector(L, u32), @splat(@intCast(j))), m2, m1);
                    const neg = par != (v < zero);
                    const msg = @select(T, neg, -mag, mag);
                    d.c2v[(e0 + j) * zp + lo ..][0..L].* = msg;
                    tv.* = clampV(v + msg, app_max);
                }
            }
            // scatter: app[col][(i + s) mod z] = t_j[i]
            for (ents, 0..) |e, j| {
                const tb = d.t[j * zp ..][0..zp];
                const dst = d.app[@as(usize, e.col) * zp ..][0..z];
                const s = e.shift;
                for (0..z - s) |i| dst[i + s] = tb[i];
                for (z - s..z) |i| dst[i + s - z] = tb[i];
            }
        }

        fn syndromeZero(d: *Self, c: *const Code) bool {
            const z = c.z;
            const zp = c.zp;
            for (0..c.dec_cols * zp) |i| d.hard[i] = @intFromBool(d.app[i] <= 0);
            var acc: [ldpc.max_z]u8 = undefined;
            for (0..c.dec_rows) |row| {
                @memset(acc[0..z], 0);
                for (c.rowEntries(row)) |e| ldpc.xorShifted(acc[0..z], d.hard[@as(usize, e.col) * zp ..][0..z], e.shift);
                for (acc[0..z]) |a| if (a != 0) return false;
            }
            return true;
        }
    };
}

test "noiseless codeword decodes, fixed and float" {
    const alloc = std.testing.allocator;
    const c = try Code.init(4776, 9552);
    var prng = std.Random.DefaultPrng.init(1);
    const rnd = prng.random();
    const u = try alloc.alloc(u8, c.k);
    defer alloc.free(u);
    for (u) |*b| b.* = rnd.int(u1);
    const x = try alloc.alloc(u8, c.n);
    defer alloc.free(x);
    const scratch = try alloc.alloc(u8, c.n_ldpc);
    defer alloc.free(scratch);
    c.encode(u, x, scratch);
    const out = try alloc.alloc(u8, c.k);
    defer alloc.free(out);
    const p = Params{ .llr_bits = 6, .app_bits = 8, .msg_bits = 6, .offset = 1, .num_iter = 20, .early_stop = 1 };
    inline for (.{ i32, f32 }) |T| {
        const llr = try alloc.alloc(T, c.n);
        defer alloc.free(llr);
        for (x, llr) |b, *l| l.* = if (b == 1) 8 else -8;
        var d = try Decoder(T).init(alloc);
        defer d.deinit(alloc);
        const iters = d.decode(&c, llr, p, out);
        try std.testing.expectEqualSlices(u8, u, out);
        try std.testing.expect(iters <= 2);
    }
}
