//! Fixed-point primitives of the M4 modem. Every rounding and saturation
//! in the fixed-point chain goes through these, and harness/twin.py
//! implements the same functions on numpy int64, so the two are bit exact.
//!
//! Conventions: rshr rounds half up (add half an LSB, arithmetic shift);
//! sat clamps to a two's complement width, satSym to the symmetric range
//! (LLRs and decoder messages, so negation never overflows). Angles are
//! binary: 65536 units per turn.

const std = @import("std");
const C = @import("ofdm.zig").C;

pub inline fn rshr(x: i64, n: u6) i64 {
    if (n == 0) return x;
    return (x + (@as(i64, 1) << (n - 1))) >> n;
}

/// Round half to even. The FFT needs it: its trivial twiddles make exact
/// halves frequent, and half-up's bias then adds up coherently into a few
/// bins (14 LSB at DC, 8 at bin 1 in a 256-point transform).
pub inline fn rshrEven(x: i64, n: u6) i64 {
    const q = x >> n;
    const r = x - (q << n);
    const half = @as(i64, 1) << (n - 1);
    return if (r > half or (r == half and (q & 1) == 1)) q + 1 else q;
}

pub inline fn sat(x: i64, bits: u6) i64 {
    const hi = (@as(i64, 1) << (bits - 1)) - 1;
    return std.math.clamp(x, -hi - 1, hi);
}

pub inline fn satSym(x: i64, bits: u6) i64 {
    const hi = (@as(i64, 1) << (bits - 1)) - 1;
    return std.math.clamp(x, -hi, hi);
}

/// a / b rounded half up, b > 0.
pub inline fn divRound(a: i64, b: i64) i64 {
    return @divFloor(2 * a + b, 2 * b);
}

/// Complex integer sample. Its width is a convention of the block that
/// writes it (16 bits for samples and channel estimates).
pub const Ci = extern struct {
    re: i32,
    im: i32,
    pub const zero: Ci = .{ .re = 0, .im = 0 };
};

// ---- CORDIC ------------------------------------------------------------------

pub const turn: i64 = 65536;
pub const cordic_iters = 14;
/// round(atan(2^-i) / 2 pi * 65536)
pub const cordic_atan = [cordic_iters]i64{ 8192, 4836, 2555, 1297, 651, 326, 163, 81, 41, 20, 10, 5, 3, 1 };
/// round(2^15 / K), K = prod sqrt(1 + 2^-2i), the start of a rotation
pub const cordic_x0: i64 = 19898;

/// Vectoring: angle of (x, y) in (-32768, 32768] and magnitude times the
/// CORDIC gain K = 1.6468. Inputs below about 2^60.
pub fn vector(x_in: i64, y_in: i64) struct { mag: i64, ang: i64 } {
    var x = x_in;
    var y = y_in;
    var ang: i64 = 0;
    if (x < 0) {
        x = -x;
        y = -y;
        ang = turn / 2;
    }
    for (0..cordic_iters) |i| {
        const sh: u6 = @intCast(i);
        const dx = y >> sh;
        const dy = x >> sh;
        if (y > 0) {
            x += dx;
            y -= dy;
            ang += cordic_atan[i];
        } else {
            x -= dx;
            y += dy;
            ang -= cordic_atan[i];
        }
    }
    if (ang > turn / 2) ang -= turn;
    return .{ .mag = x, .ang = ang };
}

/// Rotation: 2^15 (cos, sin) of an angle in binary units (any integer).
pub fn cossin(ang_in: i64) Ci {
    var a = @mod(ang_in, turn);
    if (a > turn / 2) a -= turn;
    var neg = false;
    if (a > turn / 4) {
        a -= turn / 2;
        neg = true;
    } else if (a < -turn / 4) {
        a += turn / 2;
        neg = true;
    }
    var x: i64 = cordic_x0;
    var y: i64 = 0;
    for (0..cordic_iters) |i| {
        const sh: u6 = @intCast(i);
        const dx = y >> sh;
        const dy = x >> sh;
        if (a >= 0) {
            x -= dx;
            y += dy;
            a -= cordic_atan[i];
        } else {
            x += dx;
            y -= dy;
            a += cordic_atan[i];
        }
    }
    if (neg) {
        x = -x;
        y = -y;
    }
    return .{ .re = @intCast(x), .im = @intCast(y) };
}

/// v * (c + j s) / 2^15, rounded, 16-bit.
pub inline fn rot16(v: Ci, cs: Ci) Ci {
    const re = @as(i64, v.re) * cs.re - @as(i64, v.im) * cs.im;
    const im = @as(i64, v.re) * cs.im + @as(i64, v.im) * cs.re;
    return .{ .re = @intCast(sat(rshr(re, 15), 16)), .im = @intCast(sat(rshr(im, 15), 16)) };
}

// ---- FFT -----------------------------------------------------------------------

pub const data_bits = 16;
pub const tw_frac = 16; // twiddles Q16 in 18 bits: 1.0 is exact

/// Radix-2 decimation in time, unitary like ofdm.Fft: the even stages
/// halve (with rounding), so N = 2^m with m even. Fixed: 16-bit data,
/// Q16 twiddles, each butterfly output rounded once (half to even) and
/// saturated.
/// Float: the same stages with exact twiddles and no rounding.
pub const Fft = struct {
    n: u32,
    stages: u32,
    rev: []u32,
    twq: []Ci, // round(65536 e^{-2 pi i j / n}), j < n / 2
    twf: []C,

    pub fn init(alloc: std.mem.Allocator, n: u32) !Fft {
        if (n < 4 or !std.math.isPowerOfTwo(n)) return error.UnsupportedFftSize;
        const m = std.math.log2_int(u32, n);
        if (m % 2 != 0) return error.UnsupportedFftSize;
        const rev = try alloc.alloc(u32, n);
        errdefer alloc.free(rev);
        const twq = try alloc.alloc(Ci, n / 2);
        errdefer alloc.free(twq);
        const twf = try alloc.alloc(C, n / 2);
        for (0..n / 2) |j| {
            const a = -2.0 * std.math.pi * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(n));
            twq[j] = .{ .re = @intFromFloat(@floor(@cos(a) * 65536.0 + 0.5)), .im = @intFromFloat(@floor(@sin(a) * 65536.0 + 0.5)) };
            twf[j] = .{ .re = @floatCast(@cos(a)), .im = @floatCast(@sin(a)) };
        }
        for (rev, 0..) |*r, i| r.* = @bitReverse(@as(u32, @intCast(i))) >> @intCast(@as(u6, 32) - m);
        return .{ .n = n, .stages = m, .rev = rev, .twq = twq, .twf = twf };
    }

    pub fn deinit(f: *Fft, alloc: std.mem.Allocator) void {
        alloc.free(f.rev);
        alloc.free(f.twq);
        alloc.free(f.twf);
    }

    pub fn runFixed(f: *const Fft, x: []Ci, inverse: bool) void {
        const n = f.n;
        for (0..n) |i| {
            const r = f.rev[i];
            if (r > i) std.mem.swap(Ci, &x[i], &x[r]);
        }
        var size: u32 = 2;
        var stage: u32 = 0;
        while (size <= n) : ({
            size *= 2;
            stage += 1;
        }) {
            const half = size / 2;
            const step = n / size;
            const sh: u6 = tw_frac + @as(u6, @intFromBool(stage % 2 == 0));
            var start: u32 = 0;
            while (start < n) : (start += size) {
                for (0..half) |j| {
                    const w = f.twq[j * step];
                    const wi: i64 = if (inverse) -w.im else w.im;
                    const a = x[start + j];
                    const b = x[start + j + half];
                    const tr = @as(i64, w.re) * b.re - wi * b.im;
                    const ti = @as(i64, w.re) * b.im + wi * b.re;
                    const ar = @as(i64, a.re) << tw_frac;
                    const ai = @as(i64, a.im) << tw_frac;
                    x[start + j] = .{ .re = @intCast(sat(rshrEven(ar + tr, sh), data_bits)), .im = @intCast(sat(rshrEven(ai + ti, sh), data_bits)) };
                    x[start + j + half] = .{ .re = @intCast(sat(rshrEven(ar - tr, sh), data_bits)), .im = @intCast(sat(rshrEven(ai - ti, sh), data_bits)) };
                }
            }
        }
    }

    pub fn runFloat(f: *const Fft, x: []C, inverse: bool) void {
        const n = f.n;
        for (0..n) |i| {
            const r = f.rev[i];
            if (r > i) std.mem.swap(C, &x[i], &x[r]);
        }
        var size: u32 = 2;
        var stage: u32 = 0;
        while (size <= n) : ({
            size *= 2;
            stage += 1;
        }) {
            const half = size / 2;
            const step = n / size;
            const g: f32 = if (stage % 2 == 0) 0.5 else 1.0;
            var start: u32 = 0;
            while (start < n) : (start += size) {
                for (0..half) |j| {
                    var w = f.twf[j * step];
                    if (inverse) w.im = -w.im;
                    const t = w.mul(x[start + j + half]);
                    const a = x[start + j];
                    x[start + j] = a.add(t).scale(g);
                    x[start + j + half] = a.sub(t).scale(g);
                }
            }
        }
    }
};

test "rounding, saturation and division" {
    try std.testing.expectEqual(@as(i64, 2), rshr(3, 1));
    try std.testing.expectEqual(@as(i64, -1), rshr(-3, 1));
    try std.testing.expectEqual(@as(i64, 127), sat(1000, 8));
    try std.testing.expectEqual(@as(i64, -128), sat(-1000, 8));
    try std.testing.expectEqual(@as(i64, -127), satSym(-1000, 8));
    try std.testing.expectEqual(@as(i64, 3), divRound(5, 2));
    try std.testing.expectEqual(@as(i64, -2), divRound(-5, 2));
}

test "cordic angle and rotation" {
    const v = vector(-1000000, 1000000); // 135 deg
    try std.testing.expect(@abs(v.ang - 24576) <= 2);
    try std.testing.expect(@abs(@as(f64, @floatFromInt(v.mag)) / 1.6467602 - 1414213.6) < 50);
    const cs = cossin(-24576);
    try std.testing.expect(@abs(cs.re + 23170) <= 3 and @abs(cs.im + 23170) <= 3);
}

test "fixed fft matches float fft within rounding" {
    const alloc = std.testing.allocator;
    var f = try Fft.init(alloc, 256);
    defer f.deinit(alloc);
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    var xi: [256]Ci = undefined;
    var xf: [256]C = undefined;
    for (&xi, &xf) |*a, *b| {
        a.* = .{ .re = rnd.intRangeAtMost(i32, -2000, 2000), .im = rnd.intRangeAtMost(i32, -2000, 2000) };
        b.* = .{ .re = @floatFromInt(a.re), .im = @floatFromInt(a.im) };
    }
    f.runFixed(&xi, false);
    f.runFloat(&xf, false);
    // about 1 LSB rms of rounding noise, no bias
    for (xi, xf) |a, b| {
        try std.testing.expect(@abs(@as(f32, @floatFromInt(a.re)) - b.re) < 6);
        try std.testing.expect(@abs(@as(f32, @floatFromInt(a.im)) - b.im) < 6);
    }
}
