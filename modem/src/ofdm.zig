//! Complex type and a radix-2 FFT, unitary (1 / sqrt(N)) in both
//! directions like Sionna's fft / ifft.

const std = @import("std");

pub const C = extern struct {
    re: f32,
    im: f32,

    pub const zero: C = .{ .re = 0, .im = 0 };

    pub inline fn add(a: C, b: C) C {
        return .{ .re = a.re + b.re, .im = a.im + b.im };
    }
    pub inline fn sub(a: C, b: C) C {
        return .{ .re = a.re - b.re, .im = a.im - b.im };
    }
    pub inline fn mul(a: C, b: C) C {
        return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
    }
    pub inline fn conj(a: C) C {
        return .{ .re = a.re, .im = -a.im };
    }
    pub inline fn scale(a: C, s: f32) C {
        return .{ .re = a.re * s, .im = a.im * s };
    }
    pub inline fn abs2(a: C) f32 {
        return a.re * a.re + a.im * a.im;
    }
    pub inline fn expj(theta: f32) C {
        return .{ .re = @cos(theta), .im = @sin(theta) };
    }
};

pub const Fft = struct {
    n: u32,
    tw: []C, // e^{-2 pi i j / n}, j < n / 2
    rev: []u32,
    norm: f32,

    pub fn init(alloc: std.mem.Allocator, n: u32) !Fft {
        if (n < 2 or !std.math.isPowerOfTwo(n)) return error.UnsupportedFftSize;
        const tw = try alloc.alloc(C, n / 2);
        errdefer alloc.free(tw);
        const rev = try alloc.alloc(u32, n);
        const bits: u5 = @intCast(std.math.log2_int(u32, n));
        for (tw, 0..) |*w, j| {
            const a = -2.0 * std.math.pi * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(n));
            w.* = .{ .re = @floatCast(@cos(a)), .im = @floatCast(@sin(a)) };
        }
        for (rev, 0..) |*r, i| r.* = @bitReverse(@as(u32, @intCast(i))) >> @intCast(@as(u6, 32) - bits);
        return .{ .n = n, .tw = tw, .rev = rev, .norm = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(n)))) };
    }

    pub fn deinit(f: *Fft, alloc: std.mem.Allocator) void {
        alloc.free(f.tw);
        alloc.free(f.rev);
    }

    /// In place; inverse uses e^{+2 pi i k n / N}.
    pub fn run(f: *const Fft, x: []C, inverse: bool) void {
        const n = f.n;
        for (0..n) |i| {
            const r = f.rev[i];
            if (r > i) std.mem.swap(C, &x[i], &x[r]);
        }
        var size: u32 = 2;
        while (size <= n) : (size *= 2) {
            const half = size / 2;
            const step = n / size;
            var start: u32 = 0;
            while (start < n) : (start += size) {
                for (0..half) |j| {
                    var w = f.tw[j * step];
                    if (inverse) w.im = -w.im;
                    const t = w.mul(x[start + j + half]);
                    x[start + j + half] = x[start + j].sub(t);
                    x[start + j] = x[start + j].add(t);
                }
            }
        }
        for (x[0..n]) |*v| v.* = v.scale(f.norm);
    }
};

test "fft is unitary and inverts" {
    const alloc = std.testing.allocator;
    var f = try Fft.init(alloc, 16);
    defer f.deinit(alloc);
    var x: [16]C = undefined;
    for (&x, 0..) |*v, i| v.* = .{ .re = @floatFromInt(i), .im = @as(f32, @floatFromInt(i % 3)) - 1 };
    const orig = x;
    f.run(&x, false);
    // bin 0 is sum / sqrt(N)
    var s = C.zero;
    for (orig) |v| s = s.add(v);
    try std.testing.expectApproxEqAbs(s.re / 4.0, x[0].re, 1e-5);
    f.run(&x, true);
    for (orig, x) |a, b| {
        try std.testing.expectApproxEqAbs(a.re, b.re, 1e-5);
        try std.testing.expectApproxEqAbs(a.im, b.im, 1e-5);
    }
}
