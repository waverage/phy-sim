//! Vectorised f32 math for the LDPC check node: exp and log as Cephes
//! polynomials (about 1 ulp), evaluated 8 lanes at a time. LLVM lowers
//! @exp / @log on vectors to one libm call per lane, 10x slower.

const std = @import("std");

pub const lanes = 8;
pub const V = @Vector(lanes, f32);
const I = @Vector(lanes, i32);
const U = @Vector(lanes, u32);

pub inline fn splat(x: f32) V {
    return @splat(x);
}

pub fn exp(x_in: V) V {
    const x = @min(@max(x_in, splat(-87.0)), splat(88.0));
    const n = @round(x * splat(1.44269504088896341));
    var r = x - n * splat(0.693359375);
    r = r - n * splat(-2.12194440e-4);
    const z = r * r;
    var p = splat(1.9875691500e-4);
    p = p * r + splat(1.3981999507e-3);
    p = p * r + splat(8.3334519073e-3);
    p = p * r + splat(4.1665795894e-2);
    p = p * r + splat(1.6666665459e-1);
    p = p * r + splat(5.0000001201e-1);
    p = p * z + r + splat(1.0);
    const ni: I = @intFromFloat(n);
    const sh: @Vector(lanes, u5) = @splat(23);
    const scale: V = @bitCast((ni + @as(I, @splat(127))) << sh);
    return p * scale;
}

/// e^x - 1 without the cancellation of exp(x) - 1 near 0 (x >= 0 here).
pub fn expm1(x: V) V {
    var t = splat(1.0 / 5040.0);
    t = t * x + splat(1.0 / 720.0);
    t = t * x + splat(1.0 / 120.0);
    t = t * x + splat(1.0 / 24.0);
    t = t * x + splat(1.0 / 6.0);
    t = t * x + splat(0.5);
    t = t * x + splat(1.0);
    t = t * x;
    return @select(f32, x < splat(0.5), t, exp(x) - splat(1.0));
}

/// Natural log of positive normal numbers.
pub fn log(x: V) V {
    const bits: U = @bitCast(x);
    const sh: @Vector(lanes, u5) = @splat(23);
    var e: V = @floatFromInt(@as(I, @bitCast(bits >> sh)) - @as(I, @splat(126)));
    var m: V = @bitCast((bits & @as(U, @splat(0x807fffff))) | @as(U, @splat(0x3f000000)));
    const small = m < splat(0.707106781186547524);
    e = @select(f32, small, e - splat(1.0), e);
    m = @select(f32, small, m + m - splat(1.0), m - splat(1.0));
    const z = m * m;
    var y = splat(7.0376836292e-2);
    y = y * m - splat(1.1514610310e-1);
    y = y * m + splat(1.1676998740e-1);
    y = y * m - splat(1.2420140846e-1);
    y = y * m + splat(1.4249322787e-1);
    y = y * m - splat(1.6668057665e-1);
    y = y * m + splat(2.0000714765e-1);
    y = y * m - splat(2.4999993993e-1);
    y = y * m + splat(3.3333331174e-1);
    y = y * m * z;
    y += e * splat(-2.12194440e-4);
    y -= splat(0.5) * z;
    return m + y + e * splat(0.693359375);
}

/// log(1 + u), u >= 0, with a series below 0.05 where 1 + u rounds.
pub fn log1p(u: V) V {
    var s = splat(-1.0 / 6.0);
    s = s * u + splat(1.0 / 5.0);
    s = s * u - splat(1.0 / 4.0);
    s = s * u + splat(1.0 / 3.0);
    s = s * u - splat(1.0 / 2.0);
    s = s * u + splat(1.0);
    s = s * u;
    return @select(f32, u < splat(0.05), s, log(splat(1.0) + u));
}

/// phi(x) = -log(tanh(x / 2)) = log(1 + 2 / (e^x - 1)), on Sionna's f32
/// clamp range. phi is its own inverse; the check node applies it twice.
pub fn phi(x_in: V) V {
    const x = @min(@max(x_in, splat(8.5e-8)), splat(16.635532));
    return log1p(splat(2.0) / expm1(x));
}

test "vector exp, log and phi against std.math" {
    var xs = [_]f32{ 1e-7, 1e-4, 0.01, 0.3, 0.49, 0.51, 1.0, 2.5, 7.0, 16.0, 30.0, 80.0, 1e-30, 1e20, 0.7071, 3.3 };
    var i: usize = 0;
    while (i < xs.len) : (i += lanes) {
        const v: V = xs[i..][0..lanes].*;
        const lg: [lanes]f32 = log(v);
        const ex: [lanes]f32 = exp(@min(v, splat(80.0)));
        const ph: [lanes]f32 = phi(v);
        for (0..lanes) |j| {
            const x: f64 = xs[i + j];
            try std.testing.expectApproxEqRel(@log(x), lg[j], 2e-6);
            try std.testing.expectApproxEqRel(@exp(@min(x, 80.0)), ex[j], 2e-6);
            const xc = @min(@max(x, 8.5e-8), 16.635532);
            const want = -@log(std.math.tanh(xc / 2));
            try std.testing.expectApproxEqRel(want, ph[j], 1e-4);
        }
    }
}
