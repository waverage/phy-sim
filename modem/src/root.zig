//! Nestua PHY modem library. Fixed-point signal path, no allocation after
//! init, C ABI for the Python harness. M0 placeholder: version + a checked
//! saturating add used by the ctypes smoke test.

pub const version_major: u32 = 0;
pub const version_minor: u32 = 1;

export fn nestuaphy_version() u32 {
    return (version_major << 16) | version_minor;
}

/// Saturating add of two Q15 values. The first fixed-point primitive; every
/// later block uses helpers of this shape so saturation is explicit.
export fn nestuaphy_sat_add_q15(a: i16, b: i16) i16 {
    return a +| b;
}

/// Sum of a Q15 vector into i32, exercising SIMD vectors and the array ABI.
export fn nestuaphy_sum_q15(ptr: [*]const i16, len: usize) i32 {
    const V = @Vector(8, i16);
    var acc: @Vector(8, i32) = @splat(0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const v: V = ptr[i..][0..8].*;
        acc += @as(@Vector(8, i32), v);
    }
    var s: i32 = @reduce(.Add, acc);
    while (i < len) : (i += 1) s += ptr[i];
    return s;
}

const std = @import("std");
test "saturating add clamps" {
    try std.testing.expectEqual(@as(i16, 32767), nestuaphy_sat_add_q15(30000, 10000));
    try std.testing.expectEqual(@as(i16, -32768), nestuaphy_sat_add_q15(-30000, -10000));
    try std.testing.expectEqual(@as(i16, 5), nestuaphy_sat_add_q15(2, 3));
}
test "vector sum matches scalar" {
    var buf: [37]i16 = undefined;
    var expect: i32 = 0;
    for (&buf, 0..) |*x, k| {
        x.* = @intCast(@as(i32, @intCast(k)) * 100 - 1800);
        expect += x.*;
    }
    try std.testing.expectEqual(expect, nestuaphy_sum_q15(&buf, buf.len));
}
