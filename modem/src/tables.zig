//! Generated waveform tables, as harness/tables.py defines them: QPSK
//! pilots from the TS 38.211 Gold sequence and the frame bit interleaver
//! from a xorshift32 Fisher-Yates shuffle.

const std = @import("std");

pub const pilot_seed: u32 = 0x4E5354;
pub const perm_seed: u32 = 0x4E535450;

/// c(0..out.len-1) of TS 38.211 5.2.1, one bit per byte.
pub fn gold(c_init: u32, out: []u8) void {
    var x1: u32 = 1; // bit i = x1(n + i), 31 bits
    var x2: u32 = c_init & 0x7FFF_FFFF;
    var n: usize = 0;
    while (n < 1600 + out.len) : (n += 1) {
        if (n >= 1600) out[n - 1600] = @intCast((x1 ^ x2) & 1);
        const b1 = ((x1 >> 3) ^ x1) & 1;
        const b2 = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
        x1 = (x1 >> 1) | (b1 << 30);
        x2 = (x2 >> 1) | (b2 << 30);
    }
}

/// signs[2p], signs[2p + 1] = (a, b) of pilot p = (a + j b) / sqrt(2).
pub fn pilotSigns(seed: u32, signs: []i8, scratch: []u8) void {
    gold(seed, scratch[0..signs.len]);
    for (signs, scratch[0..signs.len]) |*s, c| s.* = 1 - 2 * @as(i8, @intCast(c));
}

/// perm[i]: the coded bit sent as transmitted bit i.
pub fn interleaver(seed: u32, perm: []u32) void {
    for (perm, 0..) |*p, i| p.* = @intCast(i);
    var s: u32 = seed;
    var i: usize = perm.len;
    while (i > 1) {
        i -= 1;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        const j: usize = @intCast((@as(u64, s) * (i + 1)) >> 32);
        std.mem.swap(u32, &perm[i], &perm[j]);
    }
}

test "gold sequence starts as the reference implementation" {
    // c_init = 0 reduces x2 to zero: c is x1 alone after 1600 steps
    var out: [8]u8 = undefined;
    gold(0, &out);
    var x1: [1640]u8 = @splat(0);
    x1[0] = 1;
    for (0..1609) |i| x1[i + 31] = x1[i + 3] ^ x1[i];
    try std.testing.expectEqualSlices(u8, x1[1600..1608], &out);
}

test "interleaver is a permutation" {
    var perm: [1000]u32 = undefined;
    interleaver(perm_seed, &perm);
    var seen: [1000]bool = @splat(false);
    for (perm) |p| seen[p] = true;
    for (seen) |s| try std.testing.expect(s);
}
