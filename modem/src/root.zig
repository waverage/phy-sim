//! Nestua PHY modem library, C ABI for the Python harness (ctypes).
//! M3: the whole TX and RX chain in f32 (modem.zig), plus the single
//! blocks for bit-exact and statistical tests against Sionna. The
//! fixed-point conversion is M4. No allocation after create.

const std = @import("std");
const modem = @import("modem.zig");
const ldpc = @import("ldpc.zig");
const Qam = @import("qam.zig").Qam;
const C = @import("ofdm.zig").C;

pub const version_major: u32 = 0;
pub const version_minor: u32 = 3;

const alloc = std.heap.page_allocator;

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

// ---- frame modem ---------------------------------------------------------

export fn nphy_create(cfg: *const modem.Config) ?*modem.Modem {
    return modem.Modem.create(alloc, cfg) catch null;
}

export fn nphy_destroy(m: *modem.Modem) void {
    m.destroy();
}

/// Complex samples per frame at the transmitter output.
export fn nphy_num_samples(m: *const modem.Modem) u32 {
    return m.numSamples();
}

/// wb may be null when phase tracking is off. Returns 0 on success.
export fn nphy_set_filter(m: *modem.Modem, w: [*]const C, err_var: [*]const f32, wb: ?[*]const C) i32 {
    m.setFilter(w, err_var, wb) catch return -1;
    return 0;
}

/// info: [batch][num_codewords * k] bits (0/1) -> out: [batch][num_samples].
export fn nphy_tx(m: *modem.Modem, info: [*]const u8, batch: u32, out: [*]C) void {
    m.tx(info, batch, out);
}

/// y: [batch][len_in] received samples, the FFT window of symbol 0 starting
/// after the CP at sample 0, l_min as Sionna's OFDMDemodulator. no: noise
/// variance per sample. out: [batch][num_codewords * k] decoded bits;
/// llr_out (optional): [batch][num_codewords * n] deinterleaved LLRs.
export fn nphy_rx(m: *modem.Modem, y: [*]const C, len_in: u32, l_min: i32, no: f32, batch: u32, out: [*]u8, llr_out: ?[*]f32) i32 {
    m.rx(y, len_in, l_min, no, batch, out, llr_out) catch return -1;
    return 0;
}

/// out[0] = decoder iterations, out[1] = codewords since the last reset.
export fn nphy_decoder_stats(m: *modem.Modem, out: *[2]u64, reset: u32) void {
    m.stats(out, reset != 0);
}

// ---- single blocks, for tests ----------------------------------------------

/// out: bg, z, k_filler, n_ldpc, dec_rows, dec_cols. Returns 0 on success.
export fn nphy_ldpc_info(k: u32, n: u32, out: *[6]u32) i32 {
    const c = ldpc.Code.init(k, n) catch return -1;
    out.* = .{ c.bg, c.z, c.k_filler, c.n_ldpc, c.dec_rows, c.dec_cols };
    return 0;
}

export fn nphy_ldpc_encode(k: u32, n: u32, bits: [*]const u8, count: u32, out: [*]u8) i32 {
    const c = ldpc.Code.init(k, n) catch return -1;
    const scratch = alloc.alloc(u8, c.n_ldpc) catch return -1;
    defer alloc.free(scratch);
    for (0..count) |i| c.encode(bits[i * k ..][0..k], out[i * n ..][0..n], scratch);
    return 0;
}

/// Returns the total number of iterations run, or -1.
export fn nphy_ldpc_decode(k: u32, n: u32, llr: [*]const f32, count: u32, num_iter: u32, early_stop: u32, out: [*]u8) i64 {
    const c = ldpc.Code.init(k, n) catch return -1;
    var d = ldpc.Decoder.init(alloc) catch return -1;
    defer d.deinit(alloc);
    var total: i64 = 0;
    for (0..count) |i| total += d.decode(&c, llr[i * n ..][0..n], num_iter, early_stop != 0, out[i * k ..][0..k]);
    return total;
}

/// bits: [count * bits_per_symbol] -> out: [count] symbols.
export fn nphy_qam_map(bits_per_symbol: u32, bits: [*]const u8, count: u32, out: [*]C) i32 {
    const q = Qam.init(bits_per_symbol) catch return -1;
    for (0..count) |i| out[i] = q.map(bits[i * bits_per_symbol ..][0..bits_per_symbol]);
    return 0;
}

/// y, no: [count] -> out: [count * bits_per_symbol] LLRs log p(1) / p(0).
export fn nphy_qam_demap(bits_per_symbol: u32, y: [*]const C, no: [*]const f32, count: u32, out: [*]f32) i32 {
    const q = Qam.init(bits_per_symbol) catch return -1;
    for (0..count) |i| q.demap(y[i], no[i], out[i * bits_per_symbol ..][0..bits_per_symbol]);
    return 0;
}

test {
    _ = @import("fmath.zig");
    _ = @import("ldpc.zig");
    _ = @import("qam.zig");
    _ = @import("ofdm.zig");
    _ = @import("modem.zig");
}

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
