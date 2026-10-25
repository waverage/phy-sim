//! Nestua PHY modem library, C ABI for the Python harness (ctypes).
//! M3: the whole TX and RX chain in f32 (modem.zig), plus the single
//! blocks for bit-exact and statistical tests against Sionna.
//! M4: the product modem in fixed point and its float twin (fxmodem.zig),
//! plus its blocks for the bit-exact tests against harness/twin.py.
//! No allocation after create.

const std = @import("std");
const modem = @import("modem.zig");
const ldpc = @import("ldpc.zig");
const Qam = @import("qam.zig").Qam;
const C = @import("ofdm.zig").C;
const fxldpc = @import("fxldpc.zig");
const par = @import("par.zig");
const fx = @import("fx.zig");
const fxmodem = @import("fxmodem.zig");
const tables = @import("tables.zig");

pub const version_major: u32 = 0;
pub const version_minor: u32 = 4;

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

// ---- M4 fixed-point blocks ---------------------------------------------------

/// Batch LDPC decode on worker threads. algo 0: flooding boxplus-phi
/// (the M3 decoder; llr in nats, widths and offset ignored), 1: layered
/// offset min-sum in f32, 2: the same in fixed point (llr integers in LSB
/// units). llr: [count][n] logits log p(1) / p(0). iters (optional):
/// [count] iterations per codeword. Returns the total iterations, or -1.
export fn nphy_fx_ldpc_decode(k: u32, n: u32, llr: [*]const f32, count: u32, algo: u32, p: *const fxldpc.Params, num_threads: u32, out: [*]u8, iters: ?[*]u32) i64 {
    const c = ldpc.Code.init(k, n) catch return -1;
    return switch (algo) {
        0 => decodeBatch(ldpc.Decoder, &c, llr, count, p, num_threads, out, iters),
        1 => decodeBatch(fxldpc.Decoder(f32), &c, llr, count, p, num_threads, out, iters),
        2 => decodeBatch(fxldpc.Decoder(i32), &c, llr, count, p, num_threads, out, iters),
        else => -1,
    } catch -1;
}

fn decodeBatch(comptime D: type, c: *const ldpc.Code, llr: [*]const f32, count: u32, p: *const fxldpc.Params, num_threads: u32, out: [*]u8, iters: ?[*]u32) !i64 {
    const W = struct { dec: D, buf: []i32, total: i64 };
    const ws = try alloc.alloc(W, par.threadCount(num_threads));
    defer alloc.free(ws);
    var made: usize = 0;
    defer for (ws[0..made]) |*w| {
        w.dec.deinit(alloc);
        alloc.free(w.buf);
    };
    for (ws) |*w| {
        w.* = .{ .dec = try D.init(alloc), .buf = undefined, .total = 0 };
        w.buf = alloc.alloc(i32, c.n) catch |e| {
            w.dec.deinit(alloc);
            return e;
        };
        made += 1;
    }
    const Ctx = struct {
        c: *const ldpc.Code,
        llr: [*]const f32,
        p: *const fxldpc.Params,
        out: [*]u8,
        iters: ?[*]u32,
        fn run(x: @This(), w: *W, i: usize) void {
            const n = x.c.n;
            const k = x.c.k;
            const l = x.llr[i * n ..][0..n];
            const o = x.out[i * k ..][0..k];
            const it = if (D == ldpc.Decoder)
                w.dec.decode(x.c, l, x.p.num_iter, x.p.early_stop != 0, o)
            else if (D == fxldpc.Decoder(f32))
                w.dec.decode(x.c, l, x.p.*, o)
            else blk: {
                for (l, w.buf) |v, *b| b.* = @intFromFloat(std.math.clamp(v, -1e9, 1e9));
                break :blk w.dec.decode(x.c, w.buf, x.p.*, o);
            };
            w.total += it;
            if (x.iters) |ip| ip[i] = it;
        }
    };
    par.run(W, ws, count, Ctx{ .c = c, .llr = llr, .p = p, .out = out, .iters = iters }, Ctx.run);
    var total: i64 = 0;
    for (ws) |w| total += w.total;
    return total;
}

// ---- M4 frame modem: fixed point and its float twin --------------------------

const FxHandle = union(enum) { q: *fxmodem.FxModem(true), f: *fxmodem.FxModem(false) };

/// fixed != 0: the integer modem (samples as int16 I/Q pairs); 0: the
/// same algorithm in f32 (complex float samples).
export fn nphy_fx_create(cfg: *const fxmodem.FxConfig, fixed: u32) ?*FxHandle {
    const h = alloc.create(FxHandle) catch return null;
    h.* = if (fixed != 0)
        .{ .q = fxmodem.FxModem(true).create(alloc, cfg) catch {
            alloc.destroy(h);
            return null;
        } }
    else
        .{ .f = fxmodem.FxModem(false).create(alloc, cfg) catch {
            alloc.destroy(h);
            return null;
        } };
    return h;
}

export fn nphy_fx_destroy(h: *FxHandle) void {
    switch (h.*) {
        inline else => |m| m.destroy(),
    }
    alloc.destroy(h);
}

export fn nphy_fx_num_samples(h: *FxHandle) u32 {
    return switch (h.*) {
        inline else => |m| m.numSamples(),
    };
}

/// The product estimator in floating point (fixed point quantises it):
/// bt [F * taps] complex, start [F], at [S * Npsym], r [S * F].
export fn nphy_fx_set_filter(h: *FxHandle, bt: [*]const C, start: [*]const u32, at: [*]const f32, r: [*]const f32) i32 {
    switch (h.*) {
        inline else => |m| m.setFilter(bt, start, at, r) catch return -1,
    }
    return 0;
}

/// info: [batch][num_codewords * k] -> out: [batch][num_samples] samples.
export fn nphy_fx_tx(h: *FxHandle, info: [*]const u8, batch: u32, out: *anyopaque) void {
    switch (h.*) {
        inline else => |m| m.tx(info, batch, @ptrCast(@alignCast(out))),
    }
}

/// y: [batch][len_in] samples; llr_scale: [batch] (see FxModem.rx);
/// out: [batch][num_codewords * k]; llr_out, probe fields optional.
export fn nphy_fx_rx(h: *FxHandle, y: *const anyopaque, len_in: u32, l_min: i32, llr_scale: [*]const f64, batch: u32, out: [*]u8, llr_out: ?[*]f32, probe: *const fxmodem.Probe) i32 {
    switch (h.*) {
        inline else => |m| m.rx(@ptrCast(@alignCast(y)), len_in, l_min, llr_scale, batch, out, llr_out, probe.*) catch return -1,
    }
    return 0;
}

export fn nphy_fx_stats(h: *FxHandle, out: *[2]u64, reset: u32) void {
    switch (h.*) {
        inline else => |m| m.stats(out, reset != 0),
    }
}

/// The fixed-point FFT on count transforms of n int32 I/Q pairs (16-bit values), in place.
export fn nphy_fx_fft(n: u32, x: [*]fx.Ci, count: u32, inverse: u32) i32 {
    var f = fx.Fft.init(alloc, n) catch return -1;
    defer f.deinit(alloc);
    for (0..count) |i| f.runFixed(x[i * n ..][0..n], inverse != 0);
    return 0;
}

/// CORDIC vectoring on count (x, y) pairs: out [count][2] (magnitude times K, angle).
export fn nphy_fx_cordic_vector(xy: [*]const [2]i64, count: u32, out: [*][2]i64) void {
    for (0..count) |i| {
        const v = fx.vector(xy[i][0], xy[i][1]);
        out[i] = .{ v.mag, v.ang };
    }
}

/// CORDIC rotation: out [count] 2^15 (cos, sin) of binary angles.
export fn nphy_fx_cossin(ang: [*]const i64, count: u32, out: [*]fx.Ci) void {
    for (0..count) |i| out[i] = fx.cossin(ang[i]);
}

/// Gold-sequence pilot signs and the interleaver, for the table tests.
export fn nphy_fx_tables(num_pilots: u32, pilot_seed: u32, signs: [*]i8, frame_bits: u32, perm_seed: u32, perm: [*]u32) i32 {
    const scratch = alloc.alloc(u8, 2 * num_pilots) catch return -1;
    defer alloc.free(scratch);
    tables.pilotSigns(pilot_seed, signs[0 .. 2 * num_pilots], scratch);
    tables.interleaver(perm_seed, perm[0..frame_bits]);
    return 0;
}

test {
    _ = @import("tables.zig");
    _ = @import("fxmodem.zig");
    _ = @import("fx.zig");
    _ = @import("fxldpc.zig");
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
