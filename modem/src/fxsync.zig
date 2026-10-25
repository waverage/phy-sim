//! The synchroniser (M5) in fixed point, the model the RTL is built
//! against. harness/sync.py holds the same algorithm in floating point and
//! harness/twin.py (TwinSync) its bit-exact NumPy twin.
//!
//! Preamble: one slot of reps x period samples (9 x 32 = 288), an OFDM
//! symbol on the subcarriers that are multiples of fft_size / period, QPSK
//! from the Gold sequence (stf_seed), through the 16-bit IFFT at the data
//! symbols' amplitude per unit and the DAC.
//!
//! Synchroniser, on 12-bit ADC samples in a buffer:
//!  1. detect: P = sum r[n + period] conj(r[n]), E = sum |r[n]|^2 +
//!     |r[n + period]|^2 over `window` samples, running sums; the frame is
//!     there at the first d with 4 |P|^2 256 > threshold_q8 E^2 (P and E
//!     shifted down by metric_shift first);
//!  2. coarse timing at the largest |P|^2 within one preamble after that,
//!     coarse CFO from its CORDIC angle: an NCO increment per sample of
//!     angle * 2^32 / (65536 period);
//!  3. fine timing: the first pilot symbol's window derotated by the coarse
//!     NCO, FFT, times the conjugate pilot, IFFT: the channel impulse
//!     response. The first path is the earliest tap within a CP before the
//!     peak whose power is at least the peak's / 2^first_path_shift and
//!     noise_floor times the mean of the taps more than a CP from the peak;
//!     a peak below cir_min_ratio times the mean power rejects the frame;
//!  4. medium CFO: sum r[n + N] conj(r[n]) over every symbol's cyclic
//!     prefix, its angle less fft_size / period times the coarse one
//!     (wrapped) is the residual over N samples;
//!  5. align: the frame from backoff samples before the first path,
//!     derotated by the NCO (32-bit phase, its top 16 bits to the CORDIC).
//!     The output feeds FxModem.rx with l_min = 0.

const std = @import("std");
const fx = @import("fx.zig");
const tables = @import("tables.zig");
const par = @import("par.zig");
const Ci = fx.Ci;

pub const stf_seed: u32 = 0x4E533C;

pub const SyncConfig = extern struct {
    fft_size: u32,
    cp_len: u32,
    guard_lo: u32,
    guard_hi: u32,
    dc_null: u32,
    num_symbols: u32,
    pilot_symbol: u32, // the first pilot OFDM symbol: its pilots are the first F of the sequence
    period: u32,
    reps: u32,
    window: u32,
    metric_shift: u32,
    threshold_q8: u32,
    first_path_shift: u32,
    noise_floor: u32,
    cir_min_ratio: u32,
    backoff: u32,
    cp_cfo: u32,
    tx_scale: u32,
    dac_bits: u32,
    stf_seed: u32,
    pilot_seed: u32,
    num_threads: u32,
};

/// Per buffer. detected: bit 0 the metric crossed, bit 1 the CIR check
/// passed; the frame is synchronised when both are set.
pub const SyncOut = extern struct {
    detected: u32,
    start: i32, // first sample of the frame (after the preamble) through the first path
    peak: i32, // coarse timing: d at the largest |P|^2
    tau: i32, // first path of the CIR relative to the coarse frame start
    ang_coarse: i32, // angle of P, 65536 per turn
    ang_cp: i32, // residual over fft_size samples, wrapped
    inc: i32, // NCO increment per sample, 2^32 per turn
    inc_coarse: i32,
    p_re: i64, // P and E at the peak, shifted by metric_shift
    p_im: i64,
    e: i64,
    cir_peak: i64,
    cir_sum: i64,
};

pub const Sync = struct {
    alloc: std.mem.Allocator,
    cfg: SyncConfig,
    f: u32,
    slot: u32,
    pre: u32,
    fft: fx.Fft,
    bin: []u32, // [F] FFT bin of each effective subcarrier
    psign: []i8, // [F * 2] the first pilot symbol
    ws: []Workspace,

    const Workspace = struct { x: []Ci, pw: []i64 };

    pub fn create(alloc: std.mem.Allocator, cfg: *const SyncConfig) !*Sync {
        const s = try alloc.create(Sync);
        errdefer alloc.destroy(s);
        const c = cfg.*;
        if (c.period == 0 or c.fft_size > 512 or c.fft_size % c.period != 0 or c.period * c.reps != c.fft_size + c.cp_len) return error.BadConfig;
        if (c.window == 0 or c.window > c.period * (c.reps - 1)) return error.BadConfig;
        if (c.guard_lo + c.guard_hi + @intFromBool(c.dc_null != 0) >= c.fft_size) return error.BadConfig;
        if (c.pilot_symbol >= c.num_symbols or c.backoff > c.cp_len or c.dac_bits < 2 or c.dac_bits > 16) return error.BadConfig;
        if (c.threshold_q8 > 256 or c.metric_shift > 32 or c.first_path_shift > 16) return error.BadConfig;
        s.alloc = alloc;
        s.cfg = c;
        s.slot = c.fft_size + c.cp_len;
        s.pre = c.period * c.reps;
        s.fft = try fx.Fft.init(alloc, c.fft_size);
        s.f = c.fft_size - c.guard_lo - c.guard_hi - @intFromBool(c.dc_null != 0);
        s.bin = try alloc.alloc(u32, s.f);
        var f: u32 = 0;
        for (c.guard_lo..c.fft_size - c.guard_hi) |i| {
            if (c.dc_null != 0 and i == c.fft_size / 2) continue;
            s.bin[f] = @intCast((i + c.fft_size / 2) % c.fft_size);
            f += 1;
        }
        s.psign = try alloc.alloc(i8, 2 * s.f);
        const scratch = try alloc.alloc(u8, 2 * s.f);
        defer alloc.free(scratch);
        tables.pilotSigns(c.pilot_seed, s.psign, scratch);
        s.ws = try alloc.alloc(Workspace, par.threadCount(c.num_threads));
        for (s.ws) |*w| w.* = .{ .x = try alloc.alloc(Ci, c.fft_size), .pw = try alloc.alloc(i64, c.fft_size) };
        return s;
    }

    pub fn destroy(s: *Sync) void {
        for (s.ws) |w| {
            s.alloc.free(w.x);
            s.alloc.free(w.pw);
        }
        s.alloc.free(s.ws);
        s.alloc.free(s.bin);
        s.alloc.free(s.psign);
        s.fft.deinit(s.alloc);
        s.alloc.destroy(s);
    }

    /// The preamble: pre samples, int16 I/Q at the DAC.
    pub fn preamble(s: *const Sync, out: [][2]i16) !void {
        const c = s.cfg;
        const n = c.fft_size;
        const step = n / c.period;
        var ntones: u32 = 0;
        for (c.guard_lo..n - c.guard_hi) |i| {
            const k = @as(i64, @intCast(i)) - n / 2;
            if (k != 0 and @mod(k, step) == 0) ntones += 1;
        }
        const signs = try s.alloc.alloc(i8, 2 * ntones);
        defer s.alloc.free(signs);
        const scratch = try s.alloc.alloc(u8, 2 * ntones);
        defer s.alloc.free(scratch);
        tables.pilotSigns(c.stf_seed, signs, scratch);
        const ts: f64 = @floatFromInt(c.tx_scale);
        const ratio = @as(f64, @floatFromInt(s.f)) / @as(f64, @floatFromInt(ntones));
        const amp: i32 = @intFromFloat(@floor(ts * @sqrt(ratio) / @sqrt(2.0) + 0.5));
        const x = try s.alloc.alloc(Ci, n);
        defer s.alloc.free(x);
        @memset(x, Ci.zero);
        var t: usize = 0;
        for (c.guard_lo..n - c.guard_hi) |i| {
            const k = @as(i64, @intCast(i)) - n / 2;
            if (k == 0 or @mod(k, step) != 0) continue;
            x[@intCast(@mod(k, n))] = .{ .re = signs[2 * t] * amp, .im = signs[2 * t + 1] * amp };
            t += 1;
        }
        s.fft.runFixed(x, true);
        for (out[0..s.pre], 0..) |*o, i| {
            const v = x[i % c.period];
            o.* = .{ @intCast(fx.sat(v.re, @intCast(c.dac_bits))), @intCast(fx.sat(v.im, @intCast(c.dac_bits))) };
        }
    }

    inline fn ci(v: [2]i16) Ci {
        return .{ .re = v[0], .im = v[1] };
    }

    /// a * conj(b)
    inline fn mulConj(a: Ci, b: Ci) [2]i64 {
        return .{ @as(i64, a.re) * b.re + @as(i64, a.im) * b.im, @as(i64, a.im) * b.re - @as(i64, a.re) * b.im };
    }

    /// CORDIC angle of a wide complex sum, normalised to 30 bits first.
    fn angle(re: i64, im: i64) i64 {
        const mx: u64 = @max(@abs(re), @abs(im));
        const bl: u32 = 64 - @clz(mx);
        const sh: u6 = if (bl > 30) @intCast(bl - 30) else 0;
        return fx.vector(re >> sh, im >> sh).ang;
    }

    /// NCO: the CORDIC (cos, sin) of -n inc, from the top 16 bits of the 32-bit phase.
    inline fn nco(n: i64, inc: i64) Ci {
        return fx.cossin(@mod(-n * inc, @as(i64, 1) << 32) >> 16);
    }

    fn wrap(a: i64) i64 {
        var v = @mod(a, fx.turn);
        if (v > fx.turn / 2) v -= fx.turn;
        return v;
    }

    pub fn runOne(s: *const Sync, w: *Workspace, r: []const [2]i16, out: *SyncOut) void {
        const c = s.cfg;
        const L: i64 = @intCast(r.len);
        const P: usize = c.period;
        const W: usize = c.window;
        const N: usize = c.fft_size;
        out.* = std.mem.zeroes(SyncOut);
        if (r.len < P + W + 1) return;
        const dmax: usize = r.len - P - W;
        const ms: u6 = @intCast(c.metric_shift);

        // 1-2. detection and coarse timing, running sums
        var pr: i64 = 0;
        var pi: i64 = 0;
        var e: i64 = 0;
        for (0..W) |n| {
            const v = mulConj(ci(r[n + P]), ci(r[n]));
            pr += v[0];
            pi += v[1];
            e += energy(r[n]) + energy(r[n + P]);
        }
        var d: usize = 0;
        var found = false;
        var best: i64 = -1;
        var best_d: usize = 0;
        var best_p = [2]i64{ 0, 0 };
        var best_ps = [3]i64{ 0, 0, 0 };
        var d0: usize = 0;
        while (true) {
            const psr = pr >> ms;
            const psi = pi >> ms;
            const es = e >> ms;
            const mag = psr * psr + psi * psi;
            if (!found) {
                const lhs: i128 = @as(i128, mag) * 4 * 256;
                const rhs: i128 = @as(i128, c.threshold_q8) * es * es;
                if (lhs > rhs) {
                    found = true;
                    d0 = d;
                }
            }
            if (found) {
                if (mag > best) {
                    best = mag;
                    best_d = d;
                    best_p = .{ pr, pi };
                    best_ps = .{ psr, psi, es };
                }
                if (d == d0 + s.pre or d == dmax) break;
            } else if (d == dmax) break;
            // slide the window by one
            const a = mulConj(ci(r[d + P]), ci(r[d]));
            const b = mulConj(ci(r[d + W + P]), ci(r[d + W]));
            pr += b[0] - a[0];
            pi += b[1] - a[1];
            e += energy(r[d + W]) + energy(r[d + W + P]) - energy(r[d]) - energy(r[d + P]);
            d += 1;
        }
        if (!found) return;
        out.detected = 1;
        out.peak = @intCast(best_d);
        out.p_re = best_ps[0];
        out.p_im = best_ps[1];
        out.e = best_ps[2];
        const a1 = angle(best_p[0], best_p[1]);
        const inc1: i64 = @divExact(a1 * (@as(i64, 1) << 32), fx.turn * @as(i64, @intCast(P)));
        out.ang_coarse = @intCast(a1);
        out.inc_coarse = @intCast(inc1);

        // 3. channel impulse response from the first pilot symbol
        const coarse: i64 = @as(i64, @intCast(best_d)) + s.pre;
        const ws0 = coarse + @as(i64, c.pilot_symbol) * s.slot + c.cp_len;
        for (w.x, 0..) |*x, m| {
            const n = std.math.clamp(ws0 + @as(i64, @intCast(m)), 0, L - 1);
            x.* = fx.rot16(ci(r[@intCast(n)]), nco(n, inc1));
        }
        s.fft.runFixed(w.x, false);
        var z: [512]Ci = undefined;
        @memset(z[0..N], Ci.zero);
        for (s.bin, 0..) |b, f| {
            const y = w.x[b];
            const sa: i64 = s.psign[2 * f];
            const sb: i64 = s.psign[2 * f + 1];
            z[b] = .{ .re = @intCast(fx.sat(fx.rshr(sa * y.re + sb * y.im, 1), 16)), .im = @intCast(fx.sat(fx.rshr(sa * y.im - sb * y.re, 1), 16)) };
        }
        @memcpy(w.x, z[0..N]);
        s.fft.runFixed(w.x, true);
        var total: i64 = 0;
        var pk: usize = 0;
        for (w.x, w.pw, 0..) |x, *p, t| {
            p.* = @as(i64, x.re) * x.re + @as(i64, x.im) * x.im;
            total += p.*;
            if (p.* > w.pw[pk]) pk = t;
        }
        const cp: usize = c.cp_len;
        var near: i64 = 0;
        for (0..2 * cp + 1) |j| near += w.pw[(pk + N - cp + j) % N];
        const floor_sum = total - near;
        const floor_n: i64 = @intCast(N - (2 * cp + 1));
        const pmax = w.pw[pk];
        var first: usize = pk;
        for (0..cp) |j| {
            const t = (pk + N - (cp - 1) + j) % N;
            const p = w.pw[t];
            if (j == cp - 1 or ((p << @intCast(c.first_path_shift)) >= pmax and p * floor_n >= @as(i64, c.noise_floor) * floor_sum)) {
                first = t;
                break;
            }
        }
        const tau: i64 = if (first >= N / 2) @as(i64, @intCast(first)) - @as(i64, @intCast(N)) else @intCast(first);
        out.tau = @intCast(tau);
        out.cir_peak = pmax;
        out.cir_sum = total;
        if (pmax * @as(i64, @intCast(N)) >= @as(i64, c.cir_min_ratio) * total) out.detected |= 2;
        const start = coarse + tau;
        out.start = @intCast(start);

        // 4. medium CFO from the cyclic prefixes
        var inc = inc1;
        if (c.cp_cfo != 0) {
            var qr: i64 = 0;
            var qi: i64 = 0;
            for (0..c.num_symbols) |sym| {
                for (0..cp) |i| {
                    const n0 = std.math.clamp(start + @as(i64, @intCast(sym * s.slot + i)), 0, L - @as(i64, @intCast(N)) - 1);
                    const n: usize = @intCast(n0);
                    const v = mulConj(ci(r[n + N]), ci(r[n]));
                    qr += v[0];
                    qi += v[1];
                }
            }
            const acp = angle(qr, qi);
            const delta = wrap(acp - a1 * @as(i64, @intCast(N / P)));
            out.ang_cp = @intCast(delta);
            inc = inc1 + delta * @divExact(@as(i64, 1) << 32, fx.turn * @as(i64, @intCast(N)));
        }
        out.inc = @intCast(inc);
    }

    inline fn energy(v: [2]i16) i64 {
        return @as(i64, v[0]) * v[0] + @as(i64, v[1]) * v[1];
    }

    /// out [len]: the buffer from start - backoff, derotated by inc; zeros outside it.
    pub fn alignOne(s: *const Sync, r: []const [2]i16, start: i32, inc: i32, out: [][2]i16) void {
        const L: i64 = @intCast(r.len);
        for (out, 0..) |*o, m| {
            const n = @as(i64, start) - s.cfg.backoff + @as(i64, @intCast(m));
            if (n < 0 or n >= L) {
                o.* = .{ 0, 0 };
                continue;
            }
            const v = fx.rot16(ci(r[@intCast(n)]), nco(n, inc));
            o.* = .{ @intCast(v.re), @intCast(v.im) };
        }
    }

    pub fn run(s: *Sync, buf: [*]const [2]i16, len: u32, batch: u32, out: [*]SyncOut) void {
        const Ctx = struct {
            s: *Sync,
            buf: [*]const [2]i16,
            len: u32,
            out: [*]SyncOut,
            fn go(x: @This(), w: *Workspace, i: usize) void {
                x.s.runOne(w, x.buf[i * x.len ..][0..x.len], &x.out[i]);
            }
        };
        par.run(Workspace, s.ws, batch, Ctx{ .s = s, .buf = buf, .len = len, .out = out }, Ctx.go);
    }

    pub fn alignBatch(s: *Sync, buf: [*]const [2]i16, len: u32, batch: u32, start: [*]const i32, inc: [*]const i32, out_len: u32, out: [*][2]i16) void {
        const Ctx = struct {
            s: *Sync,
            buf: [*]const [2]i16,
            len: u32,
            start: [*]const i32,
            inc: [*]const i32,
            out_len: u32,
            out: [*][2]i16,
            fn go(x: @This(), _: *Workspace, i: usize) void {
                x.s.alignOne(x.buf[i * x.len ..][0..x.len], x.start[i], x.inc[i], x.out[i * x.out_len ..][0..x.out_len]);
            }
        };
        par.run(Workspace, s.ws, batch, Ctx{ .s = s, .buf = buf, .len = len, .start = start, .inc = inc, .out_len = out_len, .out = out }, Ctx.go);
    }
};

pub fn defaultConfig() SyncConfig {
    return .{ .fft_size = 256, .cp_len = 32, .guard_lo = 28, .guard_hi = 28, .dc_null = 1, .num_symbols = 14, .pilot_symbol = 2, .period = 32, .reps = 9, .window = 256, .metric_shift = 8, .threshold_q8 = 26, .first_path_shift = 3, .noise_floor = 10, .cir_min_ratio = 8, .backoff = 4, .cp_cfo = 1, .tx_scale = 512, .dac_bits = 12, .stf_seed = stf_seed, .pilot_seed = tables.pilot_seed, .num_threads = 1 };
}

test "sync finds a noiseless frame with a carrier offset" {
    const alloc = std.testing.allocator;
    const cfg = defaultConfig();
    const s = try Sync.create(alloc, &cfg);
    defer s.destroy();
    var pre: [288][2]i16 = undefined;
    try s.preamble(&pre);
    // preamble, then noise-like samples
    const lead = 300;
    var buf: [lead + 288 * 18][2]i16 = @splat(.{ 0, 0 });
    const inc_true: i64 = 1 << 22; // 1/1024 turn per sample, 9.8 kHz
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    for (0..288 * 16) |i| {
        const n: i64 = lead + @as(i64, @intCast(i));
        const x: Ci = if (i < 288) .{ .re = pre[i][0], .im = pre[i][1] } else .{ .re = rnd.intRangeAtMost(i32, -400, 400), .im = rnd.intRangeAtMost(i32, -400, 400) };
        const v = fx.rot16(x, Sync.nco(n, -inc_true));
        buf[@intCast(n)] = .{ @intCast(v.re), @intCast(v.im) };
    }
    var out: SyncOut = undefined;
    s.runOne(&s.ws[0], &buf, &out);
    try std.testing.expect(out.detected & 1 != 0);
    try std.testing.expectEqual(@as(i32, lead), out.peak);
    try std.testing.expect(@abs(@as(i64, out.inc_coarse) - inc_true) < 1 << 14);
}
