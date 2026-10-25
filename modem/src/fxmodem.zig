//! The product modem (M4): the whole frame in integer arithmetic with
//! chosen widths, the model the RTL is built against. FxModem(true) is the
//! fixed-point modem, FxModem(false) the same algorithm in floating point
//! (no rounding, no saturation, exact twiddles and filter taps), which is
//! what the fixed-point loss is measured against. harness/twin.py is the
//! bit-exact NumPy twin of FxModem(true).
//!
//! TX: 5G LDPC -> frame interleaver -> integer QAM levels (amplitude
//! tx_scale per unit) and Gold-sequence QPSK pilots -> 16-bit IFFT -> CP ->
//! DAC saturation to dac_bits.
//!
//! RX, from ADC samples (16-bit containers):
//!  1. FFT per symbol (fx.Fft, 16-bit), timing phase l_min as a twiddle.
//!  2. LS on the pilot symbols without division: ls2 = Y (a - jb) for the
//!     pilot (a + jb) / sqrt(2) (17 bits; the 1/sqrt(2) is in the taps).
//!  3. Frequency filter per pilot symbol: L complex taps (Q15) per
//!     subcarrier over a window of pilots, 16-bit out.
//!  4. Phase tracker (optional): per pilot symbol t, z_t = sum_f H_t conj(sum H),
//!     CORDIC angle and magnitude, one weighted phase ramp per frame, CORDIC
//!     cos/sin per symbol, rotate Y and H.
//!  5. Time filter: real Q14 weights over the pilot symbols per data symbol.
//!  6. Max-log demapper without division: with z = Y conj(h) and
//!     u = a |h|^2, the metric of level L is u L^2 - 2 z L, and
//!     LLR = (min over bit 0 - min over bit 1) * scale * r, scale = a /
//!     (N0 LSB) per frame as a 16-bit mantissa and shift, r per RE (Q15) =
//!     1 / (1 + estimator noise gain).
//!  7. Deinterleave, layered offset min-sum (fxldpc.zig).

const std = @import("std");
const ldpc = @import("ldpc.zig");
const fxldpc = @import("fxldpc.zig");
const fx = @import("fx.zig");
const tables = @import("tables.zig");
const par = @import("par.zig");
const C = @import("ofdm.zig").C;
const Ci = fx.Ci;

pub const max_symbols = 64;
pub const max_pilot_symbols = 8;
pub const max_bits = 8;

pub const FxConfig = extern struct {
    fft_size: u32,
    cp_len: u32,
    guard_lo: u32,
    guard_hi: u32,
    dc_null: u32,
    num_symbols: u32,
    bits_per_symbol: u32,
    num_codewords: u32,
    n: u32,
    k: u32,
    phase_track: u32,
    num_threads: u32,
    tx_scale: u32, // QAM amplitude per unit at the IFFT input
    dac_bits: u32,
    freq_taps: u32,
    pilot_seed: u32,
    perm_seed: u32,
    track_gain: u32, // Q15: the fraction of the fitted phase ramp removed
    dec: fxldpc.Params,
    pilot_mask: [*]const u8, // [S * F], block pilots only
};

/// Optional per-frame intermediate results, as f32 (integers in fixed point).
pub const Probe = extern struct {
    y: ?[*]f32, // [batch][S * F][2] after the phase tracker
    h: ?[*]f32, // [batch][num_data][2] channel estimate at the data REs
    phi: ?[*]f32, // [batch][S] tracker phase per symbol (binary units / radians)
};

pub fn FxModem(comptime fixed: bool) type {
    return struct {
        const Self = @This();
        pub const Cx = if (fixed) Ci else C;
        pub const Sample = if (fixed) [2]i16 else C;
        const T = if (fixed) i32 else f32;
        const Dec = fxldpc.Decoder(T);

        const Workspace = struct {
            sym: []Cx, // one FFT
            y: []Cx, // [S * F]
            hf: []Cx, // [Npsym * F] frequency-filtered estimates
            llr_tx: []T,
            llr: []T,
            bits: []u8,
            enc: []u8,
            dec: Dec,
            iters: u64,
            codewords: u64,
        };

        alloc: std.mem.Allocator,
        s: u32,
        nfft: u32,
        cp: u32,
        f: u32,
        bps: u32,
        frame_bits: u32,
        num_cw: u32,
        num_data: u32,
        npsym: u32,
        psym: [max_pilot_symbols]u32,
        sum_psym: i64,
        phase_track: bool,
        track_gain: i64,
        dac_bits: u32,
        taps: u32,
        dec_params: fxldpc.Params,
        code: ldpc.Code,
        fft: fx.Fft,
        bin: []u32,
        is_pilot: []u8,
        data_re: []u32,
        signs: []i8, // [Npsym * F * 2]
        perm: []u32,
        // QAM: integer level per label, TX amplitude per label, demapper
        levels: [1 << (max_bits / 2)]i64,
        tx_level: [1 << (max_bits / 2)]if (fixed) i32 else f32,
        tx_pilot: if (fixed) i32 else f32,
        a_q: i64, // round(a 2^16)
        a_f: f64,
        // estimator
        bt: []Cx, // [F * taps] frequency taps (Q15 / float)
        start: []u32, // [F]
        at: []T, // [S * Npsym] time weights (Q14 / float)
        r: []T, // [num_data] LLR weight (Q15 / float)
        have_filter: bool,
        rot: []Cx, // [N] timing phase per bin
        rot_l_min: i32,
        ws: []Workspace,

        pub fn create(alloc: std.mem.Allocator, cfg: *const FxConfig) !*Self {
            const m = try alloc.create(Self);
            errdefer alloc.destroy(m);
            m.alloc = alloc;
            if (cfg.num_symbols == 0 or cfg.num_symbols > max_symbols) return error.BadConfig;
            if (cfg.guard_lo + cfg.guard_hi + @intFromBool(cfg.dc_null != 0) >= cfg.fft_size) return error.BadConfig;
            if (cfg.bits_per_symbol == 0 or cfg.bits_per_symbol % 2 != 0 or cfg.bits_per_symbol > max_bits) return error.BadConfig;
            if (cfg.dac_bits < 2 or cfg.dac_bits > 16) return error.BadConfig;
            m.s = cfg.num_symbols;
            m.nfft = cfg.fft_size;
            m.cp = cfg.cp_len;
            m.f = cfg.fft_size - cfg.guard_lo - cfg.guard_hi - @intFromBool(cfg.dc_null != 0);
            m.bps = cfg.bits_per_symbol;
            m.num_cw = cfg.num_codewords;
            m.phase_track = cfg.phase_track != 0;
            if (cfg.track_gain > 32768) return error.BadConfig;
            m.track_gain = cfg.track_gain;
            m.dac_bits = cfg.dac_bits;
            m.taps = cfg.freq_taps;
            if (m.taps == 0 or m.taps > m.f) return error.BadConfig;
            m.dec_params = cfg.dec;
            m.code = try ldpc.Code.init(cfg.k, cfg.n);
            m.fft = try fx.Fft.init(alloc, cfg.fft_size);

            m.bin = try alloc.alloc(u32, m.f);
            var f: u32 = 0;
            for (cfg.guard_lo..cfg.fft_size - cfg.guard_hi) |i| {
                if (cfg.dc_null != 0 and i == cfg.fft_size / 2) continue;
                m.bin[f] = @intCast((i + cfg.fft_size / 2) % cfg.fft_size);
                f += 1;
            }

            // pilots: whole OFDM symbols only (M2: no continual pilots)
            const sf = m.s * m.f;
            m.is_pilot = try alloc.dupe(u8, cfg.pilot_mask[0..sf]);
            m.npsym = 0;
            m.sum_psym = 0;
            for (0..m.s) |s| {
                const row = m.is_pilot[s * m.f ..][0..m.f];
                if (std.mem.allEqual(u8, row, 1)) {
                    if (m.npsym == max_pilot_symbols) return error.BadConfig;
                    m.psym[m.npsym] = @intCast(s);
                    m.npsym += 1;
                    m.sum_psym += @intCast(s);
                } else if (!std.mem.allEqual(u8, row, 0)) return error.BadConfig;
            }
            if (m.npsym == 0) return error.BadConfig;
            m.num_data = (m.s - m.npsym) * m.f;
            m.frame_bits = m.num_data * m.bps;
            if (m.num_cw * cfg.n > m.frame_bits) return error.BadConfig;
            m.data_re = try alloc.alloc(u32, m.num_data);
            var nd: u32 = 0;
            for (m.is_pilot, 0..) |p, i| {
                if (p == 0) {
                    m.data_re[nd] = @intCast(i);
                    nd += 1;
                }
            }
            const np = m.npsym * m.f;
            m.signs = try alloc.alloc(i8, 2 * np);
            const scratch = try alloc.alloc(u8, 2 * np);
            defer alloc.free(scratch);
            tables.pilotSigns(cfg.pilot_seed, m.signs, scratch);
            m.perm = try alloc.alloc(u32, m.frame_bits);
            tables.interleaver(cfg.perm_seed, m.perm);

            // QAM tables
            const mdim = m.bps / 2;
            var sq: i64 = 0;
            for (0..(@as(usize, 1) << @intCast(mdim))) |lab| {
                m.levels[lab] = pamGray(@intCast(lab), mdim);
                sq += m.levels[lab] * m.levels[lab];
            }
            // norm = 2 mean(L^2), a = 1 / sqrt(norm)
            const norm: f64 = 2.0 * @as(f64, @floatFromInt(sq)) / @as(f64, @floatFromInt(@as(u32, 1) << @intCast(mdim)));
            m.a_f = 1.0 / @sqrt(norm);
            m.a_q = @intFromFloat(@floor(65536.0 / @sqrt(norm) + 0.5));
            const ts: f64 = @floatFromInt(cfg.tx_scale);
            for (0..(@as(usize, 1) << @intCast(mdim))) |lab| {
                const lv: f64 = @floatFromInt(m.levels[lab]);
                m.tx_level[lab] = if (fixed) @intFromFloat(@floor(ts * lv / @sqrt(norm) + 0.5)) else @floatCast(ts * lv / @sqrt(norm));
            }
            m.tx_pilot = if (fixed) @intFromFloat(@floor(ts / @sqrt(2.0) + 0.5)) else @floatCast(ts / @sqrt(2.0));

            m.bt = try alloc.alloc(Cx, m.f * m.taps);
            m.start = try alloc.alloc(u32, m.f);
            m.at = try alloc.alloc(T, m.s * m.npsym);
            m.r = try alloc.alloc(T, m.num_data);
            m.have_filter = false;
            m.rot = try alloc.alloc(Cx, m.nfft);
            m.setRot(0);

            m.ws = try alloc.alloc(Workspace, par.threadCount(cfg.num_threads));
            for (m.ws) |*w| {
                w.* = .{
                    .sym = try alloc.alloc(Cx, m.nfft),
                    .y = try alloc.alloc(Cx, sf),
                    .hf = try alloc.alloc(Cx, np),
                    .llr_tx = try alloc.alloc(T, m.frame_bits),
                    .llr = try alloc.alloc(T, m.frame_bits),
                    .bits = try alloc.alloc(u8, m.frame_bits),
                    .enc = try alloc.alloc(u8, m.code.n_ldpc),
                    .dec = try Dec.init(alloc),
                    .iters = 0,
                    .codewords = 0,
                };
            }
            return m;
        }

        pub fn destroy(m: *Self) void {
            const a = m.alloc;
            for (m.ws) |*w| {
                inline for (.{ "sym", "y", "hf", "llr_tx", "llr", "bits", "enc" }) |name| a.free(@field(w, name));
                w.dec.deinit(a);
            }
            a.free(m.ws);
            m.fft.deinit(a);
            inline for (.{ "bin", "is_pilot", "data_re", "signs", "perm", "bt", "start", "at", "r", "rot" }) |name| {
                a.free(@field(m, name));
            }
            a.destroy(m);
        }

        fn pamGray(label: u32, mdim: u32) i64 {
            const b0: i64 = @intCast((label >> @intCast(mdim - 1)) & 1);
            if (mdim == 1) return 1 - 2 * b0;
            const rest = label & ((@as(u32, 1) << @intCast(mdim - 1)) - 1);
            return (1 - 2 * b0) * ((@as(i64, 1) << @intCast(mdim - 1)) - pamGray(rest, mdim - 1));
        }

        pub fn numSamples(m: *const Self) u32 {
            return m.s * (m.nfft + m.cp);
        }

        /// The product estimator (harness/design.py), in floating point;
        /// fixed point quantises it here: taps Q15, time weights Q14, r Q15.
        /// bt [F * taps] complex, start [F], at [S * Npsym], r [S * F].
        pub fn setFilter(m: *Self, bt: [*]const C, start: [*]const u32, at: [*]const f32, r: [*]const f32) !void {
            for (0..m.f) |f| {
                if (start[f] + m.taps > m.f) return error.BadFilter;
                m.start[f] = start[f];
            }
            for (m.bt, bt[0 .. m.f * m.taps]) |*d, v| {
                d.* = if (fixed) .{ .re = @intCast(fx.sat(q(v.re, 15), 16)), .im = @intCast(fx.sat(q(v.im, 15), 16)) } else v;
            }
            for (m.at, at[0 .. m.s * m.npsym]) |*d, v| d.* = if (fixed) @intCast(fx.sat(q(v, 14), 16)) else v;
            for (m.data_re, m.r) |re, *d| d.* = if (fixed) @intCast(std.math.clamp(q(r[re], 15), 0, 32768)) else r[re];
            m.have_filter = true;
        }

        inline fn q(v: f32, frac: u6) i64 {
            return @intFromFloat(@floor(@as(f64, v) * @as(f64, @floatFromInt(@as(i64, 1) << frac)) + 0.5));
        }

        fn setRot(m: *Self, l_min: i32) void {
            // exp(-2 pi i l_min k / N) = W^(l_min k mod N)
            const n: i64 = m.nfft;
            for (m.rot, 0..) |*r, k| {
                const j: usize = @intCast(@mod(@as(i64, l_min) * @as(i64, @intCast(k)), n));
                const half = m.nfft / 2;
                if (fixed) {
                    const w = m.fft.twq[j % half];
                    r.* = if (j < half) w else .{ .re = -w.re, .im = -w.im };
                } else {
                    const w = m.fft.twf[j % half];
                    r.* = if (j < half) w else w.scale(-1);
                }
            }
            m.rot_l_min = l_min;
        }

        // ---- TX ----------------------------------------------------------------

        fn txFrame(m: *const Self, ws: *Workspace, info: []const u8, out: []Sample) void {
            const n = m.code.n;
            const k = m.code.k;
            @memset(ws.bits, 0);
            for (0..m.num_cw) |c| m.code.encode(info[c * k ..][0..k], ws.bits[c * n ..][0..n], ws.enc);
            const B = m.bps;
            const md = B / 2;
            var d: usize = 0;
            var p: usize = 0;
            const symlen = m.nfft + m.cp;
            for (0..m.s) |s| {
                @memset(ws.sym, std.mem.zeroes(Cx));
                for (0..m.f) |f| {
                    var v: Cx = undefined;
                    if (m.is_pilot[s * m.f + f] != 0) {
                        v = if (fixed)
                            .{ .re = @as(i32, m.signs[2 * p]) * m.tx_pilot, .im = @as(i32, m.signs[2 * p + 1]) * m.tx_pilot }
                        else
                            .{ .re = @as(f32, @floatFromInt(m.signs[2 * p])) * m.tx_pilot, .im = @as(f32, @floatFromInt(m.signs[2 * p + 1])) * m.tx_pilot };
                        p += 1;
                    } else {
                        var li: u32 = 0;
                        var lq: u32 = 0;
                        for (0..md) |j| {
                            li = (li << 1) | ws.bits[m.perm[d * B + 2 * j]];
                            lq = (lq << 1) | ws.bits[m.perm[d * B + 2 * j + 1]];
                        }
                        v = .{ .re = m.tx_level[li], .im = m.tx_level[lq] };
                        d += 1;
                    }
                    ws.sym[m.bin[f]] = v;
                }
                if (fixed) m.fft.runFixed(ws.sym, true) else m.fft.runFloat(ws.sym, true);
                const o = out[s * symlen ..][0..symlen];
                for (0..symlen) |i| {
                    const v = ws.sym[(i + m.nfft - m.cp) % m.nfft];
                    o[i] = if (fixed)
                        .{ @intCast(fx.sat(v.re, @intCast(m.dac_bits))), @intCast(fx.sat(v.im, @intCast(m.dac_bits))) }
                    else
                        v;
                }
            }
        }

        // ---- RX ----------------------------------------------------------------

        fn rxFrame(m: *const Self, ws: *Workspace, y_in: []const Sample, scale: f64, out: []u8, llr_out: ?[]f32, probe: Probe, frame: usize) void {
            const symlen = m.nfft + m.cp;
            const F = m.f;
            const S = m.s;
            // 1. FFT and timing phase
            for (0..S) |s| {
                for (ws.sym, y_in[s * symlen + m.cp ..][0..m.nfft]) |*d, v| {
                    d.* = if (fixed) .{ .re = v[0], .im = v[1] } else v;
                }
                if (fixed) m.fft.runFixed(ws.sym, false) else m.fft.runFloat(ws.sym, false);
                for (0..F) |f| {
                    const b = m.bin[f];
                    ws.y[s * F + f] = if (m.rot_l_min == 0) ws.sym[b] else cmulRot(ws.sym[b], m.rot[b]);
                }
            }
            // 2-3. LS and frequency filter on every pilot symbol
            const L = m.taps;
            for (0..m.npsym) |t| {
                const ysym = ws.y[m.psym[t] * F ..][0..F];
                const sg = m.signs[t * F * 2 ..][0 .. F * 2];
                for (0..F) |f| {
                    const taps = m.bt[f * L ..][0..L];
                    const st = m.start[f];
                    if (fixed) {
                        var ar: i64 = 0;
                        var ai: i64 = 0;
                        for (taps, 0..) |w, j| {
                            const y = ysym[st + j];
                            const a: i64 = sg[2 * (st + j)];
                            const b: i64 = sg[2 * (st + j) + 1];
                            const lr = a * y.re + b * y.im;
                            const li = a * y.im - b * y.re;
                            ar += @as(i64, w.re) * lr - @as(i64, w.im) * li;
                            ai += @as(i64, w.re) * li + @as(i64, w.im) * lr;
                        }
                        ws.hf[t * F + f] = .{ .re = @intCast(fx.sat(fx.rshr(ar, 15), 16)), .im = @intCast(fx.sat(fx.rshr(ai, 15), 16)) };
                    } else {
                        var ar: f64 = 0;
                        var ai: f64 = 0;
                        for (taps, 0..) |w, j| {
                            const y = ysym[st + j];
                            const a: f64 = @floatFromInt(sg[2 * (st + j)]);
                            const b: f64 = @floatFromInt(sg[2 * (st + j) + 1]);
                            const lr = a * y.re + b * y.im;
                            const li = a * y.im - b * y.re;
                            ar += w.re * lr - w.im * li;
                            ai += w.re * li + w.im * lr;
                        }
                        ws.hf[t * F + f] = .{ .re = @floatCast(ar), .im = @floatCast(ai) };
                    }
                }
            }
            // 4. phase tracker
            if (m.phase_track) m.track(ws, probe, frame) else if (probe.phi) |ph| @memset(ph[frame * S ..][0..S], 0);
            if (probe.y) |py| {
                for (ws.y[0 .. S * F], 0..) |v, i| {
                    py[(frame * S * F + i) * 2] = toF(v.re);
                    py[(frame * S * F + i) * 2 + 1] = toF(v.im);
                }
            }

            // 5-6. time filter and demapper
            const B = m.bps;
            const md = B / 2;
            const nl = @as(usize, 1) << @intCast(md);
            var mant: i64 = 0;
            var shift: u6 = 1;
            if (fixed) {
                if (scale > 0) {
                    const fr = std.math.frexp(scale);
                    mant = @intFromFloat(@floor(fr.significand * 65536.0 + 0.5));
                    var e = fr.exponent;
                    if (mant == 65536) {
                        mant = 32768;
                        e += 1;
                    }
                    shift = @intCast(std.math.clamp(16 - e, 1, 62));
                }
            }
            for (m.data_re, 0..) |re, d| {
                const s = re / F;
                const f = re % F;
                const aw = m.at[s * m.npsym ..][0..m.npsym];
                const y = ws.y[re];
                var h: Cx = undefined;
                if (fixed) {
                    var hr: i64 = 0;
                    var hi: i64 = 0;
                    for (aw, 0..) |a, t| {
                        hr += @as(i64, a) * ws.hf[t * F + f].re;
                        hi += @as(i64, a) * ws.hf[t * F + f].im;
                    }
                    h = .{ .re = @intCast(fx.sat(fx.rshr(hr, 14), 16)), .im = @intCast(fx.sat(fx.rshr(hi, 14), 16)) };
                } else {
                    var hr: f64 = 0;
                    var hi: f64 = 0;
                    for (aw, 0..) |a, t| {
                        hr += a * ws.hf[t * F + f].re;
                        hi += a * ws.hf[t * F + f].im;
                    }
                    h = .{ .re = @floatCast(hr), .im = @floatCast(hi) };
                }
                if (probe.h) |ph| {
                    ph[(frame * m.num_data + d) * 2] = toF(h.re);
                    ph[(frame * m.num_data + d) * 2 + 1] = toF(h.im);
                }
                const lo = ws.llr_tx[d * B ..][0..B];
                if (fixed) {
                    const yr: i64 = y.re;
                    const yi: i64 = y.im;
                    const hr: i64 = h.re;
                    const hi: i64 = h.im;
                    const zv = [2]i64{ yr * hr + yi * hi, yi * hr - yr * hi };
                    const u = fx.rshr((hr * hr + hi * hi) * m.a_q, 16);
                    const mult = fx.rshr(mant * m.r[d], 15);
                    for (zv, 0..) |v, dim| {
                        var met: [1 << (max_bits / 2)]i64 = undefined;
                        for (0..nl) |l| met[l] = u * m.levels[l] * m.levels[l] - 2 * v * m.levels[l];
                        for (0..md) |j| {
                            const dd = minDiff(i64, met[0..nl], @intCast(md - 1 - j));
                            lo[2 * j + dim] = @intCast(fx.satSym(fx.rshr(dd * mult, shift), @intCast(m.dec_params.llr_bits)));
                        }
                    }
                } else {
                    const yr: f64 = y.re;
                    const yi: f64 = y.im;
                    const hr: f64 = h.re;
                    const hi: f64 = h.im;
                    const zv = [2]f64{ yr * hr + yi * hi, yi * hr - yr * hi };
                    const u = (hr * hr + hi * hi) * m.a_f;
                    const mult = scale * m.r[d];
                    for (zv, 0..) |v, dim| {
                        var met: [1 << (max_bits / 2)]f64 = undefined;
                        for (0..nl) |l| {
                            const lv: f64 = @floatFromInt(m.levels[l]);
                            met[l] = u * lv * lv - 2 * v * lv;
                        }
                        for (0..md) |j| {
                            const dd = minDiff(f64, met[0..nl], @intCast(md - 1 - j));
                            lo[2 * j + dim] = @floatCast(dd * mult);
                        }
                    }
                }
            }

            // 7. deinterleave and decode
            for (m.perm, 0..) |p, i| ws.llr[p] = ws.llr_tx[i];
            const n = m.code.n;
            const k = m.code.k;
            if (llr_out) |lo| {
                for (lo, ws.llr[0 .. m.num_cw * n]) |*d, v| d.* = toF(v);
            }
            for (0..m.num_cw) |c| {
                ws.iters += ws.dec.decode(&m.code, ws.llr[c * n ..][0..n], m.dec_params, out[c * k ..][0..k]);
                ws.codewords += 1;
            }
        }

        inline fn toF(v: anytype) f32 {
            return if (fixed) @floatFromInt(v) else v;
        }

        /// min over the labels whose bit `sh` is 0, minus min over those where it is 1
        inline fn minDiff(comptime N: type, met: []const N, sh: u5) N {
            const top: N = if (N == f64) std.math.inf(f64) else std.math.maxInt(i64);
            var mn = [2]N{ top, top };
            for (met, 0..) |v, l| {
                const b = (l >> sh) & 1;
                mn[b] = @min(mn[b], v);
            }
            return mn[0] - mn[1];
        }

        inline fn cmulRot(v: Cx, w: Cx) Cx {
            if (fixed) {
                const re = @as(i64, v.re) * w.re - @as(i64, v.im) * w.im;
                const im = @as(i64, v.re) * w.im + @as(i64, v.im) * w.re;
                return .{ .re = @intCast(fx.sat(fx.rshr(re, fx.tw_frac), 16)), .im = @intCast(fx.sat(fx.rshr(im, fx.tw_frac), 16)) };
            } else return v.mul(w);
        }

        /// One phase ramp per frame from the pilot symbols' common phases.
        fn track(m: *const Self, ws: *Workspace, probe: Probe, frame: usize) void {
            const F = m.f;
            const S = m.s;
            const np = m.npsym;
            const npi: i64 = np;
            var phi: [max_symbols]if (fixed) i64 else f64 = undefined;
            if (fixed) {
                var zr: [max_pilot_symbols]i64 = undefined;
                var zi: [max_pilot_symbols]i64 = undefined;
                for (0..np) |t| {
                    zr[t] = 0;
                    zi[t] = 0;
                }
                for (0..F) |f| {
                    var sr: i64 = 0;
                    var si: i64 = 0;
                    for (0..np) |t| {
                        sr += ws.hf[t * F + f].re;
                        si += ws.hf[t * F + f].im;
                    }
                    for (0..np) |t| {
                        const hr: i64 = ws.hf[t * F + f].re;
                        const hi: i64 = ws.hf[t * F + f].im;
                        zr[t] += hr * sr + hi * si;
                        zi[t] += hi * sr - hr * si;
                    }
                }
                var mx: u64 = 0;
                for (0..np) |t| mx = @max(mx, @max(@abs(zr[t]), @abs(zi[t])));
                const bl: u32 = 64 - @clz(mx);
                const sh: u6 = if (bl > 30) @intCast(bl - 30) else 0;
                var num: i64 = 0;
                var den: i64 = 0;
                for (0..np) |t| {
                    const v = fx.vector(zr[t] >> sh, zi[t] >> sh);
                    const w = fx.rshr(v.mag, 16);
                    const sv = npi * m.psym[t] - m.sum_psym;
                    num += w * sv * v.ang;
                    den += w * sv * sv;
                }
                for (0..S) |s| {
                    const sv = npi * @as(i64, @intCast(s)) - m.sum_psym;
                    phi[s] = if (den > 0) fx.divRound(num * sv * m.track_gain, den << 15) else 0;
                }
                for (0..S) |s| {
                    const cs = fx.cossin(-phi[s]);
                    for (ws.y[s * F ..][0..F]) |*v| v.* = fx.rot16(v.*, cs);
                }
                for (0..np) |t| {
                    const cs = fx.cossin(-phi[m.psym[t]]);
                    for (ws.hf[t * F ..][0..F]) |*v| v.* = fx.rot16(v.*, cs);
                }
            } else {
                var zr: [max_pilot_symbols]f64 = @splat(0);
                var zi: [max_pilot_symbols]f64 = @splat(0);
                for (0..F) |f| {
                    var sr: f64 = 0;
                    var si: f64 = 0;
                    for (0..np) |t| {
                        sr += ws.hf[t * F + f].re;
                        si += ws.hf[t * F + f].im;
                    }
                    for (0..np) |t| {
                        const hr: f64 = ws.hf[t * F + f].re;
                        const hi: f64 = ws.hf[t * F + f].im;
                        zr[t] += hr * sr + hi * si;
                        zi[t] += hi * sr - hr * si;
                    }
                }
                var num: f64 = 0;
                var den: f64 = 0;
                for (0..np) |t| {
                    const w = @sqrt(zr[t] * zr[t] + zi[t] * zi[t]);
                    const sv: f64 = @floatFromInt(npi * m.psym[t] - m.sum_psym);
                    num += w * sv * std.math.atan2(zi[t], zr[t]);
                    den += w * sv * sv;
                }
                for (0..S) |s| {
                    const sv: f64 = @floatFromInt(npi * @as(i64, @intCast(s)) - m.sum_psym);
                    phi[s] = if (den > 0) num * sv / den * @as(f64, @floatFromInt(m.track_gain)) / 32768.0 else 0;
                }
                for (0..S) |s| {
                    const cs = C.expj(@floatCast(-phi[s]));
                    for (ws.y[s * F ..][0..F]) |*v| v.* = v.mul(cs);
                }
                for (0..np) |t| {
                    const cs = C.expj(@floatCast(-phi[m.psym[t]]));
                    for (ws.hf[t * F ..][0..F]) |*v| v.* = v.mul(cs);
                }
            }
            if (probe.phi) |ph| {
                for (0..S) |s| ph[frame * S + s] = if (fixed) @floatFromInt(phi[s]) else @floatCast(phi[s]);
            }
        }

        // ---- batches -----------------------------------------------------------

        pub fn tx(m: *Self, info: [*]const u8, batch: u32, out: [*]Sample) void {
            const Ctx = struct {
                m: *Self,
                info: [*]const u8,
                out: [*]Sample,
                fn run(c: @This(), w: *Workspace, i: usize) void {
                    const ib = c.m.num_cw * c.m.code.k;
                    const ns = c.m.numSamples();
                    c.m.txFrame(w, c.info[i * ib ..][0..ib], c.out[i * ns ..][0..ns]);
                }
            };
            par.run(Workspace, m.ws, batch, Ctx{ .m = m, .info = info, .out = out }, Ctx.run);
        }

        /// llr_scale [batch]: a / (N0 LSB), N0 the noise variance per RE in
        /// squared input units, LSB the LLR step in nats.
        pub fn rx(m: *Self, y: [*]const Sample, len_in: u32, l_min: i32, llr_scale: [*]const f64, batch: u32, out: [*]u8, llr_out: ?[*]f32, probe: Probe) !void {
            if (!m.have_filter) return error.NoFilter;
            if (len_in < m.numSamples()) return error.BadLength;
            if (l_min != m.rot_l_min) m.setRot(l_min);
            const Ctx = struct {
                m: *Self,
                y: [*]const Sample,
                len_in: u32,
                scale: [*]const f64,
                out: [*]u8,
                llr_out: ?[*]f32,
                probe: Probe,
                fn run(c: @This(), w: *Workspace, i: usize) void {
                    const ob = c.m.num_cw * c.m.code.k;
                    const nl = c.m.num_cw * c.m.code.n;
                    const lo = if (c.llr_out) |p| p[i * nl ..][0..nl] else null;
                    c.m.rxFrame(w, c.y[i * c.len_in ..][0..c.len_in], c.scale[i], c.out[i * ob ..][0..ob], lo, c.probe, i);
                }
            };
            par.run(Workspace, m.ws, batch, Ctx{ .m = m, .y = y, .len_in = len_in, .scale = llr_scale, .out = out, .llr_out = llr_out, .probe = probe }, Ctx.run);
        }

        pub fn stats(m: *Self, out: *[2]u64, reset: bool) void {
            out.* = .{ 0, 0 };
            for (m.ws) |*w| {
                out[0] += w.iters;
                out[1] += w.codewords;
                if (reset) {
                    w.iters = 0;
                    w.codewords = 0;
                }
            }
        }
    };
}
