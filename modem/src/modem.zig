//! Frame-level modem, floating point (M3). TX: 5G LDPC per codeword ->
//! frame bit interleaver -> QAM -> resource grid with pilots -> IFFT + CP.
//! RX: CP removal + FFT -> optional linear phase tracker on the block
//! pilots -> LS on the pilots and a 2D linear filter -> one-tap LMMSE
//! equaliser -> exact LLRs -> deinterleave -> LDPC. It mirrors
//! harness.chain.OfdmLink with estimator="lmmse" block for block.
//!
//! The numerology, codeword plan, pilots and interleaver come in a Config
//! from harness/waveform.py; the estimator filter (W, error variance per
//! RE) is set per noise level, as GenieLmmseEstimator computes it. All
//! memory is allocated in create(); tx() and rx() split the batch over
//! worker threads, one workspace each.

const std = @import("std");
const ldpc = @import("ldpc.zig");
const Qam = @import("qam.zig").Qam;
const ofdm = @import("ofdm.zig");
const C = ofdm.C;

pub const max_threads = 64;

pub const Config = extern struct {
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
    num_iter: u32,
    early_stop: u32,
    phase_track: u32, // 0 off, 1 linear (one phase ramp per frame)
    num_threads: u32, // 0: one per CPU
    pilot_mask: [*]const u8, // [num_symbols * F], 1 = pilot, symbol-major
    pilots: [*]const C, // in mask order
    perm: [*]const u32, // frame bit interleaver, tx bit i = coded bit perm[i]
};

const Workspace = struct {
    sym: []C, // one OFDM symbol, fft_size
    y: []C, // received effective subcarriers [S * F]
    pre: []f32, // [2 * Np] planar LS estimate, or [2 * Npb] for the tracker
    h: []C, // estimate at the data REs [num_data]
    llr_tx: []f32, // [frame_bits] in transmission order
    llr: []f32, // [frame_bits] deinterleaved
    bits: []u8, // [frame_bits]
    enc: []u8, // [n_ldpc]
    dec: ldpc.Decoder,
    iters: u64,
    codewords: u64,
};

pub const Modem = struct {
    alloc: std.mem.Allocator,
    s: u32,
    nfft: u32,
    cp: u32,
    f: u32,
    frame_bits: u32,
    num_cw: u32,
    num_data: u32,
    num_pilots: u32,
    num_block: u32,
    num_iter: u32,
    early_stop: bool,
    phase_track: bool,
    anchor: f32,
    code: ldpc.Code,
    qam: Qam,
    fft: ofdm.Fft,
    bin: []u32, // FFT bin of effective subcarrier f
    is_pilot: []u8, // [S * F]
    data_re: []u32, // [num_data] grid index s * F + f, symbol-major
    pilot_re: []u32, // [Np]
    pilot: []C, // [Np]
    inv_pilot: []C, // [Np]
    block_pilot: []u32, // [Npb] indices into the pilot list: pilots on full pilot symbols
    perm: []u32,
    // estimator, set per noise level
    w: []f32, // [num_data][2][Np] planar rows (re block, im block) at the data REs
    err_var: []f32, // [num_data]
    wb: []f32, // [Np][2][Npb] tracker filter rows at the pilot REs
    have_filter: bool,
    rot: []C, // demodulator phase compensation for l_min
    rot_l_min: i32,
    threads: u32,
    ws: []Workspace,

    pub fn create(alloc: std.mem.Allocator, cfg: *const Config) !*Modem {
        const m = try alloc.create(Modem);
        errdefer alloc.destroy(m);
        m.alloc = alloc;
        if (cfg.num_symbols == 0 or cfg.num_symbols > 64) return error.BadConfig;
        m.s = cfg.num_symbols;
        m.nfft = cfg.fft_size;
        m.cp = cfg.cp_len;
        if (cfg.guard_lo + cfg.guard_hi + @intFromBool(cfg.dc_null != 0) >= cfg.fft_size) return error.BadConfig;
        m.f = cfg.fft_size - cfg.guard_lo - cfg.guard_hi - @intFromBool(cfg.dc_null != 0);
        m.num_cw = cfg.num_codewords;
        m.num_iter = cfg.num_iter;
        m.early_stop = cfg.early_stop != 0;
        m.phase_track = cfg.phase_track != 0;
        m.code = try ldpc.Code.init(cfg.k, cfg.n);
        m.qam = try Qam.init(cfg.bits_per_symbol);
        m.fft = try ofdm.Fft.init(alloc, cfg.fft_size);

        // effective subcarriers in Sionna's centred order, guards and DC
        // removed; centred index i sits in bin (i + N/2) mod N
        m.bin = try alloc.alloc(u32, m.f);
        var f: u32 = 0;
        for (cfg.guard_lo..cfg.fft_size - cfg.guard_hi) |i| {
            if (cfg.dc_null != 0 and i == cfg.fft_size / 2) continue;
            m.bin[f] = @intCast((i + cfg.fft_size / 2) % cfg.fft_size);
            f += 1;
        }

        const sf = m.s * m.f;
        m.is_pilot = try alloc.dupe(u8, cfg.pilot_mask[0..sf]);
        var np: u32 = 0;
        for (m.is_pilot) |p| np += @intFromBool(p != 0);
        m.num_pilots = np;
        m.num_data = sf - np;
        m.frame_bits = m.num_data * cfg.bits_per_symbol;
        if (m.num_cw * cfg.n > m.frame_bits) return error.BadConfig;
        m.data_re = try alloc.alloc(u32, m.num_data);
        m.pilot_re = try alloc.alloc(u32, np);
        var nd: u32 = 0;
        var ip: u32 = 0;
        for (m.is_pilot, 0..) |p, i| {
            if (p != 0) {
                m.pilot_re[ip] = @intCast(i);
                ip += 1;
            } else {
                m.data_re[nd] = @intCast(i);
                nd += 1;
            }
        }
        m.pilot = try alloc.dupe(C, cfg.pilots[0..np]);
        m.inv_pilot = try alloc.alloc(C, np);
        for (m.pilot, m.inv_pilot) |p, *q| {
            // 1 / p in f64 then rounded, as GenieLmmseEstimator.inv_pilots
            const re: f64 = p.re;
            const im: f64 = p.im;
            const d = re * re + im * im;
            q.* = .{ .re = @floatCast(re / d), .im = @floatCast(-im / d) };
        }
        // block pilots: those on symbols that are pilots on every subcarrier
        var nb: u32 = 0;
        var anchor_sum: f32 = 0;
        var anchor_n: f32 = 0;
        for (0..m.s) |s| {
            if (std.mem.allEqual(u8, m.is_pilot[s * m.f ..][0..m.f], 1)) {
                nb += m.f;
                anchor_sum += @floatFromInt(s);
                anchor_n += 1;
            }
        }
        if (m.phase_track and nb == 0) return error.BadConfig;
        m.num_block = nb;
        m.anchor = if (anchor_n > 0) anchor_sum / anchor_n else 0;
        m.block_pilot = try alloc.alloc(u32, nb);
        var jb: u32 = 0;
        for (m.pilot_re, 0..) |re, p| {
            const s = re / m.f;
            if (std.mem.allEqual(u8, m.is_pilot[s * m.f ..][0..m.f], 1)) {
                m.block_pilot[jb] = @intCast(p);
                jb += 1;
            }
        }
        m.perm = try alloc.dupe(u32, cfg.perm[0..m.frame_bits]);
        for (m.perm) |p| if (p >= m.frame_bits) return error.BadConfig;

        m.w = try alloc.alloc(f32, @as(usize, m.num_data) * 2 * np);
        m.err_var = try alloc.alloc(f32, m.num_data);
        m.wb = try alloc.alloc(f32, if (m.phase_track) @as(usize, np) * 2 * nb else 0);
        m.have_filter = false;
        m.rot = try alloc.alloc(C, m.nfft);
        m.setRot(0);

        const cpus: u32 = @intCast(std.Thread.getCpuCount() catch 1);
        m.threads = @min(max_threads, if (cfg.num_threads == 0) cpus else cfg.num_threads);
        m.ws = try alloc.alloc(Workspace, m.threads);
        for (m.ws) |*w| {
            w.* = .{
                .sym = try alloc.alloc(C, m.nfft),
                .y = try alloc.alloc(C, sf),
                .pre = try alloc.alloc(f32, 2 * np),
                .h = try alloc.alloc(C, m.num_data),
                .llr_tx = try alloc.alloc(f32, m.frame_bits),
                .llr = try alloc.alloc(f32, m.frame_bits),
                .bits = try alloc.alloc(u8, m.frame_bits),
                .enc = try alloc.alloc(u8, m.code.n_ldpc),
                .dec = try ldpc.Decoder.init(alloc),
                .iters = 0,
                .codewords = 0,
            };
        }
        return m;
    }

    pub fn destroy(m: *Modem) void {
        const a = m.alloc;
        for (m.ws) |*w| {
            a.free(w.sym);
            a.free(w.y);
            a.free(w.pre);
            a.free(w.h);
            a.free(w.llr_tx);
            a.free(w.llr);
            a.free(w.bits);
            a.free(w.enc);
            w.dec.deinit(a);
        }
        a.free(m.ws);
        m.fft.deinit(a);
        inline for (.{ "bin", "is_pilot", "data_re", "pilot_re", "pilot", "inv_pilot", "block_pilot", "perm", "w", "err_var", "wb", "rot" }) |name| {
            a.free(@field(m, name));
        }
        a.destroy(m);
    }

    pub fn numSamples(m: *const Modem) u32 {
        return m.s * (m.nfft + m.cp);
    }

    /// w: [S * F][Np] complex, h_hat = w h_ls; err_var: [S * F];
    /// wb: [S * F][Npb] complex for the tracker's block-pilot estimate
    /// (required when phase tracking is on). Only the rows the receiver
    /// uses are kept.
    pub fn setFilter(m: *Modem, w: [*]const C, err_var: [*]const f32, wb: ?[*]const C) !void {
        const np = m.num_pilots;
        for (m.data_re, 0..) |re, d| {
            const row = w[@as(usize, re) * np ..][0..np];
            const dst = m.w[d * 2 * np ..][0 .. 2 * np];
            for (row, 0..) |v, p| {
                dst[p] = v.re;
                dst[np + p] = v.im;
            }
            m.err_var[d] = err_var[re];
        }
        if (m.phase_track) {
            const b = wb orelse return error.BadConfig;
            const nb = m.num_block;
            for (m.pilot_re, 0..) |re, p| {
                const row = b[@as(usize, re) * nb ..][0..nb];
                const dst = m.wb[p * 2 * nb ..][0 .. 2 * nb];
                for (row, 0..) |v, j| {
                    dst[j] = v.re;
                    dst[nb + j] = v.im;
                }
            }
        }
        m.have_filter = true;
    }

    fn setRot(m: *Modem, l_min: i32) void {
        // OFDMDemodulator: exp(-2 pi i l_min k / N), computed in f32
        const c: f32 = -2.0 * std.math.pi * @as(f32, @floatFromInt(l_min)) / @as(f32, @floatFromInt(m.nfft));
        for (m.rot, 0..) |*r, k| r.* = C.expj(c * @as(f32, @floatFromInt(k)));
        m.rot_l_min = l_min;
    }

    // ---- TX ----------------------------------------------------------------

    fn txFrame(m: *const Modem, ws: *Workspace, info: []const u8, out: []C) void {
        const n = m.code.n;
        const k = m.code.k;
        @memset(ws.bits, 0); // padding bits after the last codeword
        for (0..m.num_cw) |c| m.code.encode(info[c * k ..][0..k], ws.bits[c * n ..][0..n], ws.enc);
        const B = m.qam.bits;
        var tx_bits: [8]u8 = undefined;
        var d: usize = 0;
        var p: usize = 0;
        const symlen = m.nfft + m.cp;
        for (0..m.s) |s| {
            @memset(ws.sym, C.zero);
            for (0..m.f) |f| {
                const v = if (m.is_pilot[s * m.f + f] != 0) blk: {
                    p += 1;
                    break :blk m.pilot[p - 1];
                } else blk: {
                    for (0..B) |j| tx_bits[j] = ws.bits[m.perm[d * B + j]];
                    d += 1;
                    break :blk m.qam.map(tx_bits[0..B]);
                };
                ws.sym[m.bin[f]] = v;
            }
            m.fft.run(ws.sym, true);
            const o = out[s * symlen ..][0..symlen];
            @memcpy(o[0..m.cp], ws.sym[m.nfft - m.cp ..]);
            @memcpy(o[m.cp..], ws.sym);
        }
    }

    // ---- RX ----------------------------------------------------------------

    fn rxFrame(m: *const Modem, ws: *Workspace, y_in: []const C, no: f32, out: []u8, llr_out: ?[]f32) void {
        const symlen = m.nfft + m.cp;
        const F = m.f;
        for (0..m.s) |s| {
            @memcpy(ws.sym, y_in[s * symlen + m.cp ..][0..m.nfft]);
            m.fft.run(ws.sym, false);
            for (0..F) |f| {
                const b = m.bin[f];
                ws.y[s * F + f] = ws.sym[b].mul(m.rot[b]);
            }
        }
        if (m.phase_track) m.derotate(ws);

        // LS on every pilot, then h = W h_ls at the data REs
        const np = m.num_pilots;
        const pre = ws.pre[0 .. 2 * np];
        for (m.pilot_re, m.inv_pilot, 0..) |re, ip, p| {
            const h = ws.y[re].mul(ip);
            pre[p] = h.re;
            pre[np + p] = h.im;
        }
        for (0..m.num_data) |d| {
            ws.h[d] = cdot(m.w[d * 2 * np ..][0 .. 2 * np], pre, np);
        }

        // equalise and demap: x = y / h, no_eff = (no + err_var) / |h|^2
        const B = m.qam.bits;
        for (m.data_re, 0..) |re, d| {
            const h = ws.h[d];
            const g = h.abs2();
            const x = ws.y[re].mul(h.conj()).scale(1.0 / g);
            m.qam.demap(x, (no + m.err_var[d]) / g, ws.llr_tx[d * B ..][0..B]);
        }
        for (m.perm, 0..) |p, i| ws.llr[p] = ws.llr_tx[i];
        const n = m.code.n;
        const k = m.code.k;
        if (llr_out) |lo| @memcpy(lo, ws.llr[0 .. m.num_cw * n]);
        for (0..m.num_cw) |c| {
            ws.iters += ws.dec.decode(&m.code, ws.llr[c * n ..][0..n], m.num_iter, m.early_stop, out[c * k ..][0..k]);
            ws.codewords += 1;
        }
    }

    /// OfdmLink.derotate with phase_track="linear": estimate the channel
    /// from the block pilots, measure each symbol's phase on its pilots
    /// (z), fit one phase ramp b (s - anchor) and rotate it out.
    fn derotate(m: *const Modem, ws: *Workspace) void {
        const nb = m.num_block;
        const pre = ws.pre[0 .. 2 * nb];
        for (m.block_pilot, 0..) |p, j| {
            const h = ws.y[m.pilot_re[p]].mul(m.inv_pilot[p]);
            pre[j] = h.re;
            pre[nb + j] = h.im;
        }
        var z: [64]C = @splat(C.zero);
        const S = m.s;
        for (m.pilot_re, m.pilot, 0..) |re, x, p| {
            const h0 = cdot(m.wb[p * 2 * nb ..][0 .. 2 * nb], pre, nb);
            const s = re / m.f;
            z[s] = z[s].add(ws.y[re].mul(h0.mul(x).conj()));
        }
        var acc = C.zero;
        for (1..S) |s| acc = acc.add(z[s].mul(z[s - 1].conj()));
        const b0 = std.math.atan2(acc.im, acc.re);
        var num: f32 = 0;
        var den: f32 = 0;
        for (0..S) |s| {
            const sv = @as(f32, @floatFromInt(s)) - m.anchor;
            const r = z[s].mul(C.expj(-b0 * sv));
            const resid = std.math.atan2(r.im, r.re);
            const w = @sqrt(z[s].abs2());
            num += w * sv * resid;
            den += w * sv * sv;
        }
        const slope = b0 + num / den;
        for (0..S) |s| {
            const rot = C.expj(-slope * (@as(f32, @floatFromInt(s)) - m.anchor));
            for (ws.y[s * m.f ..][0..m.f]) |*v| v.* = v.mul(rot);
        }
    }

    // ---- batches over threads ----------------------------------------------

    pub fn tx(m: *Modem, info: [*]const u8, batch: u32, out: [*]C) void {
        const Ctx = struct {
            m: *Modem,
            info: [*]const u8,
            out: [*]C,
            fn run(ctx: @This(), ws: *Workspace, i: usize) void {
                const ib = ctx.m.num_cw * ctx.m.code.k;
                const ns = ctx.m.numSamples();
                ctx.m.txFrame(ws, ctx.info[i * ib ..][0..ib], ctx.out[i * ns ..][0..ns]);
            }
        };
        m.parallel(batch, Ctx{ .m = m, .info = info, .out = out }, Ctx.run);
    }

    pub fn rx(m: *Modem, y: [*]const C, len_in: u32, l_min: i32, no: f32, batch: u32, out: [*]u8, llr_out: ?[*]f32) !void {
        if (!m.have_filter) return error.NoFilter;
        if (len_in < m.numSamples()) return error.BadLength;
        if (l_min != m.rot_l_min) m.setRot(l_min);
        const Ctx = struct {
            m: *Modem,
            y: [*]const C,
            len_in: u32,
            no: f32,
            out: [*]u8,
            llr_out: ?[*]f32,
            fn run(ctx: @This(), ws: *Workspace, i: usize) void {
                const ob = ctx.m.num_cw * ctx.m.code.k;
                const nl = ctx.m.num_cw * ctx.m.code.n;
                const lo = if (ctx.llr_out) |p| p[i * nl ..][0..nl] else null;
                ctx.m.rxFrame(ws, ctx.y[i * ctx.len_in ..][0..ctx.len_in], ctx.no, ctx.out[i * ob ..][0..ob], lo);
            }
        };
        m.parallel(batch, Ctx{ .m = m, .y = y, .len_in = len_in, .no = no, .out = out, .llr_out = llr_out }, Ctx.run);
    }

    fn parallel(m: *Modem, batch: u32, ctx: anytype, comptime func: fn (@TypeOf(ctx), *Workspace, usize) void) void {
        const T = @min(m.threads, batch);
        var next = std.atomic.Value(u32).init(0);
        const Worker = struct {
            fn work(c: @TypeOf(ctx), ws: *Workspace, counter: *std.atomic.Value(u32), total: u32) void {
                while (true) {
                    const i = counter.fetchAdd(1, .monotonic);
                    if (i >= total) return;
                    func(c, ws, i);
                }
            }
        };
        var handles: [max_threads]?std.Thread = @splat(null);
        for (1..T) |t| {
            // a thread that fails to start leaves its frames to the others
            handles[t] = std.Thread.spawn(.{}, Worker.work, .{ ctx, &m.ws[t], &next, batch }) catch null;
        }
        Worker.work(ctx, &m.ws[0], &next, batch);
        for (handles[1..T]) |h| if (h) |th| th.join();
    }

    pub fn stats(m: *Modem, out: *[2]u64, reset: bool) void {
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

/// Complex dot product of a planar row [re | im] (length 2n) with a planar
/// vector [re | im].
fn cdot(row: []const f32, x: []const f32, n: usize) C {
    const L = 8;
    const V = @Vector(L, f32);
    var rr: V = @splat(0);
    var ri: V = @splat(0);
    var ir: V = @splat(0);
    var ii: V = @splat(0);
    var i: usize = 0;
    while (i + L <= n) : (i += L) {
        const wr: V = row[i..][0..L].*;
        const wi: V = row[n + i ..][0..L].*;
        const xr: V = x[i..][0..L].*;
        const xi: V = x[n + i ..][0..L].*;
        rr += wr * xr;
        ii += wi * xi;
        ri += wr * xi;
        ir += wi * xr;
    }
    var re = @reduce(.Add, rr) - @reduce(.Add, ii);
    var im = @reduce(.Add, ri) + @reduce(.Add, ir);
    while (i < n) : (i += 1) {
        re += row[i] * x[i] - row[n + i] * x[n + i];
        im += row[i] * x[n + i] + row[n + i] * x[i];
    }
    return .{ .re = re, .im = im };
}
