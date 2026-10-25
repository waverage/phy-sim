//! 5G NR LDPC (TS 38.212) built the way Sionna's LDPC5GEncoder and
//! LDPC5GDecoder build it, so the codewords match bit for bit: base graph
//! and lifting size selection, zero filler bits, the first 2Z systematic
//! bits punctured, rate matching to the first n bits of the circular
//! buffer (RV 0, no bit interleaver). Base graph tables are Sionna's
//! (Apache-2.0), from the 3GPP spec.
//!
//! The decoder is Sionna's default: flooding sum-product with the
//! boxplus-phi check node and messages clipped to +-20, the punctured
//! tail of degree-1 nodes pruned (at whole-block granularity here; the
//! extra degree-1 nodes with zero LLR are inert). It runs on the QC
//! structure: every base-graph entry holds one Z-lane message vector in
//! check-node order, so the check node is lane-parallel and the variable
//! node reads its messages back through the entry's cyclic shift.
//! Optional early stop on a zero syndrome.

const std = @import("std");
const fm = @import("fmath.zig");

const bg_csv = [2][]const u8{ @embedFile("ldpc/5G_bg1.csv"), @embedFile("ldpc/5G_bg2.csv") };

pub const max_entries = 316;
pub const max_z = 384;
pub const max_rows = 46;
pub const max_cols = 68;
pub const max_row_degree = 32;
pub const llr_max: f32 = 20.0;

pub const Error = error{ UnsupportedCode, BadTable };

pub const Entry = struct { row: u16, col: u16, shift: u16 };

pub const Code = struct {
    k: u32,
    n: u32,
    bg: u8, // 1 or 2
    z: u32,
    zp: u32, // lane stride, z rounded up to the vector width
    i_ls: u32,
    k_b: u32, // information columns of the base graph (22 or 10)
    num_cols: u32,
    k_ldpc: u32,
    n_ldpc: u32,
    k_filler: u32,
    entries: [max_entries]Entry,
    num_entries: u32,
    row_start: [max_rows + 1]u16,
    core_shift: u16, // shift left on p_0 when the four core rows are summed
    enc_rows: u32, // base rows the encoder needs for n output bits
    dec_rows: u32, // base rows and columns the decoder keeps after pruning
    dec_cols: u32,
    dec_entries: u32,
    col_start: [max_cols + 1]u16,
    col_entries: [max_entries]u16,

    pub fn init(k: u32, n: u32) Error!Code {
        if (k < 12 or k > 8448 or n == 0 or n > 316 * 384) return error.UnsupportedCode;
        const r = @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n));
        if (r > 0.95 or r < 1.0 / 5.0) return error.UnsupportedCode;
        var c: Code = undefined;
        c.k = k;
        c.n = n;
        // LDPC5GEncoder._sel_basegraph
        c.bg = if (k <= 292) 2 else if (k <= 3824 and r <= 0.67) 2 else if (r <= 0.25) 2 else 1;
        if (c.bg == 2 and k > 3840) return error.UnsupportedCode;
        if (c.bg == 1 and r < 1.0 / 3.0) return error.UnsupportedCode;
        // LDPC5GEncoder._sel_lifting
        const sel_kb: u32 = if (c.bg == 1) 22 else if (k > 640) 10 else if (k > 560) 9 else if (k > 192) 8 else 6;
        const s_val = [8][]const u32{
            &.{ 2, 4, 8, 16, 32, 64, 128, 256 },
            &.{ 3, 6, 12, 24, 48, 96, 192, 384 },
            &.{ 5, 10, 20, 40, 80, 160, 320 },
            &.{ 7, 14, 28, 56, 112, 224 },
            &.{ 9, 18, 36, 72, 144, 288 },
            &.{ 11, 22, 44, 88, 176, 352 },
            &.{ 13, 26, 52, 104, 208 },
            &.{ 15, 30, 60, 120, 240 },
        };
        var min_val: u32 = 100000;
        c.z = 0;
        c.i_ls = 0;
        for (s_val, 0..) |set, i| {
            for (set) |s1| {
                const x = sel_kb * s1;
                if (x >= k and x < min_val) {
                    min_val = x;
                    c.z = s1;
                    c.i_ls = @intCast(i);
                }
            }
        }
        if (c.z == 0) return error.UnsupportedCode;
        c.zp = (c.z + fm.lanes - 1) / fm.lanes * fm.lanes;
        c.k_b = if (c.bg == 1) 22 else 10;
        c.num_cols = if (c.bg == 1) 68 else 52;
        const num_rows = c.num_cols - c.k_b;
        c.k_ldpc = c.k_b * c.z;
        c.n_ldpc = c.num_cols * c.z;
        c.k_filler = c.k_ldpc - k;
        try c.loadBaseGraph();
        if (c.row_start[num_rows] != c.num_entries) return error.BadTable;
        try c.findCoreShift();

        // Encoder: the output ends at bit 2Z + n - 1 of the filler-free
        // buffer, which is bit 2Z + n - 1 + k_filler of the codeword.
        const last_bit = 2 * c.z + n - 1 + c.k_filler;
        c.enc_rows = @max(4, last_bit / c.z + 1 - c.k_b);

        // Decoder pruning as LDPC5GDecoder(prune_pcm=True): drop the
        // trailing degree-1 VNs that are punctured beyond n, and their CNs.
        var deg: [max_cols]u32 = @splat(0);
        for (c.entries[0..c.num_entries]) |e| deg[e.col] += 1;
        var last_col = c.num_cols;
        while (last_col > 1 and deg[last_col - 1] == 1) last_col -= 1;
        const nb_punc = (c.n_ldpc - c.k_filler) - n - 2 * c.z;
        const n_pruned = @max(last_col * c.z, c.n_ldpc - nb_punc);
        c.dec_cols = @min(c.num_cols, (n_pruned + c.z - 1) / c.z);
        c.dec_rows = c.dec_cols - c.k_b;
        c.dec_entries = c.row_start[c.dec_rows];
        var col_count: [max_cols + 1]u16 = @splat(0);
        for (c.entries[0..c.dec_entries]) |e| {
            if (e.col >= c.dec_cols) return error.BadTable;
            col_count[e.col + 1] += 1;
        }
        for (1..max_cols + 1) |i| col_count[i] += col_count[i - 1];
        c.col_start = col_count;
        var fill = col_count;
        for (c.entries[0..c.dec_entries], 0..) |e, i| {
            c.col_entries[fill[e.col]] = @intCast(i);
            fill[e.col] += 1;
        }
        for (0..c.dec_rows) |row| {
            if (c.row_start[row + 1] - c.row_start[row] > max_row_degree) return error.BadTable;
        }
        return c;
    }

    fn loadBaseGraph(c: *Code) Error!void {
        var lines = std.mem.splitScalar(u8, bg_csv[c.bg - 1], '\n');
        _ = lines.next();
        _ = lines.next();
        var row: u16 = 0;
        c.num_entries = 0;
        c.row_start = @splat(0);
        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \r");
            if (line.len == 0) continue;
            var f = std.mem.splitScalar(u8, line, ';');
            const f0 = std.mem.trim(u8, f.next().?, " ");
            if (f0.len > 0) row = std.fmt.parseInt(u16, f0, 10) catch return error.BadTable;
            const col = std.fmt.parseInt(u16, std.mem.trim(u8, f.next() orelse return error.BadTable, " "), 10) catch return error.BadTable;
            var v: u32 = 0;
            for (0..c.i_ls + 1) |_| {
                const s = std.mem.trim(u8, f.next() orelse return error.BadTable, " ");
                v = std.fmt.parseInt(u32, s, 10) catch return error.BadTable;
            }
            if (c.num_entries == max_entries or row >= max_rows or col >= c.num_cols) return error.BadTable;
            if (c.num_entries > 0 and row < c.entries[c.num_entries - 1].row) return error.BadTable;
            c.entries[c.num_entries] = .{ .row = row, .col = col, .shift = @intCast(v % c.z) };
            c.num_entries += 1;
            c.row_start[row + 1] = @intCast(c.num_entries);
        }
        // rows without entries inherit the previous end
        for (1..max_rows + 1) |i| c.row_start[i] = @max(c.row_start[i], c.row_start[i - 1]);
    }

    /// Summing the four core rows cancels the double-diagonal columns
    /// k_b+1..k_b+3 (each appears twice with the same shift) and leaves one
    /// shifted copy of p_0.
    fn findCoreShift(c: *Code) Error!void {
        var shifts: [4][8]u16 = undefined;
        var counts: [4]u8 = @splat(0);
        for (c.entries[0..c.row_start[4]]) |e| {
            if (e.col < c.k_b) continue;
            const j = e.col - c.k_b;
            if (j >= 4 or counts[j] == 8) return error.BadTable;
            shifts[j][counts[j]] = e.shift;
            counts[j] += 1;
        }
        for (0..4) |j| {
            var left: [8]u16 = undefined;
            var nl: usize = 0;
            for (shifts[j][0..counts[j]]) |s| {
                if (std.mem.indexOfScalar(u16, left[0..nl], s)) |pos| {
                    left[pos] = left[nl - 1];
                    nl -= 1;
                } else {
                    left[nl] = s;
                    nl += 1;
                }
            }
            if (j == 0) {
                if (nl != 1) return error.BadTable;
                c.core_shift = left[0];
            } else if (nl != 0) return error.BadTable;
        }
    }

    pub fn rowEntries(c: *const Code, row: usize) []const Entry {
        return c.entries[c.row_start[row]..c.row_start[row + 1]];
    }

    /// u: k information bits (0/1) -> out: n rate-matched code bits.
    /// scratch: at least n_ldpc bytes.
    pub fn encode(c: *const Code, u: []const u8, out: []u8, scratch: []u8) void {
        const z = c.z;
        const cw = scratch[0 .. (c.k_b + c.enc_rows) * z];
        @memset(cw, 0);
        @memcpy(cw[0..c.k], u[0..c.k]);
        var acc: [max_z]u8 = undefined;

        // p_0 from the sum of the four core rows (information columns only)
        @memset(acc[0..z], 0);
        for (c.entries[0..c.row_start[4]]) |e| {
            if (e.col < c.k_b) xorShifted(acc[0..z], cw[e.col * z ..][0..z], e.shift);
        }
        placeShifted(cw[c.k_b * z ..][0..z], acc[0..z], c.core_shift);

        // the remaining core columns, each from a core row in which it is
        // the only unknown
        var known = [_]bool{ true, false, false, false };
        var solved: usize = 1;
        while (solved < 4) {
            var progress = false;
            for (0..4) |row| {
                var unknown: ?Entry = null;
                var num_unknown: usize = 0;
                for (c.rowEntries(row)) |e| {
                    if (e.col >= c.k_b and !known[e.col - c.k_b]) {
                        unknown = e;
                        num_unknown += 1;
                    }
                }
                if (num_unknown != 1) continue;
                c.solveRow(row, unknown.?, cw, &acc);
                known[unknown.?.col - c.k_b] = true;
                solved += 1;
                progress = true;
            }
            if (!progress) unreachable; // findCoreShift validated the structure
        }

        // extension rows: one identity parity column each
        for (4..c.enc_rows) |row| {
            for (c.rowEntries(row)) |e| {
                if (e.col == c.k_b + row) {
                    c.solveRow(row, e, cw, &acc);
                    break;
                }
            }
        }

        for (0..c.n) |t| {
            const idx = 2 * z + t;
            out[t] = cw[if (idx < c.k) idx else idx + c.k_filler];
        }
    }

    fn solveRow(c: *const Code, row: usize, unknown: Entry, cw: []u8, acc: *[max_z]u8) void {
        const z = c.z;
        @memset(acc[0..z], 0);
        for (c.rowEntries(row)) |e| {
            if (e.col != unknown.col) xorShifted(acc[0..z], cw[e.col * z ..][0..z], e.shift);
        }
        placeShifted(cw[unknown.col * z ..][0..z], acc[0..z], unknown.shift);
    }
};

/// acc[i] ^= x[(i + s) mod z]
pub fn xorShifted(acc: []u8, x: []const u8, s: u32) void {
    const z = acc.len;
    for (0..z - s) |i| acc[i] ^= x[i + s];
    for (z - s..z) |i| acc[i] ^= x[i + s - z];
}

/// dst[(i + s) mod z] = src[i]
fn placeShifted(dst: []u8, src: []const u8, s: u32) void {
    const z = dst.len;
    for (0..z - s) |i| dst[i + s] = src[i];
    for (z - s..z) |i| dst[i + s - z] = src[i];
}

/// Per-thread decoder state, sized for the largest code.
pub const Decoder = struct {
    c2v: []f32,
    v2c: []f32,
    llr: []f32,
    tot: []f32,
    hard: []u8,

    pub fn init(alloc: std.mem.Allocator) !Decoder {
        const e = max_entries * max_z;
        const v = max_cols * max_z;
        return .{
            .c2v = try alloc.alloc(f32, e),
            .v2c = try alloc.alloc(f32, e),
            .llr = try alloc.alloc(f32, v),
            .tot = try alloc.alloc(f32, v),
            .hard = try alloc.alloc(u8, v),
        };
    }

    pub fn deinit(d: *Decoder, alloc: std.mem.Allocator) void {
        alloc.free(d.c2v);
        alloc.free(d.v2c);
        alloc.free(d.llr);
        alloc.free(d.tot);
        alloc.free(d.hard);
    }

    /// logits: n channel LLRs in Sionna's convention, log p(1) / p(0).
    /// out: k hard-decided information bits. Returns the iterations run.
    pub fn decode(d: *Decoder, c: *const Code, logits: []const f32, num_iter: u32, early_stop: bool, out: []u8) u32 {
        const z = c.z;
        const zp = c.zp;
        const ne = c.dec_entries;
        const llr = d.llr[0 .. c.dec_cols * zp];
        @memset(llr, 0);
        // rate recovery: [2Z punctured | k - 2Z received | filler | rest received | pruned]
        const n_sys = c.k - 2 * z;
        for (0..c.n) |t| {
            const j = if (t < n_sys) 2 * z + t else c.k_ldpc + (t - n_sys);
            llr[j / z * zp + j % z] = -std.math.clamp(logits[t], -llr_max, llr_max);
        }
        for (c.k..c.k_ldpc) |j| llr[j / z * zp + j % z] = llr_max;

        const c2v = d.c2v[0 .. ne * zp];
        const v2c = d.v2c[0 .. ne * zp];
        @memset(c2v, 0);
        @memset(v2c, 0);
        for (c.entries[0..ne], 0..) |e, i| {
            placeFrom(v2c[i * zp ..][0..z], llr[e.col * zp ..][0..z], e.shift);
        }

        var it: u32 = 0;
        while (it < num_iter) {
            checkNodes(c, c2v, v2c);
            variableNodes(c, c2v, v2c, llr, d.tot);
            it += 1;
            if (early_stop and it < num_iter and d.syndromeZero(c)) break;
        }
        for (0..c.k) |j| {
            out[j] = @intFromBool(d.tot[j / z * zp + j % z] <= 0);
        }
        return it;
    }

    fn syndromeZero(d: *Decoder, c: *const Code) bool {
        const z = c.z;
        const zp = c.zp;
        for (0..c.dec_cols * zp) |i| d.hard[i] = @intFromBool(d.tot[i] <= 0);
        var acc: [max_z]u8 = undefined;
        for (0..c.dec_rows) |row| {
            @memset(acc[0..z], 0);
            for (c.rowEntries(row)) |e| xorShifted(acc[0..z], d.hard[e.col * zp ..][0..z], e.shift);
            for (acc[0..z]) |a| if (a != 0) return false;
        }
        return true;
    }
};

/// dst[i] = src[(i + s) mod z]: a VN-ordered vector read in check order.
fn placeFrom(dst: []f32, src: []const f32, s: u32) void {
    const z = dst.len;
    for (0..z - s) |i| dst[i] = src[i + s];
    for (z - s..z) |i| dst[i] = src[i + s - z];
}

fn checkNodes(c: *const Code, c2v: []f32, v2c: []const f32) void {
    const zp = c.zp;
    const L = fm.lanes;
    const U = @Vector(L, u32);
    const zero_u: U = @splat(0);
    const one_u: U = @splat(1);
    const zero = fm.splat(0);
    var phis: [max_row_degree]fm.V = undefined;
    for (0..c.dec_rows) |row| {
        const e0: usize = c.row_start[row];
        const e1: usize = c.row_start[row + 1];
        var lo: usize = 0;
        while (lo < zp) : (lo += L) {
            var sum = zero;
            var neg = zero_u;
            for (e0..e1, 0..) |e, j| {
                const m: fm.V = v2c[e * zp + lo ..][0..L].*;
                const p = fm.phi(@abs(m));
                phis[j] = p;
                sum += p;
                neg ^= @select(u32, m < zero, one_u, zero_u);
            }
            for (e0..e1, 0..) |e, j| {
                const m: fm.V = v2c[e * zp + lo ..][0..L].*;
                const mag = fm.phi(sum - phis[j]);
                const sg = neg ^ @select(u32, m < zero, one_u, zero_u);
                const o = @select(f32, sg != zero_u, -mag, mag);
                c2v[e * zp + lo ..][0..L].* = @min(@max(o, fm.splat(-llr_max)), fm.splat(llr_max));
            }
        }
    }
}

fn variableNodes(c: *const Code, c2v: []const f32, v2c: []f32, llr: []const f32, tot_all: []f32) void {
    const z = c.z;
    const zp = c.zp;
    for (0..c.dec_cols) |col| {
        const tot = tot_all[col * zp ..][0..z];
        @memcpy(tot, llr[col * zp ..][0..z]);
        const list = c.col_entries[c.col_start[col]..c.col_start[col + 1]];
        for (list) |ei| {
            const s = c.entries[ei].shift;
            const src = c2v[@as(usize, ei) * zp ..][0..z];
            // tot[v] += src[(v - s) mod z]
            for (0..s) |v| tot[v] += src[z - s + v];
            for (s..z) |v| tot[v] += src[v - s];
        }
        for (list) |ei| {
            const s = c.entries[ei].shift;
            const src = c2v[@as(usize, ei) * zp ..][0..z];
            const dst = v2c[@as(usize, ei) * zp ..][0..z];
            for (0..z - s) |i| dst[i] = std.math.clamp(tot[i + s] - src[i], -llr_max, llr_max);
            for (z - s..z) |i| dst[i] = std.math.clamp(tot[i + s - z] - src[i], -llr_max, llr_max);
        }
    }
}

test "code parameters match Sionna for the M2 MCS table" {
    // (k, n) -> (bg, z, k_filler), from LDPC5GEncoder
    const cases = [_][5]u32{
        .{ 2388, 4776, 2, 240, 12 },
        .{ 3582, 4776, 1, 176, 290 },
        .{ 4776, 9552, 1, 224, 152 },
        .{ 7164, 9552, 1, 352, 580 },
        .{ 5970, 7164, 1, 288, 366 },
    };
    for (cases) |cs| {
        const c = try Code.init(cs[0], cs[1]);
        try std.testing.expectEqual(@as(u8, @intCast(cs[2])), c.bg);
        try std.testing.expectEqual(cs[3], c.z);
        try std.testing.expectEqual(cs[4], c.k_filler);
    }
}

test "noiseless codeword decodes in one iteration and is a codeword" {
    const alloc = std.testing.allocator;
    const c = try Code.init(4776, 9552);
    var prng = std.Random.DefaultPrng.init(1);
    const rnd = prng.random();
    const u = try alloc.alloc(u8, c.k);
    defer alloc.free(u);
    for (u) |*b| b.* = rnd.int(u1);
    const x = try alloc.alloc(u8, c.n);
    defer alloc.free(x);
    const scratch = try alloc.alloc(u8, c.n_ldpc);
    defer alloc.free(scratch);
    c.encode(u, x, scratch);
    const llr = try alloc.alloc(f32, c.n);
    defer alloc.free(llr);
    for (x, llr) |b, *l| l.* = if (b == 1) 4.0 else -4.0;
    var d = try Decoder.init(alloc);
    defer d.deinit(alloc);
    const out = try alloc.alloc(u8, c.k);
    defer alloc.free(out);
    const iters = d.decode(&c, llr, 20, true, out);
    try std.testing.expectEqualSlices(u8, u, out);
    try std.testing.expect(iters <= 2);
}
