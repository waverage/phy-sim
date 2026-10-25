//! Square Gray QAM as Sionna's qam(): the first bit of a symbol is the MSB
//! of its index, even bits label the in-phase PAM and odd bits the
//! quadrature PAM (3GPP TS 38.211 5.1), unit average energy.
//! The demapper is Sionna's "app": exact LLRs log p(1) / p(0) from all
//! points, evaluated per dimension, which for a product constellation is
//! the same sum.

const std = @import("std");
const C = @import("ofdm.zig").C;

pub const max_bits = 8;

pub const Qam = struct {
    bits: u32, // per symbol
    m: u32, // per dimension
    pam: [1 << (max_bits / 2)]f32, // amplitude per dimension label, MSB first

    pub fn init(bits_per_symbol: u32) error{UnsupportedQam}!Qam {
        if (bits_per_symbol == 0 or bits_per_symbol % 2 != 0 or bits_per_symbol > max_bits) return error.UnsupportedQam;
        var q: Qam = undefined;
        q.bits = bits_per_symbol;
        q.m = bits_per_symbol / 2;
        // qam_var = 1 / 2^(m-2) * sum_{i=1}^{2^(m-1)} (2i-1)^2
        var sq: f32 = 0;
        for (1..(@as(u32, 1) << @intCast(q.m - 1)) + 1) |i| {
            const a: f32 = @floatFromInt(2 * i - 1);
            sq += a * a;
        }
        const scale = std.math.pow(f32, 2.0, @as(f32, @floatFromInt(q.m)) - 2.0);
        // numpy's complex64 / float32 multiplies by the reciprocal, which
        // rounds differently from a division for 64-QAM
        const inv_norm = 1.0 / @sqrt(sq / scale);
        for (0..(@as(usize, 1) << @intCast(q.m))) |label| {
            q.pam[label] = @as(f32, @floatFromInt(pamGray(@intCast(label), q.m))) * inv_norm;
        }
        return q;
    }

    /// Sionna's pam_gray on the m-bit label, MSB first.
    fn pamGray(label: u32, m: u32) i32 {
        const b0: i32 = @intCast((label >> @intCast(m - 1)) & 1);
        if (m == 1) return 1 - 2 * b0;
        const rest = label & ((@as(u32, 1) << @intCast(m - 1)) - 1);
        return (1 - 2 * b0) * ((@as(i32, 1) << @intCast(m - 1)) - pamGray(rest, m - 1));
    }

    pub fn map(q: *const Qam, bits: []const u8) C {
        var li: u32 = 0;
        var lq: u32 = 0;
        for (0..q.m) |j| {
            li = (li << 1) | bits[2 * j];
            lq = (lq << 1) | bits[2 * j + 1];
        }
        return .{ .re = q.pam[li], .im = q.pam[lq] };
    }

    /// LLRs of the q.bits bits of one symbol.
    pub fn demap(q: *const Qam, y: C, no_in: f32, out: []f32) void {
        const no = @max(no_in, std.math.floatMin(f32));
        q.demapDim(y.re, no, out, 0);
        q.demapDim(y.im, no, out, 1);
    }

    fn demapDim(q: *const Qam, y: f32, no: f32, out: []f32, offset: usize) void {
        const npts = @as(usize, 1) << @intCast(q.m);
        var ex: [1 << (max_bits / 2)]f32 = undefined;
        for (0..npts) |l| {
            const d = y - q.pam[l];
            ex[l] = -(d * d) / no;
        }
        for (0..q.m) |j| {
            const sh: u5 = @intCast(q.m - 1 - j);
            var mx = [2]f32{ -std.math.inf(f32), -std.math.inf(f32) };
            for (0..npts) |l| {
                const b = (l >> sh) & 1;
                mx[b] = @max(mx[b], ex[l]);
            }
            var s = [2]f32{ 0, 0 };
            for (0..npts) |l| {
                const b = (l >> sh) & 1;
                s[b] += @exp(ex[l] - mx[b]);
            }
            out[2 * j + offset] = (mx[1] + @log(s[1])) - (mx[0] + @log(s[0]));
        }
    }
};

test "qam16 points and llr signs" {
    const q = try Qam.init(4);
    const s = q.map(&[_]u8{ 0, 0, 0, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / @sqrt(10.0)), s.re, 1e-7);
    const t = q.map(&[_]u8{ 1, 0, 1, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, -3.0 / @sqrt(10.0)), t.re, 1e-7);
    var llr: [4]f32 = undefined;
    q.demap(t, 0.01, &llr);
    try std.testing.expect(llr[0] > 0 and llr[1] < 0 and llr[2] > 0 and llr[3] < 0);
}
