`timescale 1ns / 1ps
// Time filter and demapper (fxmodem.rxFrame steps 5-6, twin.TwinModem
// ._demap): for each data RE d (symbol s, subcarrier f, in frame order)
//
//   h = sum_t at[s, t] hf[t, f] rounded half up by 14 bits, sat to 16
//   z = y conj(h), u = |h|^2 a_q 2^-16, mult = mant r[d] 2^-15 (rounded)
//   per dimension v of z and label l: met_l = u lv_l^2 - 2 v lv_l
//   LLR(2 j + dim) = sat6(rshr((min over bit j = 0 - min over bit j = 1) mult, shift))
//
// exactly as the i64 arithmetic of the Zig receiver, at the widths its
// values need (|z| <= 2^31, u <= 2^31, |met| < 2^38, |dd mult| < 2^55).
// LLR i = d B + b goes to the decoder's LLR memory at perm[i] = {discard,
// word, lane}; the discarded ones (padding past the codewords) are not
// written.
//
// Pipelined: one RE every T = 2^(B/2) clocks (the labels of a dimension,
// both dimensions at once; B <= T outputs follow), so about T clocks a
// data RE. The time filter reads all pilot estimates of subcarrier f in
// one word (hf_data, pilot symbol t at bits 32 t) and their weights at[s, t]
// in another; the arithmetic between the reads and the labels is a fixed
// 7-clock pipeline.
module rx_demap (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire        [7:0]  nf,
    input  wire        [3:0]  num_sym,
    input  wire        [1:0]  npsym,
    input  wire        [3:0]  psym0,
    input  wire        [3:0]  psym1,
    input  wire        [3:0]  psym2,
    input  wire        [2:0]  bps,
    input  wire        [16:0] a_q,
    input  wire        [16:0] mant,
    input  wire        [5:0]  shift,
    output wire        [3:0]  at_s,                // at_data = {at[s, 2], at[s, 1], at[s, 0]}
    input  wire        [47:0] at_data,
    output wire        [2:0]  lv_addr,
    input  wire signed [3:0]  lv_data,
    output wire        [11:0] r_addr,
    input  wire        [15:0] r_data,
    output wire        [13:0] perm_addr,
    input  wire        [14:0] perm_data,
    output wire        [7:0]  hf_addr,
    input  wire        [95:0] hf_data,
    output wire        [11:0] y_addr,
    input  wire        [31:0] y_data,
    output reg                llr_we,
    output reg         [13:0] llr_addr,
    output reg  signed [5:0]  llr_data,
    output reg                dbg_h_valid,
    output reg         [31:0] dbg_h,
    output reg         [31:0] dbg_y,
    output reg                dbg_llr_valid,
    output reg         [13:0] dbg_llr_idx,
    output reg  signed [5:0]  dbg_llr,
    output reg                done
);
    localparam D_IDLE = 2'd0, D_RUN = 2'd1, D_END = 2'd2;
    reg [1:0]  st;
    reg [3:0]  s;
    reg [7:0]  f;
    reg [11:0] d;
    reg [13:0] pbase;                       // d B
    reg [11:0] sbase;                       // s F
    reg [2:0]  k;                           // the slot's clock, 0 .. T - 1

    wire is_pilot_s = (s == psym0) || (s == psym1) || (npsym == 2'd3 && s == psym2);
    wire [1:0] md = bps[2:1];               // bits per dimension
    wire [2:0] tm1 = (3'd1 << md) - 3'd1;    // T - 1, T = 2^md
    wire       launch = (st == D_RUN) && (k == 3'd0) && (s != num_sym) && !is_pilot_s;

    assign at_s = s;
    assign hf_addr = f;
    assign y_addr = sbase + {4'd0, f};
    assign r_addr = d;

    function signed [15:0] sat16;
        input signed [47:0] q;
        begin
            if (q > 48'sd32767)
                sat16 = 16'sh7fff;
            else if (q < -48'sd32768)
                sat16 = 16'sh8000;
            else
                sat16 = q[15:0];
        end
    endfunction

    // ---- the arithmetic pipeline, one token per RE: F1 the memories
    // answer, F2 the time filter's products, F3 their sums, F4 rounded,
    // F5 saturated, F6 z, |h|^2 and mant r, F7 u and mult
    reg               f1_v, f2_v, f3_v, f4_v, f5_v, f6_v, f7_v;
    reg        [13:0] f1_pb, f2_pb, f3_pb, f4_pb, f5_pb, f6_pb, f7_pb;
    reg        [47:0] f1_at;
    reg signed [31:0] f2_pr [0:2];
    reg signed [31:0] f2_pi [0:2];
    reg        [31:0] f2_y, f3_y, f4_y, f5_y, f6_y;
    reg        [15:0] f2_r, f3_r, f4_r, f5_r;
    reg signed [47:0] f3_hr, f3_hi, f4_hr, f4_hi;
    reg signed [15:0] f5_hr, f5_hi, f6_hr, f6_hi;
    reg signed [33:0] f6_zr, f6_zi, f7_zr, f7_zi;
    reg        [31:0] f6_h2;
    /* verilator lint_off UNUSEDSIGNAL */
    reg        [33:0] f6_mm;                // mant r + 2^14: bits 32..15 are mult
    wire       [48:0] uu = f6_h2 * a_q + 49'd32768;
    /* verilator lint_on UNUSEDSIGNAL */
    reg        [32:0] f7_u;
    reg        [17:0] f7_mult;
    wire signed [15:0] f5yr = f5_y[31:16], f5yi = f5_y[15:0];
    integer t;
    always @(posedge clk) begin
        f1_v  <= launch && rst_n;
        f1_pb <= pbase;
        f1_at <= at_data;
        f2_v  <= f1_v && rst_n;
        f2_pb <= f1_pb;
        for (t = 0; t < 3; t = t + 1) begin
            f2_pr[t] <= $signed(f1_at[16*t +: 16]) * $signed(hf_data[32*t+16 +: 16]);
            f2_pi[t] <= $signed(f1_at[16*t +: 16]) * $signed(hf_data[32*t +: 16]);
        end
        f2_y  <= y_data;
        f2_r  <= r_data;
        f3_v  <= f2_v && rst_n;
        f3_pb <= f2_pb;
        f3_hr <= {{16{f2_pr[0][31]}}, f2_pr[0]} + {{16{f2_pr[1][31]}}, f2_pr[1]} +
                 ((npsym == 2'd3) ? {{16{f2_pr[2][31]}}, f2_pr[2]} : 48'sd0);
        f3_hi <= {{16{f2_pi[0][31]}}, f2_pi[0]} + {{16{f2_pi[1][31]}}, f2_pi[1]} +
                 ((npsym == 2'd3) ? {{16{f2_pi[2][31]}}, f2_pi[2]} : 48'sd0);
        f3_y  <= f2_y;
        f3_r  <= f2_r;
        f4_v  <= f3_v && rst_n;
        f4_pb <= f3_pb;
        f4_hr <= (f3_hr + 48'sd8192) >>> 14;
        f4_hi <= (f3_hi + 48'sd8192) >>> 14;
        f4_y  <= f3_y;
        f4_r  <= f3_r;
        f5_v  <= f4_v && rst_n;
        f5_pb <= f4_pb;
        f5_hr <= sat16(f4_hr);
        f5_hi <= sat16(f4_hi);
        f5_y  <= f4_y;
        f5_r  <= f4_r;
        f6_v  <= f5_v && rst_n;
        f6_pb <= f5_pb;
        f6_zr <= f5yr * f5_hr + f5yi * f5_hi;
        f6_zi <= f5yi * f5_hr - f5yr * f5_hi;
        f6_h2 <= f5_hr * f5_hr + f5_hi * f5_hi;
        f6_mm <= mant * f5_r + 34'd16384;
        f6_hr <= f5_hr;
        f6_hi <= f5_hi;
        f6_y  <= f5_y;
        f7_v  <= f6_v && rst_n;
        f7_pb <= f6_pb;
        f7_u  <= uu[48:16];
        f7_mult <= f6_mm[32:15];
        f7_zr <= f6_zr;
        f7_zi <= f6_zi;
        dbg_h_valid <= f6_v && rst_n;
        dbg_h <= {f6_hr, f6_hi};
        dbg_y <= f6_y;
    end

    // ---- labels: label 0 straight from F7, labels 1 .. T-1 from the latch.
    // L0 the level, L1 the products and the label's bits, L1b the DSP
    // register, L2 the metrics, then the minima, [dim][j][bit] at
    // (dim 3 + j) 2 + bit
    localparam MW = 43;
    localparam signed [MW-1:0] TOP = {1'b0, {(MW-1){1'b1}}};
    reg               lab_act;
    reg        [2:0]  lk;
    reg        [32:0] lab_u;
    reg signed [33:0] lab_zr, lab_zi;
    reg        [17:0] lab_mult;
    reg        [13:0] lab_pb;
    wire       [2:0]  ik = f7_v ? 3'd0 : lk;
    assign lv_addr = ik;
    reg               l0_v, l0_first, l0_last, l1_v, l1_first, l1_last, l1b_v, l1b_first, l1b_last;
    reg               l2_v, l2_first, l2_last;
    reg        [2:0]  l0_k;
    reg signed [3:0]  l0_lv;
    reg        [32:0] l0_u;
    reg signed [33:0] l0_zr, l0_zi;
    reg        [17:0] l0_mult, l1_mult, l1b_mult, l2_mult;
    reg        [13:0] l0_pb, l1_pb, l1b_pb, l2_pb;
    wire signed [7:0] lv2 = l0_lv * l0_lv;
    reg  signed [41:0] l1_ul, l1b_ul, l1c_ul;
    reg  signed [37:0] l1_zr, l1_zi, l1b_zr, l1b_zi, l1c_zr, l1c_zi;
    reg         [2:0]  l1_lb, l1b_lb, l1c_lb, l2_lb;      // bit j of the label, j < md
    reg                l1c_v, l1c_first, l1c_last;
    reg         [17:0] l1c_mult;
    reg         [13:0] l1c_pb;
    reg  signed [MW-1:0] l2_m0, l2_m1;
    reg         [2:0]  lbits;
    reg  [MW*12-1:0] mn, n_mn, base;
    integer jj, dm, ix;
    always @(*) begin
        for (jj = 0; jj < 3; jj = jj + 1)
            lbits[jj] = (jj < {30'd0, md}) ? l0_k[md - 2'd1 - jj[1:0]] : 1'b0;
        base = l2_first ? {12{TOP}} : mn;
        n_mn = base;
        for (dm = 0; dm < 2; dm = dm + 1)
            for (jj = 0; jj < 3; jj = jj + 1) begin
                ix = (dm * 3 + jj) * 2 + (l2_lb[jj] ? 1 : 0);
                if (jj < {30'd0, md}) begin
                    if (dm == 0 && l2_m0 < $signed(base[ix*MW +: MW]))
                        n_mn[ix*MW +: MW] = l2_m0;
                    if (dm == 1 && l2_m1 < $signed(base[ix*MW +: MW]))
                        n_mn[ix*MW +: MW] = l2_m1;
                end
            end
    end

    // ---- outputs, B a RE from the minima of its last label: O1 the
    // difference (and the perm read), O2 the product (and perm[i]), O3
    // rounding and saturation -> the write
    reg  [MW*12-1:0] mn_q;
    reg              o_act;
    reg        [2:0] ok;
    reg       [17:0] o_mult;
    reg       [13:0] o_pb;
    wire [3:0] o_ix = (ok[0] ? 4'd6 : 4'd0) + {1'b0, ok[2:1], 1'b0};
    wire [3:0] o_ix1 = o_ix + 4'd1;
    wire signed [MW:0] dd = $signed({mn_q[o_ix*MW + MW - 1], mn_q[o_ix*MW +: MW]})
                          - $signed({mn_q[o_ix1*MW + MW - 1], mn_q[o_ix1*MW +: MW]});
    assign perm_addr = o_pb + {11'd0, ok};
    reg                o1_v, o2_v, o3_v;
    reg  signed [MW:0] o1_dd;
    reg         [13:0] o1_i, o2_i, o3_i;
    reg         [17:0] o1_mult;
    reg         [14:0] o2_perm, o3_perm;
    reg  signed [63:0] o2_p;
    reg  signed [63:0] o3_x, half;              // product + 2^(shift - 1)
    wire signed [63:0] rnd = o3_x >>> shift;
    wire signed [5:0]  llr = (rnd > 64'sd31) ? 6'sd31 : (rnd < -64'sd31) ? -6'sd31 : rnd[5:0];

    always @(posedge clk) begin
        // label issue
        l0_v     <= (f7_v || lab_act) && rst_n;
        l0_k     <= ik;
        l0_lv    <= lv_data;
        l0_first <= f7_v;
        l0_last  <= (ik == tm1);
        l0_u     <= f7_v ? f7_u : lab_u;
        l0_zr    <= f7_v ? f7_zr : lab_zr;
        l0_zi    <= f7_v ? f7_zi : lab_zi;
        l0_mult  <= f7_v ? f7_mult : lab_mult;
        l0_pb    <= f7_v ? f7_pb : lab_pb;
        if (f7_v) begin
            lab_u    <= f7_u;
            lab_zr   <= f7_zr;
            lab_zi   <= f7_zi;
            lab_mult <= f7_mult;
            lab_pb   <= f7_pb;
        end
        // label pipeline
        l1_v     <= l0_v && rst_n;
        l1_first <= l0_first;
        l1_last  <= l0_last;
        l1_mult  <= l0_mult;
        l1_pb    <= l0_pb;
        l1_lb    <= lbits;
        l1_ul    <= $signed({1'b0, l0_u}) * lv2;
        l1_zr    <= l0_zr * l0_lv;
        l1_zi    <= l0_zi * l0_lv;
        l1b_v     <= l1_v && rst_n;            // a second product register: the DSP48s' own
        l1b_first <= l1_first;
        l1b_last  <= l1_last;
        l1b_mult  <= l1_mult;
        l1b_pb    <= l1_pb;
        l1b_lb    <= l1_lb;
        l1b_ul    <= l1_ul;
        l1b_zr    <= l1_zr;
        l1b_zi    <= l1_zi;
        l1c_v     <= l1b_v && rst_n;           // a third register: the 34 x 8 product spans two DSP48s
        l1c_first <= l1b_first;
        l1c_last  <= l1b_last;
        l1c_mult  <= l1b_mult;
        l1c_pb    <= l1b_pb;
        l1c_lb    <= l1b_lb;
        l1c_ul    <= l1b_ul;
        l1c_zr    <= l1b_zr;
        l1c_zi    <= l1b_zi;
        l2_v     <= l1c_v && rst_n;
        l2_first <= l1c_first;
        l2_last  <= l1c_last;
        l2_mult  <= l1c_mult;
        l2_pb    <= l1c_pb;
        l2_lb    <= l1c_lb;
        l2_m0    <= {l1c_ul[41], l1c_ul} - {{4{l1c_zr[37]}}, l1c_zr, 1'b0};
        l2_m1    <= {l1c_ul[41], l1c_ul} - {{4{l1c_zi[37]}}, l1c_zi, 1'b0};
        if (l2_v)
            mn <= n_mn;
        if (l2_v && l2_last) begin
            mn_q   <= n_mn;
            o_mult <= l2_mult;
            o_pb   <= l2_pb;
        end
        // output pipeline
        o1_v    <= o_act && rst_n;
        o1_dd   <= dd;
        o1_i    <= perm_addr;
        o1_mult <= o_mult;
        o2_v    <= o1_v && rst_n;
        o2_p    <= o1_dd * $signed({1'b0, o1_mult});
        o2_i    <= o1_i;
        o2_perm <= perm_data;
        half    <= 64'sd1 <<< (shift - 6'd1);
        o3_v    <= o2_v && rst_n;
        o3_x    <= o2_p + half;
        o3_i    <= o2_i;
        o3_perm <= o2_perm;
        llr_we   <= o3_v && !o3_perm[14] && rst_n;
        llr_addr <= o3_perm[13:0];
        llr_data <= llr;
        dbg_llr_valid <= o3_v && rst_n;
        dbg_llr_idx <= o3_i;
        dbg_llr <= llr;

        if (!rst_n) begin
            st <= D_IDLE;
            done <= 1'b0;
            lab_act <= 1'b0;
            o_act <= 1'b0;
        end else begin
            done <= 1'b0;
            // the label counter: labels 1 .. T-1 after F7
            if (f7_v) begin
                lab_act <= 1'b1;
                lk <= 3'd1;
            end else if (lab_act) begin
                lk <= lk + 3'd1;
                if (lk == tm1)
                    lab_act <= 1'b0;
            end
            // the output counter: B outputs after the last label
            if (l2_v && l2_last) begin
                o_act <= 1'b1;
                ok <= 3'd0;
            end else if (o_act) begin
                ok <= ok + 3'd1;
                if (ok == bps - 3'd1)
                    o_act <= 1'b0;
            end
            case (st)
                D_IDLE: if (start) begin
                    s <= 4'd0;
                    sbase <= 12'd0;
                    f <= 8'd0;
                    d <= 12'd0;
                    pbase <= 14'd0;
                    k <= 3'd0;
                    st <= D_RUN;
                end
                D_RUN: begin
                    k <= (k == tm1) ? 3'd0 : k + 3'd1;
                    if (k == 3'd0) begin
                        if (s == num_sym)
                            st <= D_END;
                        else if (is_pilot_s) begin  // a slot without a RE
                            s <= s + 4'd1;
                            sbase <= sbase + {4'd0, nf};
                        end else begin
                            d <= d + 12'd1;
                            pbase <= pbase + {11'd0, bps};
                            if (f == nf - 8'd1) begin
                                f <= 8'd0;
                                s <= s + 4'd1;
                                sbase <= sbase + {4'd0, nf};
                            end else
                                f <= f + 8'd1;
                        end
                    end
                end
                D_END: if (!f1_v && !f2_v && !f3_v && !f4_v && !f5_v && !f6_v && !f7_v && !lab_act && !l0_v && !l1_v &&
                           !l1b_v && !l1c_v && !l2_v && !o_act && !o1_v && !o2_v && !o3_v && !llr_we && !dbg_llr_valid) begin
                    st <= D_IDLE;
                    done <= 1'b1;
                end
                default: st <= D_IDLE;
            endcase
        end
    end
endmodule
