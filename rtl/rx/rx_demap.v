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
// LLR i = d B + b is written to the deinterleaver at perm[i]. Sequential,
// with the label loop and the outputs pipelined: about
// npsym + 2^(B/2) + B + 7 clocks a data RE.
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
    output wire        [5:0]  at_addr,             // {s, t}
    input  wire signed [15:0] at_data,
    output wire        [2:0]  lv_addr,
    input  wire signed [3:0]  lv_data,
    output reg         [11:0] r_addr,
    input  wire        [15:0] r_data,
    output reg         [13:0] perm_addr,
    input  wire        [13:0] perm_data,
    output reg         [9:0]  hf_addr,
    input  wire        [31:0] hf_data,
    output reg         [11:0] y_addr,
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
    localparam D_IDLE = 4'd0, D_NEXT = 4'd1, D_RD = 4'd2, D_H = 4'd3, D_P0 = 4'd4, D_P1 = 4'd5, D_LAB = 4'd6,
               D_OUT = 4'd7, D_END = 4'd8, D_H1 = 4'd9;
    reg [3:0]  st;
    reg [3:0]  s;
    reg [7:0]  f;
    reg [11:0] d;
    reg [13:0] pbase;                       // d B
    reg [3:0]  k;                           // read / label / output counter
    reg [11:0] sbase;                       // s F

    wire is_pilot_s = (s == psym0) || (s == psym1) || (npsym == 2'd3 && s == psym2);
    wire [1:0] md = bps[2:1];               // bits per dimension
    wire [3:0] nl = 4'd1 << md;

    // ---- the time filter
    reg signed [47:0] ahr, ahi;
    wire [1:0] tk = k[1:0] - 2'd1;          // the pilot symbol whose estimate arrives
    assign at_addr = {s, tk};
    wire signed [15:0] hfr = hf_data[31:16], hfi = hf_data[15:0];
    wire signed [31:0] pr = at_data * hfr;
    wire signed [31:0] pi = at_data * hfi;

    reg signed [47:0] qhr, qhi;               // the estimate rounded by 14 bits
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

    // ---- per RE: z, |h|^2, then u and mult
    reg signed [15:0] hr, hi, yr, yi;
    reg        [15:0] rq;
    reg signed [33:0] zr, zi;
    reg        [31:0] h2;
    /* verilator lint_off UNUSEDSIGNAL */
    reg        [33:0] mm;                 // mant r + 2^14: bits 32..15 are mult
    /* verilator lint_on UNUSEDSIGNAL */
    reg        [32:0] u;
    reg        [17:0] mult;
    /* verilator lint_off UNUSEDSIGNAL */
    wire       [48:0] uu = h2 * a_q + 49'd32768;
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- labels, pipelined: L0 the level, L1 the products and the label's
    // bits, L2 the metrics, then the minima, [dim][j][bit] at (dim 3 + j) 2 + bit
    localparam MW = 43;
    localparam signed [MW-1:0] TOP = {1'b0, {(MW-1){1'b1}}};
    assign lv_addr = k[2:0];
    reg                l0_v, l1_v, l2_v;
    reg         [2:0]  l0_k;
    reg  signed [3:0]  l0_lv;
    wire signed [7:0]  lv2 = l0_lv * l0_lv;
    reg  signed [41:0] l1_ul;
    reg  signed [37:0] l1_zr, l1_zi;
    reg                l1b_v;
    reg         [2:0]  l1b_lb;
    reg  signed [41:0] l1b_ul;
    reg  signed [37:0] l1b_zr, l1b_zi;
    reg         [2:0]  l1_lb, l2_lb;            // bit j of the label, j < md
    reg  signed [MW-1:0] l2_m0, l2_m1;
    reg         [2:0]  lbits;
    reg  [MW*12-1:0] mn, n_mn;
    integer jj, dm, ix;
    always @(*) begin
        for (jj = 0; jj < 3; jj = jj + 1)
            lbits[jj] = (jj < {30'd0, md}) ? l0_k[md - 2'd1 - jj[1:0]] : 1'b0;
        n_mn = mn;
        for (dm = 0; dm < 2; dm = dm + 1)
            for (jj = 0; jj < 3; jj = jj + 1) begin
                ix = (dm * 3 + jj) * 2 + (l2_lb[jj] ? 1 : 0);
                if (jj < {30'd0, md}) begin
                    if (dm == 0 && l2_m0 < $signed(mn[ix*MW +: MW]))
                        n_mn[ix*MW +: MW] = l2_m0;
                    if (dm == 1 && l2_m1 < $signed(mn[ix*MW +: MW]))
                        n_mn[ix*MW +: MW] = l2_m1;
                end
            end
    end

    // ---- outputs, pipelined: O1 the difference (and the perm read), O2 the
    // product (and perm[i]), O3 rounding and saturation -> the write
    wire [3:0] o_ix = (k[0] ? 4'd6 : 4'd0) + {1'b0, k[2:1], 1'b0};
    wire [3:0] o_ix1 = o_ix + 4'd1;
    wire signed [MW:0] dd = $signed({mn[o_ix*MW + MW - 1], mn[o_ix*MW +: MW]})
                          - $signed({mn[o_ix1*MW + MW - 1], mn[o_ix1*MW +: MW]});
    reg                o1_v, o2_v;
    reg  signed [MW:0] o1_dd;
    reg         [13:0] o1_i, o2_i, o2_perm;
    reg  signed [63:0] o2_p;
    reg                o3_v;
    reg         [13:0] o3_i, o3_perm;
    reg  signed [63:0] o3_x, half;              // product + 2^(shift - 1)
    wire signed [63:0] rnd = o3_x >>> shift;
    wire signed [5:0]  llr = (rnd > 64'sd31) ? 6'sd31 : (rnd < -64'sd31) ? -6'sd31 : rnd[5:0];

    always @(*) begin
        case (k[1:0])
            2'd0: hf_addr = {2'd0, f};
            2'd1: hf_addr = {2'd0, nf} + {2'd0, f};
            default: hf_addr = {1'b0, nf, 1'b0} + {2'd0, f};
        endcase
        y_addr = sbase + {4'd0, f};
        r_addr = d;
        perm_addr = pbase + {10'd0, k};
    end

    always @(posedge clk) begin
        // label pipeline
        l0_v  <= (st == D_LAB) && (k < nl) && rst_n;
        l0_k  <= k[2:0];
        l0_lv <= lv_data;
        l1_v  <= l0_v && rst_n;
        l1_lb <= lbits;
        l1_ul <= $signed({1'b0, u}) * lv2;
        l1_zr <= zr * l0_lv;
        l1_zi <= zi * l0_lv;
        l1b_v  <= l1_v && rst_n;          // a second product register: the DSP48s' own
        l1b_lb <= l1_lb;
        l1b_ul <= l1_ul;
        l1b_zr <= l1_zr;
        l1b_zi <= l1_zi;
        l2_v  <= l1b_v && rst_n;
        l2_lb <= l1b_lb;
        l2_m0 <= {l1b_ul[41], l1b_ul} - {{4{l1b_zr[37]}}, l1b_zr, 1'b0};
        l2_m1 <= {l1b_ul[41], l1b_ul} - {{4{l1b_zi[37]}}, l1b_zi, 1'b0};
        if (st == D_P1)
            mn <= {12{TOP}};
        else if (l2_v)
            mn <= n_mn;
        // output pipeline
        o1_v  <= (st == D_OUT) && rst_n;
        o1_dd <= dd;
        o1_i  <= pbase + {10'd0, k};
        o2_v  <= o1_v && rst_n;
        o2_p  <= o1_dd * $signed({1'b0, mult});
        o2_i  <= o1_i;
        o2_perm <= perm_data;
        half  <= 64'sd1 <<< (shift - 6'd1);
        o3_v  <= o2_v && rst_n;
        o3_x  <= o2_p + half;
        o3_i  <= o2_i;
        o3_perm <= o2_perm;
        llr_we   <= o3_v && rst_n;
        llr_addr <= o3_perm;
        llr_data <= llr;
        dbg_llr_valid <= o3_v && rst_n;
        dbg_llr_idx <= o3_i;
        dbg_llr <= llr;

        if (!rst_n) begin
            st <= D_IDLE;
            done <= 1'b0;
            dbg_h_valid <= 1'b0;
        end else begin
            done <= 1'b0;
            dbg_h_valid <= 1'b0;
            case (st)
                D_IDLE: if (start) begin
                    s <= 4'd0;
                    sbase <= 12'd0;
                    f <= 8'd0;
                    d <= 12'd0;
                    pbase <= 14'd0;
                    k <= 4'd0;
                    st <= D_NEXT;
                end
                D_NEXT: begin                       // skip the pilot symbols
                    k <= 4'd0;
                    if (s == num_sym)
                        st <= D_END;
                    else if (is_pilot_s) begin
                        s <= s + 4'd1;
                        sbase <= sbase + {4'd0, nf};
                    end else
                        st <= D_RD;
                end
                D_RD: begin                         // reads at k, the estimate of k - 1 arrives at k
                    if (k == 4'd1) begin
                        yr <= y_data[31:16];
                        yi <= y_data[15:0];
                        rq <= r_data;
                    end
                    if (k >= 4'd1) begin
                        ahr <= ((k == 4'd1) ? 48'sd0 : ahr) + {{16{pr[31]}}, pr};
                        ahi <= ((k == 4'd1) ? 48'sd0 : ahi) + {{16{pi[31]}}, pi};
                    end
                    if (k == {2'b00, npsym})
                        st <= D_H;
                    k <= k + 4'd1;
                end
                D_H: begin                          // rounded here, saturated in D_H1
                    qhr <= (ahr + 48'sd8192) >>> 14;
                    qhi <= (ahi + 48'sd8192) >>> 14;
                    st <= D_H1;
                end
                D_H1: begin
                    hr <= sat16(qhr);
                    hi <= sat16(qhi);
                    st <= D_P0;
                end
                D_P0: begin
                    zr <= yr * hr + yi * hi;
                    zi <= yi * hr - yr * hi;
                    h2 <= hr * hr + hi * hi;
                    mm <= mant * rq + 34'd16384;
                    st <= D_P1;
                end
                D_P1: begin
                    u <= uu[48:16];
                    mult <= mm[32:15];
                    dbg_h_valid <= 1'b1;
                    dbg_h <= {hr, hi};
                    dbg_y <= {yr, yi};
                    k <= 4'd0;
                    st <= D_LAB;
                end
                D_LAB: begin                        // k = 0 .. nl + 3: the last minimum lands with k = nl + 3
                    k <= k + 4'd1;
                    if (k == nl + 4'd3) begin
                        k <= 4'd0;
                        st <= D_OUT;
                    end
                end
                D_OUT: begin
                    k <= k + 4'd1;
                    if (k == {1'b0, bps} - 4'd1) begin
                        k <= 4'd0;
                        d <= d + 12'd1;
                        pbase <= pbase + {11'd0, bps};
                        if (f == nf - 8'd1) begin
                            f <= 8'd0;
                            s <= s + 4'd1;
                            sbase <= sbase + {4'd0, nf};
                            st <= D_NEXT;
                        end else begin
                            f <= f + 8'd1;
                            st <= D_RD;
                        end
                    end
                end
                D_END: if (!o1_v && !o2_v && !o3_v && !llr_we) begin   // the last LLR is written
                    st <= D_IDLE;
                    done <= 1'b1;
                end
                default: st <= D_IDLE;
            endcase
        end
    end
endmodule
