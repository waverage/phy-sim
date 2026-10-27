`timescale 1ns / 1ps
// Phase tracker (fxmodem.track, twin.TwinModem._track): one phase ramp
// per frame from the pilot symbols' common phases, then the frame and the
// pilot estimates derotated by it.
//
//   z_t = sum_f hf_t conj(sum_t' hf_t')           (exact, 48 bits)
//   common shift: the largest |re|, |im| of all z_t to 30 bits
//   (mag_t, ang_t) = CORDIC(z_t >> shift), w_t = mag_t >> 16 rounded
//   num = sum w_t sv_t ang_t, den = sum w_t sv_t^2, sv_t = np p_t - sum p
//   phi_s = round(num sv_s gain / (den 2^15)), half up (floor((2a + b) / 2b))
//   y_s *= cossin(-phi_s) for every symbol, hf_t *= cossin(-phi_p_t)
//
// Sequential, every arithmetic step registered for 100 MHz: about
// F (2 npsym + 3) clocks for z, 70 per symbol for the division
// (restoring, on magnitudes), F + 22 per derotated symbol.
module rx_track (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire        [7:0]  nf,
    input  wire        [3:0]  num_sym,
    input  wire        [1:0]  npsym,
    input  wire        [3:0]  psym0,
    input  wire        [3:0]  psym1,
    input  wire        [3:0]  psym2,
    input  wire        [5:0]  sum_psym,
    input  wire        [16:0] gain,
    output reg         [9:0]  hf_raddr,
    input  wire        [31:0] hf_rdata,
    output reg                hf_we,
    output reg         [9:0]  hf_waddr,
    output reg         [31:0] hf_wdata,
    output reg         [11:0] y_raddr,
    input  wire        [31:0] y_rdata,
    output reg                y_we,
    output reg         [11:0] y_waddr,
    output reg         [31:0] y_wdata,
    output reg                phi_valid,
    output reg         [3:0]  phi_idx,
    output reg  signed [31:0] phi_val,
    output reg                done
);
    localparam T_IDLE = 5'd0, T_Z = 5'd1, T_ZW = 5'd2, T_MX0 = 5'd3, T_MX1 = 5'd4, T_SH = 5'd5, T_CV = 5'd6,
               T_ND = 5'd7, T_PHI0 = 5'd8, T_PHI1 = 5'd9, T_PHI2 = 5'd10, T_PHI3 = 5'd11, T_PHI4 = 5'd12,
               T_DIV = 5'd13, T_RA0 = 5'd14, T_RA = 5'd15, T_RW = 5'd16, T_RUN = 5'd17, T_RDRAIN = 5'd18, T_PHIS = 5'd19;
    reg [4:0] st;
    reg       go;                                 // start, registered

    function [3:0] psym_of;
        input [1:0] t;
        begin
            psym_of = (t == 2'd0) ? psym0 : (t == 2'd1) ? psym1 : psym2;
        end
    endfunction

    // ---- z accumulation, per f: reads at kk = 0 .. np-1, captures at
    // 1 .. np, the sums at np + 1, products at np + 2 .. 2 np + 1,
    // accumulated a clock later
    reg [7:0]  f;
    reg [2:0]  kk;
    reg signed [15:0] h_re [0:3];
    reg signed [15:0] h_im [0:3];
    reg signed [47:0] zr [0:3];
    reg signed [47:0] zi [0:3];
    reg signed [17:0] sr, si;
    wire [2:0] np3 = {1'b0, npsym};
    /* verilator lint_off UNUSEDSIGNAL */
    wire [2:0] tm = kk - np3 - 3'd2;              // the product's pilot symbol
    wire [2:0] kc = kk - 3'd1;                    // the capture's pilot symbol
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [15:0] mr = h_re[tm[1:0]], mi = h_im[tm[1:0]];
    reg               q_v, p_v;                // q: the operands, p: the products
    reg        [1:0]  q_t, p_t;
    reg signed [15:0] q_mr, q_mi;
    reg signed [34:0] p_zr, p_zi;

    // ---- the common shift: |z| registered, their maximum, its bit length
    reg [47:0] az [0:5];
    reg [47:0] mx;
    reg [6:0]  bl;
    reg [47:0] m01, m23, m45;
    integer b;
    always @(*) begin
        m01 = (az[0] > az[1]) ? az[0] : az[1];
        m23 = (az[2] > az[3]) ? az[2] : az[3];
        m45 = (az[4] > az[5]) ? az[4] : az[5];
        bl = 7'd0;
        for (b = 0; b < 48; b = b + 1)
            if (mx[b])
                bl = b[6:0] + 7'd1;
    end
    function [47:0] absz;
        input signed [47:0] v;
        input               use_it;
        begin
            absz = !use_it ? 48'd0 : v[47] ? -v : v;
        end
    endfunction
    reg [6:0] sh;

    // ---- CORDIC vectoring of the shifted z_t
    reg  [1:0]  cv_n, cv_got;
    reg         cv_in;
    reg  signed [31:0] cv_x, cv_y;
    reg  [1:0]  cv_tag;
    wire        cv_ov;
    wire signed [33:0] cv_mag;
    wire signed [17:0] cv_ang;
    wire [1:0]  cv_otag;
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [47:0] zsr = zr[cv_n] >>> sh;
    wire signed [47:0] zsi = zi[cv_n] >>> sh;
    /* verilator lint_on UNUSEDSIGNAL */
    cordic_vec #(.W(32), .TAGW(2)) cvec (
        .clk(clk), .rst_n(rst_n), .in_valid(cv_in), .in_x(cv_x), .in_y(cv_y), .in_tag(cv_tag),
        .out_valid(cv_ov), .out_mag(cv_mag), .out_ang(cv_ang), .out_tag(cv_otag));
    reg signed [33:0] mag [0:3];
    reg signed [17:0] ang [0:3];

    // ---- num, den: per pilot symbol, w sv registered, then the sums
    reg [1:0]  nt;
    reg [1:0]  nph;                               // the clock of the three
    reg signed [63:0] num, den;
    wire signed [7:0]  np8 = {6'd0, npsym};
    wire signed [7:0]  svt = np8 * $signed({4'd0, psym_of(nt)}) - $signed({2'd0, sum_psym});
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [33:0] wt34 = (mag[nt] + 34'sd32768) >>> 16;
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [18:0] wt = wt34[18:0];
    reg  signed [26:0] ws;                        // w sv
    reg  signed [18:0] wtr;
    reg  signed [44:0] pn;                        // w sv ang
    reg  signed [33:0] pd;                        // w sv^2
    reg  signed [7:0]  svr, svs_r;
    reg  signed [17:0] angr;

    // ---- phi per symbol: num sv_s gain in parts, then the division
    reg [3:0]  ps;
    wire signed [7:0] svs = np8 * $signed({4'd0, ps}) - $signed({2'd0, sum_psym});
    reg signed [55:0] m1;                         // num sv_s (|num| < 2^47)
    reg        [44:0] m1lo;                       // m1[27:0] gain
    reg signed [45:0] m1hi;                       // m1[55:28] gain
    reg signed [63:0] m2, numer;
    reg [63:0] dv_n, dv_d, dv_q;
    reg [64:0] dv_r;
    reg [6:0]  dv_i;
    reg        dv_neg;
    reg signed [31:0] phi [0:13];
    wire [64:0] r_sh = {dv_r[63:0], dv_n[63]};
    wire        r_ge = (r_sh >= {1'b0, dv_d});
    wire [63:0] q_fl = dv_q + {63'd0, (dv_r != 65'd0)};
    /* verilator lint_off UNUSEDSIGNAL */
    wire [63:0] q_neg = -q_fl;
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [31:0] phi_new = (den <= 64'sd0) ? 32'sd0 : dv_neg ? q_neg[31:0] : dv_q[31:0];

    // ---- derotation: units 0 .. S-1 are the symbols, S .. S+np-1 the pilot estimates
    reg  [4:0]  ru;
    wire        ru_h = (ru >= {1'b0, num_sym});
    /* verilator lint_off UNUSEDSIGNAL */
    wire [4:0]  ru_t = ru - {1'b0, num_sym};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [3:0]  ru_s = ru_h ? psym_of(ru_t[1:0]) : ru[3:0];
    wire [9:0]  ru_tf = {ru_t[1:0], 8'd0};       // the pilot estimates are at {t, f}
    wire [11:0] ru_sf = ru[3:0] * nf;
    reg  [11:0] ru_base;
    reg         ru_hr;
    reg         cr_in;
    wire        cr_ov;
    wire signed [17:0] cr_c, cr_s;
    /* verilator lint_off UNUSEDSIGNAL */
    wire        cr_tag;
    reg  signed [31:0] nphi;
    /* verilator lint_on UNUSEDSIGNAL */
    cordic_rot #(.TAGW(1)) crot (
        .clk(clk), .rst_n(rst_n), .in_valid(cr_in), .in_ang(nphi[15:0]), .in_tag(1'b0),
        .out_valid(cr_ov), .out_cos(cr_c), .out_sin(cr_s), .out_tag(cr_tag));
    reg signed [17:0] cs_c, cs_s;
    reg  [7:0]  rf;
    reg         r1_v, r2_v, r3_v;
    reg  [11:0] r1_a, r2_a, r3_a;
    reg  [31:0] r2_d;
    reg signed [35:0] r3_re, r3_im;
    wire signed [15:0] vr = r2_d[31:16], vi = r2_d[15:0];

    function signed [15:0] rnd_sat;          // rshr by 15 (half up), sat to 16 bits
        input signed [35:0] x;
        reg signed [35:0] qq;
        begin
            qq = (x + 36'sd16384) >>> 15;
            if (qq > 36'sd32767)
                rnd_sat = 16'sh7fff;
            else if (qq < -36'sd32768)
                rnd_sat = 16'sh8000;
            else
                rnd_sat = qq[15:0];
        end
    endfunction

    wire [9:0] kk_tf = {kk[1:0], 8'd0};
    always @(*) begin
        hf_raddr = kk_tf + {2'd0, f};
        y_raddr  = ru_base + {4'd0, rf};
        if (st == T_RUN && ru_hr)
            hf_raddr = ru_base[9:0] + {2'd0, rf};
    end

    integer q;
    always @(posedge clk) begin
        go <= start && rst_n;
        // the derotation pipeline: read, data, products, write
        r1_v <= (st == T_RUN) && rst_n;
        r1_a <= ru_base + {4'd0, rf};
        r2_v <= r1_v && rst_n;
        r2_a <= r1_a;
        r2_d <= ru_hr ? hf_rdata : y_rdata;
        r3_v <= r2_v && rst_n;
        r3_a <= r2_a;
        r3_re <= vr * cs_c - vi * cs_s;
        r3_im <= vr * cs_s + vi * cs_c;
        hf_we <= r3_v && ru_hr && rst_n;
        y_we  <= r3_v && !ru_hr && rst_n;
        hf_waddr <= r3_a[9:0];
        y_waddr  <= r3_a;
        hf_wdata <= {rnd_sat(r3_re), rnd_sat(r3_im)};
        y_wdata  <= {rnd_sat(r3_re), rnd_sat(r3_im)};
        // the z products and their accumulation
        q_v  <= (st == T_Z) && (kk >= np3 + 3'd2) && rst_n;
        q_t  <= tm[1:0];
        q_mr <= mr;
        q_mi <= mi;
        p_v  <= q_v && rst_n;
        p_t  <= q_t;
        p_zr <= q_mr * sr + q_mi * si;
        p_zi <= q_mi * sr - q_mr * si;
        if (p_v) begin
            zr[p_t] <= zr[p_t] + {{13{p_zr[34]}}, p_zr};
            zi[p_t] <= zi[p_t] + {{13{p_zi[34]}}, p_zi};
        end

        if (!rst_n) begin
            st <= T_IDLE;
            done <= 1'b0;
            phi_valid <= 1'b0;
            cv_in <= 1'b0;
            cr_in <= 1'b0;
        end else begin
            done <= 1'b0;
            phi_valid <= 1'b0;
            cv_in <= 1'b0;
            cr_in <= 1'b0;
            case (st)
                T_IDLE: if (go) begin
                    st <= T_Z;
                    f <= 8'd0;
                    kk <= 3'd0;
                    for (q = 0; q < 4; q = q + 1) begin
                        zr[q] <= 48'sd0;
                        zi[q] <= 48'sd0;
                        h_re[q] <= 16'sd0;
                        h_im[q] <= 16'sd0;
                    end
                end
                T_Z: begin
                    if (kk >= 3'd1 && kk <= np3) begin
                        h_re[kc[1:0]] <= hf_rdata[31:16];
                        h_im[kc[1:0]] <= hf_rdata[15:0];
                    end
                    if (kk == np3 + 3'd1) begin       // h_re[2] is 0 with two pilot symbols
                        sr <= {{2{h_re[0][15]}}, h_re[0]} + {{2{h_re[1][15]}}, h_re[1]} + {{2{h_re[2][15]}}, h_re[2]};
                        si <= {{2{h_im[0][15]}}, h_im[0]} + {{2{h_im[1][15]}}, h_im[1]} + {{2{h_im[2][15]}}, h_im[2]};
                    end
                    if (kk == {np3[1:0], 1'b1}) begin  // 2 np + 1
                        kk <= 3'd0;
                        f <= f + 8'd1;
                        if (f == nf - 8'd1)
                            st <= T_ZW;
                    end else
                        kk <= kk + 3'd1;
                end
                T_ZW: if (!q_v && !p_v)                     // the last product is accumulated
                    st <= T_MX0;
                T_MX0: begin
                    for (q = 0; q < 3; q = q + 1) begin
                        az[2 * q]     <= absz(zr[q], q < {29'd0, npsym});
                        az[2 * q + 1] <= absz(zi[q], q < {29'd0, npsym});
                    end
                    st <= T_MX1;
                end
                T_MX1: begin
                    mx <= (m01 > m23) ? ((m01 > m45) ? m01 : m45) : ((m23 > m45) ? m23 : m45);
                    st <= T_SH;
                end
                T_SH: begin
                    sh <= (bl > 7'd30) ? bl - 7'd30 : 7'd0;
                    cv_n <= 2'd0;
                    cv_got <= 2'd0;
                    st <= T_CV;
                end
                T_CV: begin
                    if (cv_n != npsym) begin
                        cv_in <= 1'b1;
                        cv_x <= zsr[31:0];
                        cv_y <= zsi[31:0];
                        cv_tag <= cv_n;
                        cv_n <= cv_n + 2'd1;
                    end
                    if (cv_ov) begin
                        mag[cv_otag] <= cv_mag;
                        ang[cv_otag] <= cv_ang;
                        cv_got <= cv_got + 2'd1;
                        if (cv_got == npsym - 2'd1) begin
                            st <= T_ND;
                            nt <= 2'd0;
                            nph <= 2'd0;
                            num <= 64'sd0;
                            den <= 64'sd0;
                        end
                    end
                end
                T_ND: begin                       // four clocks a pilot symbol
                    nph <= nph + 2'd1;
                    if (nph == 2'd0) begin
                        wtr <= wt;
                        svr <= svt;
                        angr <= ang[nt];
                    end else if (nph == 2'd1)
                        ws <= wtr * svr;
                    else if (nph == 2'd2) begin
                        pn <= ws * angr;
                        pd <= ws * svr;
                    end else begin
                        num <= num + {{19{pn[44]}}, pn};
                        den <= den + {{30{pd[33]}}, pd};
                        nt <= nt + 2'd1;
                        if (nt == npsym - 2'd1) begin
                            st <= T_PHIS;
                            ps <= 4'd0;
                        end
                    end
                end
                T_PHIS: begin                     // sv_s
                    svs_r <= svs;
                    st <= T_PHI0;
                end
                T_PHI0: begin                     // num sv_s
                    m1 <= $signed(num[55:0]) * svs_r;
                    st <= T_PHI1;
                end
                T_PHI1: begin                     // its two parts times the gain
                    m1lo <= m1[27:0] * gain;
                    m1hi <= $signed(m1[55:28]) * $signed({1'b0, gain});
                    st <= T_PHI2;
                end
                T_PHI2: begin                     // num sv_s gain
                    m2 <= {{18{m1hi[45]}}, m1hi} * 64'sd268435456 + $signed({19'd0, m1lo});
                    st <= T_PHI3;
                end
                T_PHI3: begin                     // 2 num sv_s gain + den 2^15
                    numer <= (m2 <<< 1) + (den <<< 15);
                    st <= T_PHI4;
                end
                T_PHI4: begin
                    dv_neg <= numer[63];
                    dv_n <= numer[63] ? -numer : numer;
                    dv_d <= den <<< 16;
                    dv_q <= 64'd0;
                    dv_r <= 65'd0;
                    dv_i <= 7'd0;
                    st <= T_DIV;
                end
                T_DIV: begin
                    if (dv_i == 7'd64) begin
                        phi[ps] <= phi_new;
                        phi_valid <= 1'b1;
                        phi_idx <= ps;
                        phi_val <= phi_new;
                        ps <= ps + 4'd1;
                        if (ps == num_sym - 4'd1) begin
                            st <= T_RA0;
                            ru <= 5'd0;
                        end else
                            st <= T_PHIS;
                    end else begin
                        dv_n <= {dv_n[62:0], 1'b0};
                        dv_q <= {dv_q[62:0], r_ge};
                        dv_r <= r_ge ? r_sh - {1'b0, dv_d} : r_sh;
                        dv_i <= dv_i + 7'd1;
                    end
                end
                T_RA0: begin                      // the unit: its base, kind and angle
                    nphi <= -phi[ru_s];
                    ru_base <= ru_h ? {2'd0, ru_tf} : ru_sf;
                    ru_hr <= ru_h;
                    st <= T_RA;
                end
                T_RA: begin                       // the angle into the CORDIC
                    cr_in <= 1'b1;
                    st <= T_RW;
                end
                T_RW: if (cr_ov) begin
                    cs_c <= cr_c;
                    cs_s <= cr_s;
                    rf <= 8'd0;
                    st <= T_RUN;
                end
                T_RUN: begin
                    rf <= rf + 8'd1;
                    if (rf == nf - 8'd1)
                        st <= T_RDRAIN;
                end
                T_RDRAIN: if (!r1_v && !r2_v && !r3_v) begin
                    if (ru == {1'b0, num_sym} + {3'd0, npsym} - 5'd1) begin
                        st <= T_IDLE;
                        done <= 1'b1;
                    end else begin
                        ru <= ru + 5'd1;
                        st <= T_RA0;
                    end
                end
                default: st <= T_IDLE;
            endcase
        end
    end
endmodule
