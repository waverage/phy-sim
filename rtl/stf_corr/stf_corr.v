`timescale 1ns / 1ps
// The preamble correlator: steps 1 and 2 of fxsync.Sync.runOne
// (modem/src/fxsync.zig), bit exact against it and TwinSync (harness/twin.py).
//
// For every d, over a window of W samples:
//   P(d) = sum_{n=d}^{d+W-1} r[n + PER] conj(r[n])
//   E(d) = sum_{n=d}^{d+W-1} |r[n]|^2 + |r[n + PER]|^2
// as running sums on the sample stream: sample m = n + PER adds
// r[m] conj(r[m - PER]) and takes away the product that left the window,
// r[m - W] conj(r[m - W - PER]), from delay lines of W + PER samples in
// all. d = m - PER - W + 1 once m reaches PER + W - 1.
//
// Detection on the values shifted down by MS: the first d with
// 4 |Ps|^2 256 > threshold_q8 Es^2; then the largest |Ps|^2 (the first one
// if tied) over the next PRE samples or to the end of the buffer (`last` on
// its final sample) is the peak. trace_* gives Ps, Es for every d (the
// golden vectors' trace_p, trace_e).
//
// `restart` (rtl/sync, on a stream) starts a new buffer at the next sample:
// the sums, the sample count and the search start again, as after reset.
// DW is the width of the sample count and of d. Pipelined for 100 MHz:
// at most one sample every other clock.
/* verilator lint_off DECLFILENAME */
module stf_corr #(
    parameter PER = 32,
    parameter W = 256,
    parameter PRE = 288,
    parameter MS = 8,
    parameter DW = 16
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               restart,
    input  wire        [8:0]  threshold_q8,
    input  wire               in_valid,
    input  wire signed [11:0] in_re,
    input  wire signed [11:0] in_im,
    input  wire               in_last,
    // every d
    output reg                trace_valid,
    output reg         [DW-1:0] trace_d,
    output reg  signed [25:0] trace_pr,
    output reg  signed [25:0] trace_pi,
    output reg         [27:0] trace_e,
    // once per buffer
    output reg                done,
    output reg                detected,
    output reg         [DW-1:0] peak,
    output reg  signed [25:0] peak_pr,     // shifted by MS
    output reg  signed [25:0] peak_pi,
    output reg         [27:0] peak_e,
    output reg  signed [33:0] peak_pr_full, // unshifted, for the angle
    output reg  signed [33:0] peak_pi_full
);
    localparam D = W + PER;
    localparam [DW-1:0] PER16 = PER, D16 = D, WP1 = PER + W - 1;

    reg        [DW-1:0] m;                  // samples so far

    // Delay lines (shift registers, SRLs in an FPGA), shifted per sample:
    // at sample m, d1 holds r[m-1] .. r[m-PER], d2 r[m-PER-1] .. r[m-W] and
    // d3 r[m-W-1] .. r[m-W-PER]. They are not reset: a tap is only used
    // once the current buffer has filled it.
    reg [23:0] d1 [0:PER-1];
    reg [23:0] d2 [0:W-PER-1];
    reg [23:0] d3 [0:PER-1];
    integer i;
    always @(posedge clk) begin
        if (in_valid) begin
            d1[0] <= {in_re, in_im};
            d2[0] <= d1[PER-1];
            d3[0] <= d2[W-PER-1];
            for (i = 1; i < PER; i = i + 1) begin
                d1[i] <= d1[i-1];
                d3[i] <= d3[i-1];
            end
            for (i = 1; i < W - PER; i = i + 1)
                d2[i] <= d2[i-1];
        end
    end

    wire signed [11:0] r0re = in_re, r0im = in_im;                              // r[m]
    wire signed [11:0] rpre = d1[PER-1][23:12], rpim = d1[PER-1][11:0];         // r[m - PER]
    wire signed [11:0] rwre = d2[W-PER-1][23:12], rwim = d2[W-PER-1][11:0];     // r[m - W]
    wire signed [11:0] rdre = d3[PER-1][23:12], rdim = d3[PER-1][11:0];         // r[m - W - PER]

    // a conj(b) and |a|^2 of 12-bit samples, pipelined for 100 MHz: A1 the
    // products, A2 their sums (zero where the window is not full yet), then
    // the running sums. One update per sample, in order, two clocks late.
    reg               a1_v, a1_add, a1_sub, a1_out, a1_last;
    reg      [DW-1:0] a1_d;
    reg signed [23:0] p_rr, p_ii, p_ir, p_ri, q_rr, q_ii, q_ir, q_ri;          // add and sub cross products
    reg signed [23:0] e_0r, e_0i, e_pr, e_pi, e_wr, e_wi, e_dr, e_di;          // the squares
    reg               a2_v, a2_out, a2_last;
    reg      [DW-1:0] a2_d;
    reg signed [33:0] add_re, add_im, sub_re, sub_im;
    reg signed [35:0] add_e, sub_e;
    always @(posedge clk) begin
        a1_v    <= in_valid && rst_n && !restart;
        a1_add  <= (m >= PER16);
        a1_sub  <= (m >= D16);
        a1_out  <= (m >= WP1);
        a1_d    <= m - WP1;
        a1_last <= in_last;
        p_rr <= r0re * rpre;
        p_ii <= r0im * rpim;
        p_ir <= r0im * rpre;
        p_ri <= r0re * rpim;
        q_rr <= rwre * rdre;
        q_ii <= rwim * rdim;
        q_ir <= rwim * rdre;
        q_ri <= rwre * rdim;
        e_0r <= r0re * r0re;
        e_0i <= r0im * r0im;
        e_pr <= rpre * rpre;
        e_pi <= rpim * rpim;
        e_wr <= rwre * rwre;
        e_wi <= rwim * rwim;
        e_dr <= rdre * rdre;
        e_di <= rdim * rdim;
        a2_v    <= a1_v && rst_n && !restart;
        a2_out  <= a1_out;
        a2_d    <= a1_d;
        a2_last <= a1_last;
        add_re <= a1_add ? {{10{p_rr[23]}}, p_rr} + {{10{p_ii[23]}}, p_ii} : 34'sd0;
        add_im <= a1_add ? {{10{p_ir[23]}}, p_ir} - {{10{p_ri[23]}}, p_ri} : 34'sd0;
        add_e  <= a1_add ? {{12{e_0r[23]}}, e_0r} + {{12{e_0i[23]}}, e_0i} + {{12{e_pr[23]}}, e_pr} + {{12{e_pi[23]}}, e_pi} : 36'sd0;
        sub_re <= a1_sub ? {{10{q_rr[23]}}, q_rr} + {{10{q_ii[23]}}, q_ii} : 34'sd0;
        sub_im <= a1_sub ? {{10{q_ir[23]}}, q_ir} - {{10{q_ri[23]}}, q_ri} : 34'sd0;
        sub_e  <= a1_sub ? {{12{e_wr[23]}}, e_wr} + {{12{e_wi[23]}}, e_wi} + {{12{e_dr[23]}}, e_dr} + {{12{e_di[23]}}, e_di} : 36'sd0;
    end

    reg signed [33:0] pr, pi;               // the running sums
    reg signed [35:0] e;
    reg               sv, slast;            // the sums hold d = m - PER - W + 1
    reg        [DW-1:0] sd;

    always @(posedge clk) begin
        if (!rst_n || restart) begin
            m <= {DW{1'b0}};
            pr <= 34'sd0;
            pi <= 34'sd0;
            e <= 36'sd0;
            sv <= 1'b0;
            slast <= 1'b0;
            sd <= {DW{1'b0}};
        end else begin
            sv <= 1'b0;
            slast <= 1'b0;
            if (in_valid)
                m <= m + 1'b1;
            if (a2_v) begin
                pr <= pr + add_re - sub_re;
                pi <= pi + add_im - sub_im;
                e  <= e + add_e - sub_e;
                if (a2_out) begin
                    sv <= 1'b1;
                    sd <= a2_d;
                end
                slast <= a2_last;
            end
        end
    end

    // ---- detection and the peak, one d per clock, pipelined for 100 MHz:
    // M1 the shifted sums, M2 the squares, M3 |P|^2 and threshold E^2,
    // M4 the comparison, then the decision
    // the shifted sums fit 26 and 28 bits for 12-bit samples
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [33:0] psr_w = pr >>> MS;
    wire signed [33:0] psi_w = pi >>> MS;
    wire signed [35:0] es_w  = e >>> MS;
    /* verilator lint_on UNUSEDSIGNAL */
    reg                m1_v, m2_v, m3_v, m4_v;
    reg                m1_last, m2_last, m3_last, m4_last;
    reg       [DW-1:0] m1_d, m2_d, m3_d, m4_d;
    reg  signed [25:0] m1_pr, m1_pi, m2_pr, m2_pi, m3_pr, m3_pi, m4_pr, m4_pi;
    reg         [27:0] m1_e, m2_e, m3_e, m4_e;
    reg  signed [33:0] m1_fr, m1_fi, m2_fr, m2_fi, m3_fr, m3_fi, m4_fr, m4_fi;
    reg         [51:0] m2_sr, m2_si;
    reg         [55:0] m2_e2;
    reg         [63:0] m3_mag, m4_mag;
    reg         [73:0] m3_rhs;
    reg                m4_over;
    always @(posedge clk) begin
        m1_v    <= sv && rst_n && !restart;
        m1_last <= slast;
        m1_d    <= sd;
        m1_pr   <= psr_w[25:0];
        m1_pi   <= psi_w[25:0];
        m1_e    <= es_w[27:0];
        m1_fr   <= pr;
        m1_fi   <= pi;
        m2_v    <= m1_v && rst_n && !restart;
        m2_last <= m1_last;
        m2_d    <= m1_d;
        m2_pr   <= m1_pr;
        m2_pi   <= m1_pi;
        m2_e    <= m1_e;
        m2_fr   <= m1_fr;
        m2_fi   <= m1_fi;
        m2_sr   <= m1_pr * m1_pr;
        m2_si   <= m1_pi * m1_pi;
        m2_e2   <= m1_e * m1_e;
        m3_v    <= m2_v && rst_n && !restart;
        m3_last <= m2_last;
        m3_d    <= m2_d;
        m3_pr   <= m2_pr;
        m3_pi   <= m2_pi;
        m3_e    <= m2_e;
        m3_fr   <= m2_fr;
        m3_fi   <= m2_fi;
        m3_mag  <= {12'd0, m2_sr} + {12'd0, m2_si};
        m3_rhs  <= threshold_q8 * m2_e2;
        m4_v    <= m3_v && rst_n && !restart;
        m4_last <= m3_last;
        m4_d    <= m3_d;
        m4_pr   <= m3_pr;
        m4_pi   <= m3_pi;
        m4_e    <= m3_e;
        m4_fr   <= m3_fr;
        m4_fi   <= m3_fi;
        m4_mag  <= m3_mag;
        m4_over <= {m3_mag, 10'd0} > m3_rhs;                      // 4 |P|^2 256 > threshold E^2
    end

    reg               found, fin;
    reg        [DW-1:0] d0;
    reg        [63:0] best;
    localparam [DW-1:0] PREW = PRE;

    // M5: the comparisons against the decision's state registered. With one
    // sample every other clock at most (rtl/sync), the state they read
    // already holds the previous d's decision.
    reg                m5_v, m5_last, m5_over, m5_gt, m5_end;
    reg       [DW-1:0] m5_d;
    reg  signed [25:0] m5_pr, m5_pi;
    reg         [27:0] m5_e;
    reg  signed [33:0] m5_fr, m5_fi;
    reg         [63:0] m5_mag;
    always @(posedge clk) begin
        m5_v    <= m4_v && rst_n && !restart;
        m5_last <= m4_last;
        m5_over <= m4_over;
        m5_gt   <= m4_mag > best;
        m5_end  <= m4_d == d0 + PREW;
        m5_d    <= m4_d;
        m5_pr   <= m4_pr;
        m5_pi   <= m4_pi;
        m5_e    <= m4_e;
        m5_fr   <= m4_fr;
        m5_fi   <= m4_fi;
        m5_mag  <= m4_mag;
    end

    always @(posedge clk) begin
        if (!rst_n || restart) begin
            trace_valid <= 1'b0;
            done <= 1'b0;
            fin <= 1'b0;
            detected <= 1'b0;
            found <= 1'b0;
            d0 <= {DW{1'b0}};
            best <= 64'd0;
            peak <= {DW{1'b0}};
            peak_pr <= 26'sd0;
            peak_pi <= 26'sd0;
            peak_e <= 28'd0;
            peak_pr_full <= 34'sd0;
            peak_pi_full <= 34'sd0;
        end else begin
            trace_valid <= m1_v;
            trace_d <= m1_d;
            trace_pr <= m1_pr;
            trace_pi <= m1_pi;
            trace_e <= m1_e;
            done <= 1'b0;
            if (m5_v && !fin) begin
                if (found || m5_over) begin
                    if (!found) begin
                        found <= 1'b1;
                        d0 <= m5_d;
                    end
                    if (!found || m5_gt) begin
                        best <= m5_mag;
                        peak <= m5_d;
                        peak_pr <= m5_pr;
                        peak_pi <= m5_pi;
                        peak_e <= m5_e;
                        peak_pr_full <= m5_fr;
                        peak_pi_full <= m5_fi;
                    end
                    if ((found && m5_end) || m5_last) begin
                        done <= 1'b1;
                        fin <= 1'b1;
                        detected <= 1'b1;
                    end
                end else if (m5_last) begin
                    done <= 1'b1;
                    fin <= 1'b1;
                    detected <= 1'b0;
                end
            end
        end
    end
endmodule


// Test top and the coarse stage of the synchroniser: the correlator, then
// the angle of P at the peak (angle30) and the NCO increment per sample,
// angle * 2^32 / (65536 PER) = angle * 2^11 for PER = 32.
module stf_detect (
    input  wire               clk,
    input  wire               rst_n,
    input  wire        [8:0]  threshold_q8,
    input  wire               in_valid,
    input  wire signed [11:0] in_re,
    input  wire signed [11:0] in_im,
    input  wire               in_last,
    output wire               trace_valid,
    output wire        [15:0] trace_d,
    output wire signed [25:0] trace_pr,
    output wire signed [25:0] trace_pi,
    output wire        [27:0] trace_e,
    output wire               done,
    output wire               detected,
    output wire        [15:0] peak,
    output wire signed [25:0] peak_pr,
    output wire signed [25:0] peak_pi,
    output wire        [27:0] peak_e,
    output wire               ang_valid,
    output wire signed [17:0] ang_coarse,
    output wire signed [31:0] inc_coarse
);
    wire signed [33:0] pr_full, pi_full;
    stf_corr core (
        .clk(clk), .rst_n(rst_n), .restart(1'b0), .threshold_q8(threshold_q8), .in_valid(in_valid), .in_re(in_re), .in_im(in_im),
        .in_last(in_last), .trace_valid(trace_valid), .trace_d(trace_d), .trace_pr(trace_pr), .trace_pi(trace_pi),
        .trace_e(trace_e), .done(done), .detected(detected), .peak(peak), .peak_pr(peak_pr), .peak_pi(peak_pi),
        .peak_e(peak_e), .peak_pr_full(pr_full), .peak_pi_full(pi_full));
    /* verilator lint_off UNUSEDSIGNAL */
    wire tag_unused;
    /* verilator lint_on UNUSEDSIGNAL */
    angle30 #(.W(48), .TAGW(1)) a30 (
        .clk(clk), .rst_n(rst_n), .in_valid(done & detected),
        .in_re({{14{pr_full[33]}}, pr_full}), .in_im({{14{pi_full[33]}}, pi_full}), .in_tag(1'b0),
        .out_valid(ang_valid), .out_ang(ang_coarse), .out_tag(tag_unused));
    assign inc_coarse = {{3{ang_coarse[17]}}, ang_coarse, 11'd0};
endmodule
