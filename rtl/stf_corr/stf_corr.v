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
// DW is the width of the sample count and of d.
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

    // a conj(b) and |a|^2
    // (products of 12-bit samples, computed at the accumulators' widths)
    wire signed [33:0] add_re = r0re * rpre + r0im * rpim;
    wire signed [33:0] add_im = r0im * rpre - r0re * rpim;
    wire signed [35:0] add_e  = r0re * r0re + r0im * r0im + rpre * rpre + rpim * rpim;
    wire signed [33:0] sub_re = rwre * rdre + rwim * rdim;
    wire signed [33:0] sub_im = rwim * rdre - rwre * rdim;
    wire signed [35:0] sub_e  = rwre * rwre + rwim * rwim + rdre * rdre + rdim * rdim;

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
            if (in_valid) begin
                m <= m + 1'b1;
                if (m >= D16) begin
                    pr <= pr + add_re - sub_re;
                    pi <= pi + add_im - sub_im;
                    e  <= e + add_e - sub_e;
                end else if (m >= PER16) begin
                    pr <= pr + add_re;
                    pi <= pi + add_im;
                    e  <= e + add_e;
                end
                if (m >= WP1) begin
                    sv <= 1'b1;
                    sd <= m - WP1;
                end
                slast <= in_last;
            end
        end
    end

    // ---- detection and the peak, one d per clock
    // the shifted sums fit 26 and 28 bits for 12-bit samples
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [33:0] psr_w = pr >>> MS;
    wire signed [33:0] psi_w = pi >>> MS;
    wire signed [35:0] es_w  = e >>> MS;
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [25:0] psr = psr_w[25:0];
    wire signed [25:0] psi = psi_w[25:0];
    wire        [27:0] es  = es_w[27:0];
    wire        [63:0] mag = psr * psr + psi * psi;
    wire        [73:0] lhs = {mag, 10'd0};                          // 4 |P|^2 256
    wire        [73:0] rhs = threshold_q8 * es * es;

    reg               found, fin;
    reg        [DW-1:0] d0;
    reg        [63:0] best;
    localparam [DW-1:0] PREW = PRE;

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
            trace_valid <= sv;
            trace_d <= sd;
            trace_pr <= psr;
            trace_pi <= psi;
            trace_e <= es;
            done <= 1'b0;
            if (sv && !fin) begin
                if (found || lhs > rhs) begin
                    if (!found) begin
                        found <= 1'b1;
                        d0 <= sd;
                    end
                    if (!found || mag > best) begin
                        best <= mag;
                        peak <= sd;
                        peak_pr <= psr;
                        peak_pi <= psi;
                        peak_e <= es;
                        peak_pr_full <= pr;
                        peak_pi_full <= pi;
                    end
                    if ((found && sd == d0 + PREW) || slast) begin
                        done <= 1'b1;
                        fin <= 1'b1;
                        detected <= 1'b1;
                    end
                end else if (slast) begin
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
