`timescale 1ns / 1ps
// 256-point fixed-point FFT/IFFT, bit exact against fx.Fft.runFixed
// (modem/src/fx.zig) and twin.fft (harness/twin.py).
//
// Radix-2 decimation in time, in place. Unitary: the even stages (0, 2,
// 4, 6) halve. Each butterfly output is (a 2^16 +- w b) rounded half to
// even by 16 or 17 bits and saturated to 16 bits; twiddles are Q16 in 18
// bits (fft256_twiddle). The inverse conjugates the twiddles.
//
// The data sit in two 128 x 32 simple-dual-port banks split by address
// parity: the two addresses of a butterfly differ in one bit, so a pair is
// always one word from each bank, and within a bank address a sits at
// a[7:1]. (One true-dual-port RAM of this shape is not inferred by Vivado.)
// The butterfly is a 6-clock pipeline: a pair is read every clock and its
// results written back 6 clocks later, through the banks' write ports; the
// pipeline drains between stages, which read what the stage before wrote.
// 8 (128 + 6) clocks of butterflies, about 1590 with the load and the unload
// (15.9 us at 100 MHz, inside the 28.8 us of an OFDM symbol). Four DSP48s.
//
// Interface: 256 input samples in natural order (in_ready high while
// loading, `inverse` sampled with the first one), then the butterflies,
// then 256 output samples in natural order with out_idx.
module fft256 (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               inverse,
    input  wire               in_valid,
    input  wire signed [15:0] in_re,
    input  wire signed [15:0] in_im,
    output wire               in_ready,
    output reg                out_valid,
    output wire signed [15:0] out_re,
    output wire signed [15:0] out_im,
    output reg         [7:0]  out_idx
);
    localparam LOAD = 2'd0, CALC = 2'd1, DRAIN = 2'd2, UNLOAD = 2'd3;

    reg        [1:0]  state;
    reg        [7:0]  cnt;
    reg        [2:0]  stage;
    reg        [6:0]  k;
    reg               inv;

    assign in_ready = (state == LOAD);

    // natural index -> bit-reversed address
    function [7:0] bitrev;
        input [7:0] x;
        integer b;
        begin
            for (b = 0; b < 8; b = b + 1)
                bitrev[b] = x[7 - b];
        end
    endfunction

    // butterfly addresses: group g = k >> stage, j = k mod 2^stage
    wire [7:0] half = 8'd1 << stage;
    wire [7:0] jj   = {1'b0, k} & (half - 8'd1);
    wire [7:0] i0   = (({1'b0, k} >> stage) << ({1'b0, stage} + 4'd1)) | jj;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [7:0] i1   = i0 | half;               // bit 0 is i0[0]: the bank parity decides
    /* verilator lint_on UNUSEDSIGNAL */
    wire [6:0] tw_j = jj[6:0] << (3'd7 - stage);
    wire       issue = (state == CALC);

    // ---- the banks
    reg  [31:0] bank0 [0:127];
    reg  [31:0] bank1 [0:127];
    reg  [6:0]  ra0, ra1, wa0, wa1;
    reg         we0, we1;
    reg  [31:0] wd0, wd1;
    reg  [31:0] q0, q1;
    always @(posedge clk) begin
        if (we0)
            bank0[wa0] <= wd0;
        q0 <= bank0[ra0];
    end
    always @(posedge clk) begin
        if (we1)
            bank1[wa1] <= wd1;
        q1 <= bank1[ra1];
    end

    // ---- the pipeline. p1: the pair arrives; p2: products; p3: sums;
    // p4: quotient and rounding bit; p5: rounded, saturated; then written.
    wire       p0_par = ^i0;                   // the bank of i0 (i1 is in the other)
    reg        p1_v, p1_par, p1_sh;
    reg  [6:0] p1_a0, p1_a1;
    reg  [6:0] p1_tw;
    reg        p2_v, p2_par, p2_sh;
    reg  [6:0] p2_a0, p2_a1;
    reg signed [15:0] p2_ar, p2_ai;
    reg signed [35:0] p2_m0, p2_m1, p2_m2, p2_m3;
    reg        p3_v, p3_par, p3_sh;
    reg  [6:0] p3_a0, p3_a1;
    reg signed [39:0] p3_x [0:3];              // top re, top im, bottom re, bottom im
    reg        p4_v, p4_par;
    reg  [6:0] p4_a0, p4_a1;
    reg signed [23:0] p4_q [0:3];
    reg        [3:0]  p4_up;
    reg        p5_v, p5_par;
    reg  [6:0] p5_a0, p5_a1;
    reg  [31:0] p5_top, p5_bot;

    wire [31:0] da = p1_par ? q1 : q0;         // the word at i0
    wire [31:0] db = p1_par ? q0 : q1;         // the word at i1
    wire signed [15:0] b_re = db[31:16], b_im = db[15:0];
    wire signed [17:0] wr, wi_rom;
    fft256_twiddle rom (.j(p1_tw), .wr(wr), .wi(wi_rom));
    wire signed [17:0] wi = inv ? -wi_rom : wi_rom;

    wire signed [39:0] tr = {{4{p2_m0[35]}}, p2_m0} - {{4{p2_m1[35]}}, p2_m1};
    wire signed [39:0] ti = {{4{p2_m2[35]}}, p2_m2} + {{4{p2_m3[35]}}, p2_m3};
    wire signed [39:0] ar = {{8{p2_ar[15]}}, p2_ar, 16'd0};
    wire signed [39:0] ai = {{8{p2_ai[15]}}, p2_ai, 16'd0};

    // x >> s (s = 16 or 17) and whether rounding half to even goes up
    function [24:0] div_rnd;                   // {up, quotient[23:0]}
        input signed [39:0] x;
        input               s17;
        reg signed [39:0] q, r, h;
        begin
            q = s17 ? (x >>> 17) : (x >>> 16);
            r = x - (s17 ? (q <<< 17) : (q <<< 16));
            h = s17 ? 40'sd65536 : 40'sd32768;
            div_rnd = {(r > h || (r == h && q[0])), q[23:0]};
        end
    endfunction
    function [15:0] sat16;
        input signed [23:0] q;
        input               up;
        reg signed [24:0] v;
        begin
            v = {q[23], q} + {24'd0, up};
            if (v > 25'sd32767)
                sat16 = 16'h7fff;
            else if (v < -25'sd32768)
                sat16 = 16'h8000;
            else
                sat16 = v[15:0];
        end
    endfunction

    reg [24:0] dr [0:3];
    integer n;
    always @(*) begin
        for (n = 0; n < 4; n = n + 1)
            dr[n] = div_rnd(p3_x[n], p3_sh);
    end

    // ---- the banks' ports, from the state
    wire [7:0] lda = bitrev(cnt);
    always @(*) begin
        ra0 = 7'd0;
        ra1 = 7'd0;
        we0 = 1'b0;
        we1 = 1'b0;
        wa0 = p5_par ? p5_a1 : p5_a0;
        wa1 = p5_par ? p5_a0 : p5_a1;
        wd0 = p5_par ? p5_bot : p5_top;
        wd1 = p5_par ? p5_top : p5_bot;
        case (state)
            LOAD: begin
                wa0 = lda[7:1];
                wa1 = lda[7:1];
                wd0 = {in_re, in_im};
                wd1 = {in_re, in_im};
                we0 = in_valid && !(^lda);
                we1 = in_valid && (^lda);
            end
            UNLOAD: begin
                ra0 = cnt[7:1];
                ra1 = cnt[7:1];
            end
            default: begin                     // CALC, DRAIN
                ra0 = p0_par ? i1[7:1] : i0[7:1];
                ra1 = p0_par ? i0[7:1] : i1[7:1];
                we0 = p5_v;
                we1 = p5_v;
            end
        endcase
    end

    // the unload reads bank parity(cnt); the word lands a clock later
    reg out_par;
    wire [31:0] dout = out_par ? q1 : q0;
    assign out_re = dout[31:16];
    assign out_im = dout[15:0];

    always @(posedge clk) begin
        // pipeline
        p1_v   <= issue && rst_n;
        p1_par <= p0_par;
        p1_sh  <= ~stage[0];                   // even stages: 17 bits
        p1_a0  <= i0[7:1];
        p1_a1  <= i1[7:1];
        p1_tw  <= tw_j;
        p2_v   <= p1_v && rst_n;
        p2_par <= p1_par;
        p2_sh  <= p1_sh;
        p2_a0  <= p1_a0;
        p2_a1  <= p1_a1;
        p2_ar  <= da[31:16];
        p2_ai  <= da[15:0];
        p2_m0  <= wr * b_re;
        p2_m1  <= wi * b_im;
        p2_m2  <= wr * b_im;
        p2_m3  <= wi * b_re;
        p3_v   <= p2_v && rst_n;
        p3_par <= p2_par;
        p3_sh  <= p2_sh;
        p3_a0  <= p2_a0;
        p3_a1  <= p2_a1;
        p3_x[0] <= ar + tr;
        p3_x[1] <= ai + ti;
        p3_x[2] <= ar - tr;
        p3_x[3] <= ai - ti;
        p4_v   <= p3_v && rst_n;
        p4_par <= p3_par;
        p4_a0  <= p3_a0;
        p4_a1  <= p3_a1;
        for (n = 0; n < 4; n = n + 1) begin
            p4_q[n]  <= dr[n][23:0];
            p4_up[n] <= dr[n][24];
        end
        p5_v   <= p4_v && rst_n;
        p5_par <= p4_par;
        p5_a0  <= p4_a0;
        p5_a1  <= p4_a1;
        p5_top <= {sat16(p4_q[0], p4_up[0]), sat16(p4_q[1], p4_up[1])};
        p5_bot <= {sat16(p4_q[2], p4_up[2]), sat16(p4_q[3], p4_up[3])};
        out_par <= ^cnt;

        if (!rst_n) begin
            state     <= LOAD;
            cnt       <= 8'd0;
            stage     <= 3'd0;
            k         <= 7'd0;
            inv       <= 1'b0;
            out_valid <= 1'b0;
            out_idx   <= 8'd0;
        end else begin
            out_valid <= 1'b0;
            case (state)
                LOAD: if (in_valid) begin
                    if (cnt == 8'd0)
                        inv <= inverse;
                    cnt <= cnt + 8'd1;
                    if (cnt == 8'd255) begin
                        state <= CALC;
                        stage <= 3'd0;
                        k     <= 7'd0;
                    end
                end
                CALC: begin
                    k <= k + 7'd1;
                    if (k == 7'd127)
                        state <= DRAIN;
                end
                DRAIN: if (!p1_v && !p2_v && !p3_v && !p4_v && !p5_v) begin
                    // the stage's last write landed at the end of the previous clock
                    stage <= stage + 3'd1;
                    state <= (stage == 3'd7) ? UNLOAD : CALC;
                end
                default: begin                     // UNLOAD: the read lands next clock
                    out_valid <= 1'b1;
                    out_idx   <= cnt;
                    cnt       <= cnt + 8'd1;
                    if (cnt == 8'd255)
                        state <= LOAD;
                end
            endcase
        end
    end
endmodule
