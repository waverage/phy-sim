`timescale 1ns / 1ps
// AWGN channel emulator of the board loopback, bit exact against
// harness/chanemu.py (the formulas are there). `start` loads the frame's
// seed and restarts the sample counter; base = h(seed) is ready 8 clocks
// later (ready). Then one TX sample per clock at most, any spacing, and
// the ADC sample comes out 28 clocks after its input, in order.
/* verilator lint_off DECLFILENAME */

// lowbias32, pipelined: 7 clocks (each 32-bit multiply has two registers,
// which Vivado moves into the DSP48s).
module lowbias32 (
    input  wire        clk,
    input  wire [31:0] in,
    output reg  [31:0] out
);
    reg  [31:0] x1, m1a, m1, x2, m2a, m2;
    always @(posedge clk) begin
        x1  <= in ^ (in >> 16);
        m1a <= x1 * 32'h7feb352d;
        m1  <= m1a;
        x2  <= m1 ^ (m1 >> 15);
        m2a <= x2 * 32'h846ca68b;
        m2  <= m2a;
        out <= m2 ^ (m2 >> 16);
    end
endmodule

module chan_emu (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire        [31:0] seed,
    input  wire        [31:0] gx,                // g 2^28 / tx_scale
    input  wire        [31:0] gn,                // g sqrt(N0 / 2) 2^8
    output wire               ready,
    input  wire               in_valid,
    input  wire signed [15:0] in_re,
    input  wire signed [15:0] in_im,
    output reg                out_valid,
    output reg  signed [15:0] out_re,
    output reg  signed [15:0] out_im
);
    // ---- base = h(seed)
    reg  [3:0]  bcnt;
    reg  [31:0] base;
    wire [31:0] hb;
    lowbias32 hbase (.clk(clk), .in(seed), .out(hb));
    assign ready = (bcnt == 4'd0);
    always @(posedge clk) begin
        if (!rst_n)
            bcnt <= 4'd0;
        else if (start)
            bcnt <= 4'd8;
        else if (bcnt != 4'd0) begin
            bcnt <= bcnt - 4'd1;
            if (bcnt == 4'd1)
                base <= hb;
        end
    end

    // ---- the two uniforms of sample i
    reg  [15:0] i;
    always @(posedge clk) begin
        if (start)
            i <= 16'd0;
        else if (in_valid)
            i <= i + 16'd1;
    end
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] u1, u2;                       // u2: the top 16 bits are the angle
    /* verilator lint_on UNUSEDSIGNAL */
    lowbias32 h1 (.clk(clk), .in(base ^ {15'd0, i, 1'b0}), .out(u1));
    lowbias32 h2 (.clk(clk), .in(base ^ {15'd0, i, 1'b1}), .out(u2));
    reg  [6:0]  hv;                           // the sample beside its hashes
    reg  [31:0] hx [0:6];
    integer d;
    always @(posedge clk) begin
        hv    <= rst_n ? {hv[5:0], in_valid} : 7'd0;
        hx[0] <= {in_re, in_im};
        for (d = 1; d < 7; d = d + 1)
            hx[d] <= hx[d - 1];
    end

    // ---- the radius: R[min(clz u1, 23), 8 bits after the leading one]
    reg  [5:0] lz6;
    integer b;
    always @(*) begin
        lz6 = 6'd32;
        for (b = 0; b < 32; b = b + 1)
            if (u1[b])
                lz6 = 6'd31 - b[5:0];
    end
    wire [4:0] lz = (lz6 > 6'd23) ? 5'd23 : lz6[4:0];
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] ush = u1 << lz;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [13:0] r;
    chan_emu_rtab rtab (.clk(clk), .addr({lz, ush[30:23]}), .r(r));
    reg         tv;
    reg  [15:0] tang;
    reg  [31:0] tx;
    always @(posedge clk) begin
        tv   <= hv[6] & rst_n;
        tang <= u2[31:16];
        tx   <= hx[6];
    end

    // ---- the angle: cossin(u2 >> 16), the radius and the sample as its tag
    wire               cv;
    wire signed [17:0] c, s;
    wire        [45:0] ctag;
    cordic_rot #(.TAGW(46)) rot (
        .clk(clk), .rst_n(rst_n), .in_valid(tv), .in_ang(tang), .in_tag({r, tx}),
        .out_valid(cv), .out_cos(c), .out_sin(s), .out_tag(ctag));

    // ---- y = sat12(rshr(gx x + gn (r cs >> 6), 28))
    wire signed [14:0] rr = {1'b0, ctag[45:32]};
    wire signed [15:0] xr = ctag[31:16], xi = ctag[15:0];
    // three registered stages: the noise, the two products, their sum
    wire signed [32:0] nr_full = rr * c;
    wire signed [32:0] ni_full = rr * s;
    reg               n_v, m_v, pv;
    reg signed [26:0] nr, ni;                  // (r cs) >> 6
    reg signed [15:0] nxr, nxi;
    reg signed [48:0] sr, si;                  // gx x
    reg signed [59:0] er, ei;                  // gn n
    reg signed [63:0] ar, ai;
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [32:0] nr_sh = nr_full >>> 6;
    wire signed [32:0] ni_sh = ni_full >>> 6;
    /* verilator lint_on UNUSEDSIGNAL */
    always @(posedge clk) begin
        n_v <= cv & rst_n;
        nr  <= nr_sh[26:0];
        ni  <= ni_sh[26:0];
        nxr <= xr;
        nxi <= xi;
        m_v <= n_v & rst_n;
        sr  <= $signed({1'b0, gx}) * nxr;
        si  <= $signed({1'b0, gx}) * nxi;
        er  <= $signed({1'b0, gn}) * nr;
        ei  <= $signed({1'b0, gn}) * ni;
        pv  <= m_v & rst_n;
        ar  <= {{15{sr[48]}}, sr} + {{4{er[59]}}, er};
        ai  <= {{15{si[48]}}, si} + {{4{ei[59]}}, ei};
    end

    function signed [15:0] adc;
        input signed [63:0] v;
        reg signed [63:0] q;
        begin
            q = (v + 64'sd134217728) >>> 28;
            if (q > 64'sd2047)
                adc = 16'sd2047;
            else if (q < -64'sd2048)
                adc = -16'sd2048;
            else
                adc = q[15:0];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n)
            out_valid <= 1'b0;
        else
            out_valid <= pv;
        out_re <= adc(ar);
        out_im <= adc(ai);
    end
endmodule
