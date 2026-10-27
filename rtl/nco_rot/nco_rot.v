`timescale 1ns / 1ps
// The derotator: a 32-bit NCO, the CORDIC and rot16 (fxsync.Sync.nco and
// fx.rot16), bit exact against them and TwinSync.align (harness/twin.py).
//
// Sample i of a run gets the phase phase0 - i inc (mod 2^32); the top 16
// bits go to cordic_rot, and the sample is multiplied by 2^15 (cos, sin),
// rounded half up by 15 bits and saturated to 16. With
// phase0 = -n0 inc mod 2^32 (the control processor's job), sample i is
// rotated by -(n0 + i) inc: the synchroniser's alignment of a frame that
// starts at sample n0, and its derotation of the first pilot symbol.
// `start` (with the first sample of a run) loads phase0. Latency 19.
module nco_rot (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire        [31:0] phase0,
    input  wire        [31:0] inc,
    input  wire               in_valid,
    input  wire signed [15:0] in_re,
    input  wire signed [15:0] in_im,
    output reg                out_valid,
    output reg  signed [15:0] out_re,
    output reg  signed [15:0] out_im
);
    reg [31:0] acc;
    wire [31:0] ph = start ? phase0 : acc;

    always @(posedge clk) begin
        if (!rst_n)
            acc <= 32'd0;
        else if (in_valid)
            acc <= ph - inc;
    end

    wire               cv;
    wire signed [17:0] c, sn;
    wire        [31:0] tag;
    cordic_rot #(.TAGW(32)) rot (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ang(ph[31:16]), .in_tag({in_re, in_im}),
        .out_valid(cv), .out_cos(c), .out_sin(sn), .out_tag(tag));

    wire signed [15:0] vr = tag[31:16];
    wire signed [15:0] vi = tag[15:0];
    // the four products registered, then their sums, rounding and saturation
    reg               pv;
    reg signed [33:0] p_rc, p_is, p_rs, p_ic;
    always @(posedge clk) begin
        pv   <= cv && rst_n;
        p_rc <= vr * c;
        p_is <= vi * sn;
        p_rs <= vr * sn;
        p_ic <= vi * c;
    end
    wire signed [35:0] re = {{2{p_rc[33]}}, p_rc} - {{2{p_is[33]}}, p_is};
    wire signed [35:0] im = {{2{p_rs[33]}}, p_rs} + {{2{p_ic[33]}}, p_ic};

    function signed [15:0] rnd_sat;          // rshr by 15 (half up), sat to 16 bits
        input signed [35:0] x;
        reg signed [35:0] q;
        begin
            q = (x + 36'sd16384) >>> 15;
            if (q > 36'sd32767)
                rnd_sat = 16'sh7fff;
            else if (q < -36'sd32768)
                rnd_sat = 16'sh8000;
            else
                rnd_sat = q[15:0];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_re <= 16'sd0;
            out_im <= 16'sd0;
        end else begin
            out_valid <= pv;
            out_re <= rnd_sat(re);
            out_im <= rnd_sat(im);
        end
    end
endmodule
