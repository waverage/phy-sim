`timescale 1ns / 1ps
// CORDIC, bit exact against fx.vector / fx.cossin (modem/src/fx.zig) and
// twin.cordic_vector / twin.cossin (harness/twin.py). Angles are binary,
// 65536 per turn; 14 iterations with atan(2^-i) rounded to that unit.
// Both cores are fully pipelined: one input per clock, valid and a tag
// travel with the data; reset clears the valid bits.
/* verilator lint_off DECLFILENAME */

// Vectoring: (x, y) -> magnitude times K = 1.6468 and the angle in
// (-32768, 32768]. Latency 16 clocks.
module cordic_vec #(
    parameter W = 32,          // input width, |x|, |y| < 2^(W-2)
    parameter TAGW = 1
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire signed [W-1:0]  in_x,
    input  wire signed [W-1:0]  in_y,
    input  wire     [TAGW-1:0]  in_tag,
    output reg                  out_valid,
    output reg  signed [W+1:0]  out_mag,
    output reg  signed [17:0]   out_ang,
    output reg      [TAGW-1:0]  out_tag
);
    localparam N = 14;
    localparam XW = W + 2;     // the gain K and the pre-rotation fit in two more bits

    reg signed [XW-1:0] x [0:N];
    reg signed [XW-1:0] y [0:N];
    reg signed [17:0]   a [0:N];
    reg                 v [0:N];
    reg     [TAGW-1:0]  t [0:N];

    function signed [17:0] atan;
        input integer i;
        case (i)
            0: atan = 18'sd8192;  1: atan = 18'sd4836;  2: atan = 18'sd2555;  3: atan = 18'sd1297;
            4: atan = 18'sd651;   5: atan = 18'sd326;   6: atan = 18'sd163;   7: atan = 18'sd81;
            8: atan = 18'sd41;    9: atan = 18'sd20;   10: atan = 18'sd10;   11: atan = 18'sd5;
           12: atan = 18'sd3;    default: atan = 18'sd1;
        endcase
    endfunction

    // stage 0: the left half plane is rotated by half a turn
    always @(posedge clk) begin
        v[0] <= in_valid & rst_n;
        t[0] <= in_tag;
        if (in_x < 0) begin
            x[0] <= -{{2{in_x[W-1]}}, in_x};
            y[0] <= -{{2{in_y[W-1]}}, in_y};
            a[0] <= 18'sd32768;
        end else begin
            x[0] <= {{2{in_x[W-1]}}, in_x};
            y[0] <= {{2{in_y[W-1]}}, in_y};
            a[0] <= 18'sd0;
        end
    end

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : it
            always @(posedge clk) begin
                v[i + 1] <= v[i] & rst_n;
                t[i + 1] <= t[i];
                if (y[i] > 0) begin
                    x[i + 1] <= x[i] + (y[i] >>> i);
                    y[i + 1] <= y[i] - (x[i] >>> i);
                    a[i + 1] <= a[i] + atan(i);
                end else begin
                    x[i + 1] <= x[i] - (y[i] >>> i);
                    y[i + 1] <= y[i] + (x[i] >>> i);
                    a[i + 1] <= a[i] - atan(i);
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        out_valid <= v[N] & rst_n;
        out_tag   <= t[N];
        out_mag   <= x[N];
        out_ang   <= (a[N] > 18'sd32768) ? a[N] - 18'sd65536 : a[N];
    end
endmodule


// Rotation: an angle modulo a turn (the low 16 bits of any binary angle)
// -> 2^15 (cos, sin). Latency 16 clocks.
module cordic_rot #(
    parameter TAGW = 1
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire        [15:0]   in_ang,
    input  wire     [TAGW-1:0]  in_tag,
    output reg                  out_valid,
    output reg  signed [17:0]   out_cos,
    output reg  signed [17:0]   out_sin,
    output reg      [TAGW-1:0]  out_tag
);
    localparam N = 14;

    reg signed [19:0] x [0:N];
    reg signed [19:0] y [0:N];
    reg signed [17:0] a [0:N];
    reg               neg [0:N];
    reg               v [0:N];
    reg    [TAGW-1:0] t [0:N];

    function signed [17:0] atan;
        input integer i;
        case (i)
            0: atan = 18'sd8192;  1: atan = 18'sd4836;  2: atan = 18'sd2555;  3: atan = 18'sd1297;
            4: atan = 18'sd651;   5: atan = 18'sd326;   6: atan = 18'sd163;   7: atan = 18'sd81;
            8: atan = 18'sd41;    9: atan = 18'sd20;   10: atan = 18'sd10;   11: atan = 18'sd5;
           12: atan = 18'sd3;    default: atan = 18'sd1;
        endcase
    endfunction

    // stage 0: modulo a turn into (-32768, 32768], then into (-16384, 16384]
    // by half a turn, remembering to negate the result
    wire        [15:0] lo = in_ang;
    wire signed [17:0] am = (lo > 16'd32768) ? $signed({2'b00, lo}) - 18'sd65536 : $signed({2'b00, lo});
    always @(posedge clk) begin
        v[0] <= in_valid & rst_n;
        t[0] <= in_tag;
        x[0] <= 20'sd19898;
        y[0] <= 20'sd0;
        if (am > 18'sd16384) begin
            a[0] <= am - 18'sd32768;
            neg[0] <= 1'b1;
        end else if (am < -18'sd16384) begin
            a[0] <= am + 18'sd32768;
            neg[0] <= 1'b1;
        end else begin
            a[0] <= am;
            neg[0] <= 1'b0;
        end
    end

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : it
            always @(posedge clk) begin
                v[i + 1] <= v[i] & rst_n;
                t[i + 1] <= t[i];
                neg[i + 1] <= neg[i];
                if (a[i] >= 0) begin
                    x[i + 1] <= x[i] - (y[i] >>> i);
                    y[i + 1] <= y[i] + (x[i] >>> i);
                    a[i + 1] <= a[i] - atan(i);
                end else begin
                    x[i + 1] <= x[i] + (y[i] >>> i);
                    y[i + 1] <= y[i] - (x[i] >>> i);
                    a[i + 1] <= a[i] + atan(i);
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        out_valid <= v[N] & rst_n;
        out_tag   <= t[N];
        out_cos   <= neg[N] ? -x[N][17:0] : x[N][17:0];
        out_sin   <= neg[N] ? -y[N][17:0] : y[N][17:0];
    end
endmodule


// The angle of a wide complex sum (Sync.angle): shifted right until it fits
// 30 bits (arithmetic, so the shift floors), then vectored. Latency 19: the
// largest magnitude, its bit length and the shift take a clock each.
module angle30 #(
    parameter W = 48,
    parameter TAGW = 1
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire signed [W-1:0]  in_re,
    input  wire signed [W-1:0]  in_im,
    input  wire     [TAGW-1:0]  in_tag,
    output wire                 out_valid,
    output wire signed [17:0]   out_ang,
    output wire     [TAGW-1:0]  out_tag
);
    wire [W:0] are = in_re[W-1] ? -{in_re[W-1], in_re} : {in_re[W-1], in_re};
    wire [W:0] aim = in_im[W-1] ? -{in_im[W-1], in_im} : {in_im[W-1], in_im};
    // A1: the larger magnitude
    reg                a1v;
    reg  [TAGW-1:0]    a1tag;
    reg  signed [W-1:0] a1re, a1im;
    reg  [W:0]         mx;
    always @(posedge clk) begin
        a1v   <= in_valid & rst_n;
        a1tag <= in_tag;
        a1re  <= in_re;
        a1im  <= in_im;
        mx    <= (are > aim) ? are : aim;
    end

    // A2: the bit length of mx, the shift
    reg [7:0] bl;
    integer b;
    always @(*) begin
        bl = 8'd0;
        for (b = 0; b <= W; b = b + 1)
            if (mx[b])
                bl = b[7:0] + 8'd1;
    end
    reg                a2v;
    reg  [TAGW-1:0]    a2tag;
    reg  signed [W-1:0] a2re, a2im;
    reg  [7:0]         sh;
    always @(posedge clk) begin
        a2v   <= a1v & rst_n;
        a2tag <= a1tag;
        a2re  <= a1re;
        a2im  <= a1im;
        sh    <= (bl > 8'd30) ? bl - 8'd30 : 8'd0;
    end

    reg               nv;
    reg signed [31:0] nre, nim;
    reg    [TAGW-1:0] ntag;
    // after the shift both fit 30 bits: the top bits are sign copies
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [W-1:0] sre = a2re >>> sh;
    wire signed [W-1:0] sim = a2im >>> sh;
    /* verilator lint_on UNUSEDSIGNAL */
    always @(posedge clk) begin
        nv   <= a2v & rst_n;
        ntag <= a2tag;
        nre  <= sre[31:0];
        nim  <= sim[31:0];
    end

    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [33:0] mag_unused;
    /* verilator lint_on UNUSEDSIGNAL */
    cordic_vec #(.W(32), .TAGW(TAGW)) cv (
        .clk(clk), .rst_n(rst_n), .in_valid(nv), .in_x(nre), .in_y(nim), .in_tag(ntag),
        .out_valid(out_valid), .out_mag(mag_unused), .out_ang(out_ang), .out_tag(out_tag));
endmodule
