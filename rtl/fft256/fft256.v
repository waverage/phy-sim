`timescale 1ns / 1ps
// 256-point fixed-point FFT/IFFT, bit exact against fx.Fft.runFixed
// (modem/src/fx.zig) and twin.fft (harness/twin.py).
//
// Radix-2 decimation in time, in place. Unitary: the even stages (0, 2,
// 4, 6) halve. Each butterfly output is (a 2^16 +- w b) rounded half to
// even by 16 or 17 bits and saturated to 16 bits; twiddles are Q16 in 18
// bits (fft256_twiddle). The inverse conjugates the twiddles.
//
// The data sit in one true-dual-port RAM of 256 x 32 bits (one RAMB18):
// a butterfly reads its pair on both ports in one clock and writes the
// results back in the next, so 1024 butterflies take 2048 clocks. With
// the load and the unload, 2560 clocks a transform: 25.6 us at 100 MHz,
// inside the 28.8 us an OFDM symbol (256 + 32 samples) takes at 10 MS/s.
// Four DSP48 multipliers.
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
    localparam LOAD = 2'd0, CALC = 2'd1, UNLOAD = 2'd2;

    reg        [1:0]  state;
    reg        [7:0]  cnt;
    reg        [2:0]  stage;
    reg        [6:0]  k;
    reg               ph;                  // butterfly phase: 0 read, 1 write
    reg               inv;

    assign in_ready = (state == LOAD);

    // butterfly addresses: group g = k >> stage, j = k mod 2^stage
    wire [7:0] half = 8'd1 << stage;
    wire [7:0] jj   = {1'b0, k} & (half - 8'd1);
    wire [7:0] i0   = (({1'b0, k} >> stage) << ({1'b0, stage} + 4'd1)) | jj;
    wire [7:0] i1   = i0 | half;
    wire [6:0] tw_j = jj[6:0] << (3'd7 - stage);

    // ---- the RAM: port A and port B, each read-first with a registered read
    reg  [31:0] mem [0:255];
    reg  [7:0]  addr_a, addr_b;
    reg         we_a, we_b;
    reg  [31:0] din_a, din_b;
    reg  [31:0] dout_a, dout_b;
    always @(posedge clk) begin
        if (we_a)
            mem[addr_a] <= din_a;
        dout_a <= mem[addr_a];
    end
    always @(posedge clk) begin
        if (we_b)
            mem[addr_b] <= din_b;
        dout_b <= mem[addr_b];
    end

    // ---- the butterfly, on the pair read in the previous clock
    wire signed [17:0] wr, wi_rom;
    fft256_twiddle rom (.j(tw_j), .wr(wr), .wi(wi_rom));
    wire signed [17:0] wi = inv ? -wi_rom : wi_rom;

    wire signed [15:0] a_re = dout_a[31:16], a_im = dout_a[15:0];
    wire signed [15:0] b_re = dout_b[31:16], b_im = dout_b[15:0];
    wire signed [39:0] tr = wr * b_re - wi * b_im;
    wire signed [39:0] ti = wr * b_im + wi * b_re;
    wire signed [39:0] ar = {{8{a_re[15]}}, a_re, 16'd0};
    wire signed [39:0] ai = {{8{a_im[15]}}, a_im, 16'd0};
    wire        [4:0]  sh = stage[0] ? 5'd16 : 5'd17;     // even stages halve

    // x >> s rounded half to even, saturated to 16 bits
    function signed [15:0] rnd_sat;
        input signed [39:0] x;
        input        [4:0]  s;
        reg signed [39:0] q, r, h, v;
        begin
            q = x >>> s;
            r = x - (q <<< s);
            h = 40'sd1 <<< (s - 5'd1);
            v = (r > h || (r == h && q[0])) ? q + 40'sd1 : q;
            if (v > 40'sd32767)
                rnd_sat = 16'sh7fff;
            else if (v < -40'sd32768)
                rnd_sat = 16'sh8000;
            else
                rnd_sat = v[15:0];
        end
    endfunction

    // natural index -> bit-reversed address
    function [7:0] bitrev;
        input [7:0] x;
        integer b;
        begin
            for (b = 0; b < 8; b = b + 1)
                bitrev[b] = x[7 - b];
        end
    endfunction

    // ---- the RAM's ports, from the state
    always @(*) begin
        addr_a = 8'd0;
        addr_b = 8'd0;
        we_a = 1'b0;
        we_b = 1'b0;
        din_a = {in_re, in_im};
        din_b = 32'd0;
        case (state)
            LOAD: begin
                addr_a = bitrev(cnt);
                we_a = in_valid;
            end
            CALC: begin
                addr_a = i0;
                addr_b = i1;
                we_a = ph;
                we_b = ph;
                din_a = {rnd_sat(ar + tr, sh), rnd_sat(ai + ti, sh)};
                din_b = {rnd_sat(ar - tr, sh), rnd_sat(ai - ti, sh)};
            end
            default:                           // UNLOAD
                addr_a = cnt;
        endcase
    end

    assign out_re = dout_a[31:16];
    assign out_im = dout_a[15:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= LOAD;
            cnt       <= 8'd0;
            stage     <= 3'd0;
            k         <= 7'd0;
            ph        <= 1'b0;
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
                        ph    <= 1'b0;
                    end
                end
                CALC: begin
                    ph <= ~ph;
                    if (ph) begin
                        k <= k + 7'd1;
                        if (k == 7'd127) begin
                            stage <= stage + 3'd1;
                            if (stage == 3'd7)
                                state <= UNLOAD;
                        end
                    end
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
