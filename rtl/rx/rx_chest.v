`timescale 1ns / 1ps
// Channel estimate at the pilot symbols (fxmodem.rxFrame steps 2-3): the
// least-squares estimate y conj(pilot) of each pilot RE (the pilots are
// (+-1 +-j) times a constant, so this is two additions), then for each
// subcarrier f the frequency filter, L complex taps over the subcarriers
// start[f] .. start[f] + L - 1, accumulated exactly and rounded half up by
// 15 bits, saturated to 16. One tap per clock: npsym F L clocks, in a
// 6-stage pipeline.
//
// Tables (rx_frame): start[f] and the pilot signs read without latency,
// the taps and the frame memory with one clock.
module rx_chest (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [7:0]  nf,
    input  wire [5:0]  taps,
    input  wire [1:0]  npsym,
    input  wire [3:0]  psym0,
    input  wire [3:0]  psym1,
    input  wire [3:0]  psym2,
    output wire [7:0]  st_addr,
    input  wire [7:0]  st_data,
    output wire [9:0]  sg_addr,
    input  wire [1:0]  sg_data,            // {pilot re < 0, pilot im < 0}
    output wire [12:0] tp_addr,
    input  wire [31:0] tp_data,            // {re, im} Q15
    output wire [11:0] y_addr,
    input  wire [31:0] y_data,
    output reg         hf_we,
    output reg  [9:0]  hf_addr,
    output reg  [31:0] hf_data,
    output reg         done
);
    reg        run;
    reg [1:0]  t;
    reg [7:0]  f;
    reg [4:0]  j;
    wire [3:0] s = (t == 2'd0) ? psym0 : (t == 2'd1) ? psym1 : psym2;
    wire [11:0] sf = s * nf;
    wire [9:0]  tf = t * nf;
    wire [7:0]  pos = st_data + {3'd0, j};
    wire last_tap = ({1'b0, j} == taps - 6'd1);

    // pipeline: 0 the addresses, 1 the RAMs answer, 2 their words registered,
    // 3 LS, 4 products, then the accumulation
    reg        p0_v, p0_first, p0_last;
    reg [9:0]  p0_a, p0_sga;
    reg [11:0] p0_ya;
    reg [12:0] p0_tpa;
    assign st_addr = f;
    assign sg_addr = p0_sga;
    assign y_addr  = p0_ya;
    assign tp_addr = p0_tpa;
    reg        p1_v, p1_first, p1_last;
    reg [1:0]  p1_sg;
    reg [9:0]  p1_a;
    reg        p2_v, p2_first, p2_last;
    reg [1:0]  p2_sg;
    reg [9:0]  p2_a;
    reg [31:0] p2_y, p2_w;
    reg        p3_v, p3_first, p3_last;
    reg [9:0]  p3_a;
    reg signed [17:0] p3_lr, p3_li;
    reg signed [15:0] p3_wr, p3_wi;
    reg        p4_v, p4_first, p4_last;
    reg [9:0]  p4_a;
    reg signed [34:0] p4_pr, p4_pi;
    reg signed [47:0] acc_r, acc_i;

    wire signed [15:0] yr = p2_y[31:16], yi = p2_y[15:0];
    wire signed [17:0] ayr = p2_sg[1] ? -{{2{yr[15]}}, yr} : {{2{yr[15]}}, yr};
    wire signed [17:0] ayi = p2_sg[1] ? -{{2{yi[15]}}, yi} : {{2{yi[15]}}, yi};
    wire signed [17:0] byr = p2_sg[0] ? -{{2{yr[15]}}, yr} : {{2{yr[15]}}, yr};
    wire signed [17:0] byi = p2_sg[0] ? -{{2{yi[15]}}, yi} : {{2{yi[15]}}, yi};
    wire signed [47:0] nr = (p4_first ? 48'sd0 : acc_r) + {{13{p4_pr[34]}}, p4_pr};
    wire signed [47:0] ni = (p4_first ? 48'sd0 : acc_i) + {{13{p4_pi[34]}}, p4_pi};

    function signed [15:0] rnd_sat15;
        input signed [47:0] x;
        reg signed [47:0] q;
        begin
            q = (x + 48'sd16384) >>> 15;
            if (q > 48'sd32767)
                rnd_sat15 = 16'sh7fff;
            else if (q < -48'sd32768)
                rnd_sat15 = 16'sh8000;
            else
                rnd_sat15 = q[15:0];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            run <= 1'b0;
            p0_v <= 1'b0;
            p1_v <= 1'b0;
            p2_v <= 1'b0;
            p3_v <= 1'b0;
            p4_v <= 1'b0;
            hf_we <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start) begin
                run <= 1'b1;
                t <= 2'd0;
                f <= 8'd0;
                j <= 5'd0;
            end else if (run) begin
                if (last_tap) begin
                    j <= 5'd0;
                    if (f == nf - 8'd1) begin
                        f <= 8'd0;
                        t <= t + 2'd1;
                        if (t == npsym - 2'd1)
                            run <= 1'b0;
                    end else
                        f <= f + 8'd1;
                end else
                    j <= j + 5'd1;
            end
            p0_v     <= run;
            p0_first <= (j == 5'd0);
            p0_last  <= last_tap;
            p0_a     <= tf + {2'd0, f};
            p0_sga   <= tf + {2'd0, pos};
            p0_ya    <= sf + {4'd0, pos};
            p0_tpa   <= {f, j};
            p1_v     <= p0_v;
            p1_first <= p0_first;
            p1_last  <= p0_last;
            p1_a     <= p0_a;
            p1_sg    <= sg_data;
            p2_v     <= p1_v;
            p2_first <= p1_first;
            p2_last  <= p1_last;
            p2_a     <= p1_a;
            p2_sg    <= p1_sg;
            p2_y     <= y_data;
            p2_w     <= tp_data;
            p3_v     <= p2_v;
            p3_first <= p2_first;
            p3_last  <= p2_last;
            p3_a     <= p2_a;
            p3_lr    <= ayr + byi;                 // a yr + b yi
            p3_li    <= ayi - byr;                 // a yi - b yr
            p3_wr    <= p2_w[31:16];
            p3_wi    <= p2_w[15:0];
            p4_v     <= p3_v;
            p4_first <= p3_first;
            p4_last  <= p3_last;
            p4_a     <= p3_a;
            p4_pr    <= p3_wr * p3_lr - p3_wi * p3_li;
            p4_pi    <= p3_wr * p3_li + p3_wi * p3_lr;
            if (p4_v) begin
                acc_r <= nr;
                acc_i <= ni;
            end
            hf_we   <= p4_v && p4_last;
            hf_addr <= p4_a;
            hf_data <= {rnd_sat15(nr), rnd_sat15(ni)};
            if (p4_v && p4_last && !p3_v)
                done <= 1'b1;
        end
    end
endmodule
