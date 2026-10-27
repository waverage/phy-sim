`timescale 1ns / 1ps
// Channel estimate at the pilot symbols (fxmodem.rxFrame steps 2-3): the
// least-squares estimate y conj(pilot) of each pilot RE (the pilots are
// (+-1 +-j) times a constant, so this is two additions), then for each
// subcarrier f the frequency filter, L complex taps over the subcarriers
// start[f] .. start[f] + L - 1, accumulated exactly and rounded half up by
// 15 bits, saturated to 16.
//
// Per pilot symbol t: the F least-squares values go to a buffer of their
// own (one a clock), then the filter takes two taps a clock from it (taps
// j and j + 1 from the even and odd tap memories): about
// npsym F (1 + L / 2) clocks. The estimate of (t, f) is written at {t, f}.
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
    output wire [11:0] tp_addr,            // {f, j / 2}
    input  wire [31:0] tp0_data,           // tap j, {re, im} Q15
    input  wire [31:0] tp1_data,           // tap j + 1
    output wire [11:0] y_addr,
    input  wire [31:0] y_data,
    output reg         hf_we,
    output reg  [9:0]  hf_addr,
    output reg  [31:0] hf_data,
    output reg         done
);
    localparam C_IDLE = 3'd0, C_COPY = 3'd1, C_WAIT = 3'd2, C_FILT = 3'd3, C_NEXT = 3'd4, C_END = 3'd5;
    reg  [2:0] st;
    reg  [1:0] t;
    reg  [7:0] f, p;
    reg  [3:0] jp;                         // the tap pair
    wire [3:0] s = (t == 2'd0) ? psym0 : (t == 2'd1) ? psym1 : psym2;
    wire [11:0] sf = s * nf;
    wire [9:0]  tf = t * nf;
    wire [4:0]  npair_m1 = taps[5:1] + {4'd0, taps[0]} - 5'd1;
    wire last_pair = ({1'b0, jp} == npair_m1);

    // ---- the least-squares buffer: written in C_COPY, two reads in C_FILT
    reg  [35:0] ls_mem [0:255];
    reg         ls_we;
    reg  [7:0]  ls_wa;
    reg  [35:0] ls_wd;
    reg  [35:0] ls0_q, ls1_q;
    // A0: the pair, its subcarrier's window start (the start table) registered
    reg         a0_v, a0_first, a0_last, a0_two;
    reg  [7:0]  a0_f, a0_st;
    reg  [3:0]  a0_jp;
    reg  [1:0]  a0_t;
    wire [7:0]  pos0 = a0_st + {3'd0, a0_jp, 1'b0};
    always @(posedge clk) begin
        if (ls_we)
            ls_mem[ls_wa] <= ls_wd;
        ls0_q <= ls_mem[pos0];
    end
    always @(posedge clk)
        ls1_q <= ls_mem[pos0 + 8'd1];

    assign st_addr = f;
    assign sg_addr = tf + {2'd0, p};
    assign y_addr  = sf + {4'd0, p};
    assign tp_addr = {a0_f, a0_jp};

    // C_COPY pipeline: the address at p, the word a clock later, LS, the write
    reg        c1_v;
    reg [7:0]  c1_p;
    reg [1:0]  c1_sg;
    wire signed [15:0] yr = y_data[31:16], yi = y_data[15:0];
    wire signed [17:0] ayr = c1_sg[1] ? -{{2{yr[15]}}, yr} : {{2{yr[15]}}, yr};
    wire signed [17:0] ayi = c1_sg[1] ? -{{2{yi[15]}}, yi} : {{2{yi[15]}}, yi};
    wire signed [17:0] byr = c1_sg[0] ? -{{2{yr[15]}}, yr} : {{2{yr[15]}}, yr};
    wire signed [17:0] byi = c1_sg[0] ? -{{2{yi[15]}}, yi} : {{2{yi[15]}}, yi};
    wire signed [17:0] lr = ayr + byi;     // a yr + b yi
    wire signed [17:0] li = ayi - byr;     // a yi - b yr

    // C_FILT pipeline: A0 the pair, 0 the addresses, 1 the RAMs answer, 2 their words
    // registered, 3 the products, 4 the pair's sum, then the accumulation
    reg        p1_v, p1_first, p1_last, p1_two;
    reg [9:0]  p1_a;
    reg        p2_v, p2_first, p2_last, p2_two;
    reg [9:0]  p2_a;
    reg [35:0] p2_l0, p2_l1;
    reg [31:0] p2_w0, p2_w1;
    reg        p3_v, p3_first, p3_last;
    reg [9:0]  p3_a;
    reg signed [34:0] p3_pr0, p3_pi0, p3_pr1, p3_pi1;
    reg        p4_v, p4_first, p4_last;
    reg [9:0]  p4_a;
    reg signed [35:0] p4_pr, p4_pi;
    reg signed [47:0] acc_r, acc_i;
    wire signed [17:0] l0r = p2_l0[35:18], l0i = p2_l0[17:0], l1r = p2_l1[35:18], l1i = p2_l1[17:0];
    wire signed [15:0] w0r = p2_w0[31:16], w0i = p2_w0[15:0], w1r = p2_w1[31:16], w1i = p2_w1[15:0];
    wire signed [47:0] nr = (p4_first ? 48'sd0 : acc_r) + {{12{p4_pr[35]}}, p4_pr};
    wire signed [47:0] ni = (p4_first ? 48'sd0 : acc_i) + {{12{p4_pi[35]}}, p4_pi};

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

    wire filt = (st == C_FILT);
    wire busy_pipe = c1_v || ls_we || a0_v || p1_v || p2_v || p3_v || p4_v || hf_we;

    always @(posedge clk) begin
        // copy pipeline
        c1_v  <= (st == C_COPY) && rst_n;
        c1_p  <= p;
        c1_sg <= sg_data;
        ls_we <= c1_v && rst_n;
        ls_wa <= c1_p;
        ls_wd <= {lr, li};
        // filter pipeline
        a0_v     <= filt && rst_n;
        a0_first <= (jp == 4'd0);
        a0_last  <= last_pair;
        a0_two   <= ({2'b00, jp, 1'b1} < {1'b0, taps});
        a0_f     <= f;
        a0_st    <= st_data;
        a0_jp    <= jp;
        a0_t     <= t;
        p1_v     <= a0_v && rst_n;
        p1_first <= a0_first;
        p1_last  <= a0_last;
        p1_two   <= a0_two;
        p1_a     <= {a0_t, a0_f};
        p2_v     <= p1_v && rst_n;
        p2_first <= p1_first;
        p2_last  <= p1_last;
        p2_two   <= p1_two;
        p2_a     <= p1_a;
        p2_l0    <= ls0_q;
        p2_l1    <= ls1_q;
        p2_w0    <= tp0_data;
        p2_w1    <= tp1_data;
        p3_v     <= p2_v && rst_n;
        p3_first <= p2_first;
        p3_last  <= p2_last;
        p3_a     <= p2_a;
        p3_pr0   <= w0r * l0r - w0i * l0i;
        p3_pi0   <= w0r * l0i + w0i * l0r;
        p3_pr1   <= p2_two ? w1r * l1r - w1i * l1i : 35'sd0;
        p3_pi1   <= p2_two ? w1r * l1i + w1i * l1r : 35'sd0;
        p4_v     <= p3_v && rst_n;
        p4_first <= p3_first;
        p4_last  <= p3_last;
        p4_a     <= p3_a;
        p4_pr    <= {p3_pr0[34], p3_pr0} + {p3_pr1[34], p3_pr1};
        p4_pi    <= {p3_pi0[34], p3_pi0} + {p3_pi1[34], p3_pi1};
        if (p4_v) begin
            acc_r <= nr;
            acc_i <= ni;
        end
        hf_we   <= p4_v && p4_last && rst_n;
        hf_addr <= p4_a;
        hf_data <= {rnd_sat15(nr), rnd_sat15(ni)};

        if (!rst_n) begin
            st <= C_IDLE;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (st)
                C_IDLE: if (start) begin
                    st <= C_COPY;
                    t <= 2'd0;
                    p <= 8'd0;
                end
                C_COPY: begin
                    p <= p + 8'd1;
                    if (p == nf - 8'd1)
                        st <= C_WAIT;
                end
                C_WAIT: if (!busy_pipe) begin    // the buffer is complete (and the last symbol's writes are out)
                    st <= C_FILT;
                    f <= 8'd0;
                    jp <= 4'd0;
                end
                C_FILT: begin
                    if (last_pair) begin
                        jp <= 4'd0;
                        if (f == nf - 8'd1)
                            st <= C_NEXT;
                        else
                            f <= f + 8'd1;
                    end else
                        jp <= jp + 4'd1;
                end
                C_NEXT: if (!busy_pipe) begin    // the filter has read the buffer for the last time
                    if (t == npsym - 2'd1)
                        st <= C_END;
                    else begin
                        t <= t + 2'd1;
                        p <= 8'd0;
                        st <= C_COPY;
                    end
                end
                default: begin                   // C_END
                    st <= C_IDLE;
                    done <= 1'b1;
                end
            endcase
        end
    end
endmodule
