`timescale 1ns / 1ps
// Layered offset min-sum LDPC decoder for the 5G base graphs, bit exact
// against fxldpc.Decoder(i32) (modem/src/fxldpc.zig) and twin.ldpc_decode
// (harness/twin.py): 6-bit channel LLRs, 8-bit APPs, 6-bit messages, the
// offset in LSBs, up to 31 iterations with the syndrome early stop.
//
// Partly parallel: P = 16 lanes. The lifting z must be a multiple of 16
// (every code of the MCS table is); Q = z / 16 words hold one base column,
// word w lane j the variable w + j Q. A base row's check rows are taken in
// Q groups of 16 (group w: rows w + j Q), and for an entry (column c,
// shift s = a + b Q) group w needs variables (w + j Q + s) mod z: one word,
// c Q + (w + a) mod Q, rotated by b lanes (b + 1 if w + a wraps).
//
// A group reads its d words (one per clock) with their old messages and
// finds the two smallest magnitudes; its d results are written back in
// the next d clocks while the next group reads. Between base rows the
// pipeline drains, so a row sees every update of the row before it. One
// iteration takes about edges * Q + rows * (d + 4) clocks; the syndrome
// check after an iteration reads every edge once more.
//
// Configuration (cfg_*, from the control processor, harness/rtlcfg.py):
// registers at addresses 0..12, the edge table at 512 + e:
// {last in row, c Q, a, b}. A codeword: start, then the n LLRs in order
// (in_valid / in_ready), then out_valid with the k decoded bits in order,
// then done with the iteration count.
/* verilator lint_off UNUSEDSIGNAL */
module ldpc_dec #(
    parameter AW = 10,                 // APP words: dec_cols * Q <= 2^AW
    parameter CW = 12,                 // message words: edges * Q <= 2^CW
    parameter EW = 9                   // edge table entries <= 2^EW
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              cfg_we,
    input  wire [9:0]        cfg_addr,
    input  wire [31:0]       cfg_data,
    input  wire              start,
    output wire              busy,
    input  wire              in_valid,
    input  wire signed [5:0] in_llr,
    output wire              in_ready,
    output reg               out_valid,
    output reg               out_bit,
    output reg  [13:0]       out_idx,
    output reg               done,
    output reg  [4:0]        iters
);
    /* verilator lint_on UNUSEDSIGNAL */
    localparam P = 16;
    localparam A = 8;                  // APP bits
    localparam M = 6;                  // message bits
    localparam [4:0] MSG_MAX = 5'd31;

    // ---- configuration
    localparam R_Z = 4'd0, R_Q = 4'd1, R_K = 4'd2, R_N = 4'd3, R_NSYS = 4'd4, R_KBBASE = 4'd5, R_INITW = 4'd6,
               R_NEDGE = 4'd7, R_ITER = 4'd8, R_OFFSET = 4'd9, R_ESTOP = 4'd10, R_KLDPC = 4'd11, R_COL2 = 4'd12;
    reg [8:0]    r_z;
    reg [4:0]    r_q;
    reg [13:0]   r_k, r_n, r_nsys, r_kldpc;
    reg [AW-1:0] r_kbbase, r_initw, r_col2;
    reg [EW:0]   r_nedge;
    reg [4:0]    r_iter, r_off;
    reg          r_estop;
    reg [AW+9:0] etab [0:(1 << EW) - 1];      // {last, col base, a[4:0], b[3:0]}

    always @(posedge clk) begin
        if (cfg_we && cfg_addr[9])
            etab[cfg_addr[EW-1:0]] <= cfg_data[AW+9:0];
        if (cfg_we && !cfg_addr[9]) begin
            case (cfg_addr[3:0])
                R_Z:      r_z      <= cfg_data[8:0];
                R_Q:      r_q      <= cfg_data[4:0];
                R_K:      r_k      <= cfg_data[13:0];
                R_N:      r_n      <= cfg_data[13:0];
                R_NSYS:   r_nsys   <= cfg_data[13:0];
                R_KBBASE: r_kbbase <= cfg_data[AW-1:0];
                R_INITW:  r_initw  <= cfg_data[AW-1:0];
                R_NEDGE:  r_nedge  <= cfg_data[EW:0];
                R_ITER:   r_iter   <= cfg_data[4:0];
                R_OFFSET: r_off    <= cfg_data[4:0];
                R_ESTOP:  r_estop  <= cfg_data[0];
                R_KLDPC:  r_kldpc  <= cfg_data[13:0];
                R_COL2:   r_col2   <= cfg_data[AW-1:0];
                default: ;
            endcase
        end
    end

    // ---- the APP memory: one write port with a write enable per lane, one read port
    reg  [P*A-1:0] app_mem [0:(1 << AW) - 1];
    reg  [AW-1:0]  app_waddr, app_raddr;
    reg  [P-1:0]   app_we;
    reg  [P*A-1:0] app_wdata;
    reg  [P*A-1:0] app_dout;
    // the writes are registered once more (app_*_q): the write data path is
    // a clock of logic of its own
    reg  [AW-1:0]  app_waddr_q;
    reg  [P-1:0]   app_we_q;
    reg  [P*A-1:0] app_wdata_q;
    integer l;
    always @(posedge clk) begin
        for (l = 0; l < P; l = l + 1)
            if (app_we_q[l])
                app_mem[app_waddr_q][l*A +: A] <= app_wdata_q[l*A +: A];
    end
    always @(posedge clk)
        app_dout <= app_mem[app_raddr];

    // ---- the message memory, in processing order
    reg  [P*M-1:0] c2v_mem [0:(1 << CW) - 1];
    reg  [CW-1:0]  c2v_rp, c2v_wp;
    reg            c2v_we;
    reg  [P*M-1:0] c2v_wdata;
    reg  [P*M-1:0] c2v_dout;
    reg            c2v_we_q;
    reg  [CW-1:0]  c2v_wa_q;
    reg  [P*M-1:0] c2v_wd_q;
    always @(posedge clk) begin
        app_we_q    <= rst_n ? app_we : {P{1'b0}};
        app_waddr_q <= app_waddr;
        app_wdata_q <= app_wdata;
        c2v_we_q    <= rst_n && c2v_we;
        c2v_wa_q    <= c2v_wp;
        c2v_wd_q    <= c2v_wdata;
    end
    always @(posedge clk) begin
        if (c2v_we_q)
            c2v_mem[c2v_wa_q] <= c2v_wd_q;
        c2v_dout <= c2v_mem[c2v_rp];
    end

    // ---- lane helpers
    function [P*A-1:0] rot_down;               // out lane j = in lane (j + r) mod P
        input [P*A-1:0] x;
        input [3:0]     r;
        integer m;
        reg   [3:0] s;
        begin
            for (m = 0; m < P; m = m + 1) begin
                s = m[3:0] + r;
                rot_down[m*A +: A] = x[{s, 3'd0} +: A];
            end
        end
    endfunction
    function [P*A-1:0] rot_up;                 // out lane (j + r) mod P = in lane j
        input [P*A-1:0] x;
        input [3:0]     r;
        integer m;
        reg   [3:0] s;
        begin
            for (m = 0; m < P; m = m + 1) begin
                s = m[3:0] - r;
                rot_up[m*A +: A] = x[{s, 3'd0} +: A];
            end
        end
    endfunction
    function [A-1:0] sat_app;                  // symmetric saturation to 8 bits
        input signed [A+1:0] v;
        begin
            if (v > 10'sd127)
                sat_app = 8'd127;
            else if (v < -10'sd127)
                sat_app = 8'h81;
            else
                sat_app = v[A-1:0];
        end
    endfunction

    // ---- state
    localparam S_IDLE = 3'd0, S_INIT = 3'd1, S_LOAD = 3'd2, S_READ = 3'd3, S_DRAIN = 3'd4, S_OUT = 3'd5,
               S_DONE = 3'd6;
    reg [2:0]    state;
    reg          synd;                         // the current pass is the syndrome check
    reg          synd_fail;
    reg [4:0]    it;
    reg [EW:0]   ep, row_start;
    reg [4:0]    w;
    reg [AW-1:0] cnt;                          // INIT word counter
    reg [13:0]   t;                            // LOAD / OUT position
    reg [AW-1:0] colbase;                      // LOAD / OUT position: column * Q, word, lane, index in column
    reg [4:0]    pw;
    reg [3:0]    plane;
    reg [8:0]    pi;

    assign busy     = (state != S_IDLE);
    assign in_ready = (state == S_LOAD);

    // ---- read issue (S_READ): edge ep at group w
    wire [AW+9:0] ent = etab[ep[EW-1:0]];
    wire          e_last = ent[AW+9];
    wire [AW-1:0] e_cb = ent[AW+8:9];
    wire [4:0]    e_a = ent[8:4];
    wire [3:0]    e_b = ent[3:0];
    wire [5:0]    wa = {1'b0, w} + {1'b0, e_a};
    wire          wrap = (wa >= {1'b0, r_q});
    wire [4:0]    wsel = wrap ? wa[4:0] - r_q : wa[4:0];
    wire [3:0]    rd_rot = wrap ? e_b + 4'd1 : e_b;
    wire [AW-1:0] rd_addr = e_cb + {{(AW-5){1'b0}}, wsel};
    wire [4:0]    eidx = ep[4:0] - row_start[4:0];   // < 32: the edge's index in its row
    wire          issue = (state == S_READ);

    // stage 1: the RAM outputs arrive
    reg          s1_v, s1_first, s1_last, s1_synd;
    reg [3:0]    s1_rot;
    reg [AW-1:0] s1_addr;
    reg [4:0]    s1_eidx;
    // stage 2: tt registered
    reg          s2_v, s2_first, s2_last, s2_synd;
    reg [3:0]    s2_rot;
    reg [AW-1:0] s2_addr;
    reg [4:0]    s2_eidx;
    reg [P*A-1:0] s2_tt;
    reg [P-1:0]   s2_hard;

    wire [P*A-1:0] v_rot = rot_down(app_dout, s1_rot);
    reg  [P*A-1:0] tt_c;
    reg  [P-1:0]   hard_c;
    reg  [A+1:0]   vv, cv;
    integer j;
    always @(*) begin
        for (j = 0; j < P; j = j + 1) begin
            vv = {{2{v_rot[j*A+A-1]}}, v_rot[j*A +: A]};
            cv = (it == 5'd0) ? {(A+2){1'b0}} : {{4{c2v_dout[j*M+M-1]}}, c2v_dout[j*M +: M]};
            tt_c[j*A +: A] = sat_app(vv - cv);
            hard_c[j] = v_rot[j*A+A-1] || (v_rot[j*A +: A] == {A{1'b0}});
        end
    end

    // the group's running minimum (stage 2 -> accumulators)
    reg [5*P-1:0] min1, min2, idx;
    reg [P-1:0]   par, sacc;                   // sign parity, syndrome XOR
    reg [5*P-1:0] n_min1, n_min2, n_idx;
    reg [P-1:0]   n_par, n_sacc;
    reg [A-1:0]   tv, av;
    reg [4:0]     mag;
    always @(*) begin
        for (j = 0; j < P; j = j + 1) begin
            tv = s2_tt[j*A +: A];
            av = tv[A-1] ? -tv : tv;
            mag = (av > 8'd31) ? MSG_MAX : av[4:0];
            if (s2_first) begin
                n_min1[j*5 +: 5] = mag;
                n_min2[j*5 +: 5] = MSG_MAX;
                n_idx[j*5 +: 5]  = 5'd0;
                n_par[j]         = tv[A-1];
                n_sacc[j]        = s2_hard[j];
            end else begin
                n_par[j]  = par[j] ^ tv[A-1];
                n_sacc[j] = sacc[j] ^ s2_hard[j];
                if (mag < min1[j*5 +: 5]) begin
                    n_min1[j*5 +: 5] = mag;
                    n_min2[j*5 +: 5] = min1[j*5 +: 5];
                    n_idx[j*5 +: 5]  = s2_eidx;
                end else begin
                    n_min1[j*5 +: 5] = min1[j*5 +: 5];
                    n_min2[j*5 +: 5] = (mag < min2[j*5 +: 5]) ? mag : min2[j*5 +: 5];
                    n_idx[j*5 +: 5]  = idx[j*5 +: 5];
                end
            end
        end
    end

    // the latched result of the last complete group, and the write engine
    reg [5*P-1:0] q_min1, q_min2, q_idx;
    reg [P-1:0]   q_par;
    reg [4:0]     wr_rem;
    // the FIFO from stage 2 to the write engine
    reg [P*A-1:0] f_tt   [0:31];
    reg [AW-1:0]  f_addr [0:31];
    reg [3:0]     f_rot  [0:31];
    reg [4:0]     f_eidx [0:31];
    reg [4:0]     f_wp, f_rp;
    wire          f_push = s2_v && !s2_synd;
    wire          wr_act = (wr_rem != 5'd0);

    // W1: the FIFO head and its group's result registered; W2 (below) the
    // messages, the new APPs and their rotation into the write registers
    reg            w1_v;
    reg  [P*A-1:0] w_tt;
    reg  [4:0]     w_eidx;
    reg  [AW-1:0]  w1_addr;
    reg  [3:0]     w1_rot;
    reg  [5*P-1:0] w1_min1, w1_min2, w1_idx;
    reg  [P-1:0]   w1_par;
    always @(posedge clk) begin
        w1_v    <= wr_act && rst_n;
        w_tt    <= f_tt[f_rp];
        w_eidx  <= f_eidx[f_rp];
        w1_addr <= f_addr[f_rp];
        w1_rot  <= f_rot[f_rp];
        w1_min1 <= q_min1;
        w1_min2 <= q_min2;
        w1_idx  <= q_idx;
        w1_par  <= q_par;
    end
    reg  [P*A-1:0] w_tn;
    reg  [P*M-1:0] w_msg;
    reg  [4:0]     w_new, w_m;
    reg  [M-1:0]   w_sm;
    always @(*) begin
        for (j = 0; j < P; j = j + 1) begin
            w_m = (w_eidx == w1_idx[j*5 +: 5]) ? w1_min2[j*5 +: 5] : w1_min1[j*5 +: 5];
            w_new = (w_m > r_off) ? w_m - r_off : 5'd0;
            w_sm = (w1_par[j] ^ w_tt[j*A+A-1]) ? -{1'b0, w_new} : {1'b0, w_new};
            w_msg[j*M +: M] = w_sm;
            w_tn[j*A +: A] = sat_app({{2{w_tt[j*A+A-1]}}, w_tt[j*A +: A]} + {{4{w_sm[M-1]}}, w_sm});
        end
    end

    // W2: the new APPs and messages registered; their rotation feeds the write registers
    reg            w2_v;
    reg  [P*A-1:0] w2_tn;
    reg  [P*M-1:0] w2_msg;
    reg  [AW-1:0]  w2_addr;
    reg  [3:0]     w2_rot;
    always @(posedge clk) begin
        w2_v    <= w1_v && rst_n;
        w2_tn   <= w_tn;
        w2_msg  <= w_msg;
        w2_addr <= w1_addr;
        w2_rot  <= w1_rot;
    end

    wire pipe_empty = !s1_v && !s2_v && (f_wp == f_rp) && !wr_act && !w1_v && !w2_v && (app_we_q == {P{1'b0}}) && !c2v_we_q;

    // ---- LOAD: -sat(llr) into one lane
    wire [5:0] llr_s = (in_llr == -6'sd32) ? 6'h21 : in_llr;
    wire [A-1:0] llr_app = -{{2{llr_s[5]}}, llr_s};

    // ---- INIT: per lane, filler (k <= index < k_ldpc) -> +127. fl[j] is
    // the index of lane j of the word being written, column z + word + j Q.
    reg  [13:0]    fl [0:P-1];
    wire [13:0]    fstep = (pw == r_q - 5'd1) ? {5'd0, r_z} - {9'd0, r_q} + 14'd1 : 14'd1;
    reg  [P*A-1:0] init_word;
    integer jl;
    always @(posedge clk) begin
        for (jl = 0; jl < P; jl = jl + 1) begin
            if (state == S_IDLE)
                fl[jl] <= r_q * jl[3:0];
            else if (state == S_INIT)
                fl[jl] <= fl[jl] + fstep;
        end
    end
    always @(*) begin
        for (j = 0; j < P; j = j + 1)
            init_word[j*A +: A] = (fl[j] >= r_k && fl[j] < r_kldpc) ? 8'd127 : 8'd0;
    end

    // ---- OUT: the hard decision of the word read in the previous clock
    reg          o_v;
    reg [3:0]    o_lane;
    reg [13:0]   o_idx;
    wire [A-1:0] o_val = app_dout[{o_lane, 3'd0} +: A];

    // the memories' ports, from the state
    always @(*) begin
        app_raddr = rd_addr;
        app_waddr = w2_addr;
        app_we    = {P{1'b0}};
        app_wdata = rot_up(w2_tn, w2_rot);
        c2v_we    = 1'b0;
        c2v_wdata = w2_msg;
        case (state)
            S_INIT: begin
                app_waddr = cnt;
                app_we    = {P{1'b1}};
                app_wdata = init_word;
            end
            S_LOAD: begin
                app_waddr = colbase + {{(AW-5){1'b0}}, pw};
                app_we    = in_valid ? ({{(P-1){1'b0}}, 1'b1} << plane) : {P{1'b0}};
                app_wdata = {P{llr_app}};
            end
            S_OUT:
                app_raddr = colbase + {{(AW-5){1'b0}}, pw};
            default: begin
                app_we = w2_v ? {P{1'b1}} : {P{1'b0}};
                c2v_we = w2_v;
            end
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            s1_v <= 1'b0;
            s2_v <= 1'b0;
            o_v <= 1'b0;
            wr_rem <= 5'd0;
            f_wp <= 5'd0;
            f_rp <= 5'd0;
            out_valid <= 1'b0;
            done <= 1'b0;
            iters <= 5'd0;
        end else begin
            done <= 1'b0;
            out_valid <= 1'b0;

            // stage 1 <- issue
            s1_v     <= issue;
            s1_first <= (ep == row_start);
            s1_last  <= e_last;
            s1_rot   <= rd_rot;
            s1_addr  <= rd_addr;
            s1_eidx  <= eidx;
            s1_synd  <= synd;
            // stage 2 <- stage 1
            s2_v     <= s1_v;
            s2_first <= s1_first;
            s2_last  <= s1_last;
            s2_rot   <= s1_rot;
            s2_addr  <= s1_addr;
            s2_eidx  <= s1_eidx;
            s2_synd  <= s1_synd;
            s2_tt    <= tt_c;
            s2_hard  <= hard_c;
            // accumulators <- stage 2
            if (s2_v) begin
                min1 <= n_min1;
                min2 <= n_min2;
                idx  <= n_idx;
                par  <= n_par;
                sacc <= n_sacc;
                if (s2_last && s2_synd && n_sacc != {P{1'b0}})
                    synd_fail <= 1'b1;
            end
            if (f_push) begin
                f_tt[f_wp]   <= s2_tt;
                f_addr[f_wp] <= s2_addr;
                f_rot[f_wp]  <= s2_rot;
                f_eidx[f_wp] <= s2_eidx;
                f_wp <= f_wp + 5'd1;
            end
            // the write engine
            if (wr_act)
                f_rp <= f_rp + 5'd1;
            if (w2_v)
                c2v_wp <= c2v_wp + 1'b1;
            if (s2_v && s2_last && !s2_synd) begin
                q_min1 <= n_min1;
                q_min2 <= n_min2;
                q_idx  <= n_idx;
                q_par  <= n_par;
                wr_rem <= s2_eidx + 5'd1;
            end else if (wr_act)
                wr_rem <= wr_rem - 5'd1;

            // OUT stage 2
            o_v <= 1'b0;
            if (o_v) begin
                out_valid <= 1'b1;
                out_bit   <= o_val[A-1] || (o_val == {A{1'b0}});
                out_idx   <= o_idx;
            end

            case (state)
                S_IDLE: if (start) begin
                    state <= S_INIT;
                    cnt   <= {AW{1'b0}};
                    pw    <= 5'd0;
                end
                S_INIT: begin
                    cnt <= cnt + 1'b1;
                    if (pw == r_q - 5'd1) begin
                        pw    <= 5'd0;
                    end else begin
                        pw    <= pw + 5'd1;
                    end
                    if (cnt == r_initw - 1'b1) begin
                        state   <= S_LOAD;
                        t       <= 14'd0;
                        colbase <= r_col2;
                        pw      <= 5'd0;
                        plane   <= 4'd0;
                        pi      <= 9'd0;
                    end
                end
                S_LOAD: if (in_valid) begin
                    t <= t + 14'd1;
                    if (t == r_nsys - 14'd1) begin
                        colbase <= r_kbbase;
                        pw <= 5'd0;
                        plane <= 4'd0;
                        pi <= 9'd0;
                    end else if (pi == r_z - 9'd1) begin
                        colbase <= colbase + {{(AW-5){1'b0}}, r_q};
                        pw <= 5'd0;
                        plane <= 4'd0;
                        pi <= 9'd0;
                    end else begin
                        pi <= pi + 9'd1;
                        if (pw == r_q - 5'd1) begin
                            pw <= 5'd0;
                            plane <= plane + 4'd1;
                        end else
                            pw <= pw + 5'd1;
                    end
                    if (t == r_n - 14'd1) begin
                        state <= S_DRAIN;            // the last write lands, then the first row
                        synd <= 1'b0;
                        it <= 5'd0;
                        ep <= {(EW+1){1'b0}};
                        row_start <= {(EW+1){1'b0}};
                        w <= 5'd0;
                        c2v_rp <= {CW{1'b0}};
                        c2v_wp <= {CW{1'b0}};
                    end
                end
                S_READ: begin
                    if (!synd)
                        c2v_rp <= c2v_rp + 1'b1;
                    if (e_last) begin
                        if (w == r_q - 5'd1) begin
                            state <= S_DRAIN;
                            w <= 5'd0;
                            ep <= ep + 1'b1;
                            row_start <= ep + 1'b1;
                        end else begin
                            w <= w + 5'd1;
                            ep <= row_start;
                        end
                    end else
                        ep <= ep + 1'b1;
                end
                S_DRAIN: if (pipe_empty) begin
                    if (ep != r_nedge)
                        state <= S_READ;
                    else begin
                        // a pass over all rows is complete
                        ep <= {(EW+1){1'b0}};
                        row_start <= {(EW+1){1'b0}};
                        c2v_rp <= {CW{1'b0}};
                        c2v_wp <= {CW{1'b0}};
                        t <= 14'd0;
                        colbase <= {AW{1'b0}};
                        pw <= 5'd0;
                        plane <= 4'd0;
                        pi <= 9'd0;
                        if (synd) begin
                            synd <= 1'b0;
                            state <= synd_fail ? S_READ : S_OUT;
                        end else begin
                            it <= it + 5'd1;
                            if (it + 5'd1 == r_iter)
                                state <= S_OUT;
                            else begin
                                state <= S_READ;
                                synd <= r_estop;
                                synd_fail <= 1'b0;
                            end
                        end
                    end
                end
                S_OUT: begin
                    o_v    <= 1'b1;
                    o_lane <= plane;
                    o_idx  <= t;
                    t <= t + 14'd1;
                    if (pi == r_z - 9'd1) begin
                        colbase <= colbase + {{(AW-5){1'b0}}, r_q};
                        pw <= 5'd0;
                        plane <= 4'd0;
                        pi <= 9'd0;
                    end else begin
                        pi <= pi + 9'd1;
                        if (pw == r_q - 5'd1) begin
                            pw <= 5'd0;
                            plane <= plane + 4'd1;
                        end else
                            pw <= pw + 5'd1;
                    end
                    if (t == r_k - 14'd1)
                        state <= S_DONE;
                end
                default: if (!o_v) begin              // S_DONE: the last bit is out
                    state <= S_IDLE;
                    done  <= 1'b1;
                    iters <= it;
                end
            endcase
        end
    end
endmodule
