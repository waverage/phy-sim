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
// iteration takes about edges * Q + rows * (d + 7) clocks.
//
// The early stop runs beside the decoding: every APP write also writes its
// hard decisions to one of two hard-decision memories (iteration parity),
// so after iteration i the syndrome unit checks memory i mod 2 while
// iteration i + 1 goes on in the other. If the syndrome is zero, iteration
// i + 1 is abandoned and the bits of iteration i come out: the result and
// the iteration count of the sequential check, without its extra pass.
//
// The channel LLRs come from a word memory outside (lane layout, one word
// of 16 LLRs per clock, data one clock after the address): word
// llr_base + (c - 2) Q + w holds the transmitted LLRs of column c >= 2.
// The punctured columns 0 and 1 start at 0, the fillers k <= v < k_ldpc at
// +127, variables past the transmitted ones (v >= vend) at 0. The decoded
// bits leave as words in the same layout: word c Q + w, lane j is bit
// c z + j Q + w, out_mask the lanes below k.
//
// Configuration (cfg_*, from the control processor, harness/rtlcfg.py):
// registers at addresses 0..12, the edge table at 512 + e:
// {last in row, c Q, a, b}. A codeword: start with llr_base, the LLR reads
// (loaded when the last word is read), out_valid with the bit words, then
// done with the iteration count.
/* verilator lint_off UNUSEDSIGNAL */
module ldpc_dec #(
    parameter AW = 10,                 // APP words: dec_cols * Q <= 2^AW
    parameter CW = 12,                 // message words: edges * Q <= 2^CW
    parameter EW = 9,                  // edge table entries <= 2^EW
    parameter LW = 10                  // LLR memory word address bits
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              cfg_we,
    input  wire [9:0]        cfg_addr,
    input  wire [31:0]       cfg_data,
    input  wire              start,
    input  wire [LW-1:0]     llr_base,
    output wire              busy,
    output reg  [LW-1:0]     llr_addr,
    output wire              llr_re,
    input  wire [16*6-1:0]   llr_data,
    output reg               loaded,
    output wire              out_req,
    input  wire              out_ready,
    output reg               out_valid,
    output reg  [AW-1:0]     out_addr,
    output reg  [15:0]       out_word,
    output reg  [15:0]       out_mask,
    output reg               done,
    output reg  [4:0]        iters
);
    /* verilator lint_on UNUSEDSIGNAL */
    localparam P = 16;
    localparam A = 8;                  // APP bits
    localparam M = 6;                  // message bits
    localparam [4:0] MSG_MAX = 5'd31;

    // ---- configuration
    localparam R_Z = 4'd0, R_Q = 4'd1, R_K = 4'd2, R_KLDPC = 4'd3, R_VEND = 4'd4, R_Z2 = 4'd5, R_INITW = 4'd6,
               R_OUTW = 4'd7, R_NEDGE = 4'd8, R_ITER = 4'd9, R_OFFSET = 4'd10, R_ESTOP = 4'd11, R_COL2 = 4'd12;
    reg [8:0]    r_z;
    reg [4:0]    r_q;
    reg [13:0]   r_k, r_kldpc, r_vend, r_z2;
    reg [AW-1:0] r_initw, r_outw, r_col2;
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
                R_KLDPC:  r_kldpc  <= cfg_data[13:0];
                R_VEND:   r_vend   <= cfg_data[13:0];
                R_Z2:     r_z2     <= cfg_data[13:0];
                R_INITW:  r_initw  <= cfg_data[AW-1:0];
                R_OUTW:   r_outw   <= cfg_data[AW-1:0];
                R_NEDGE:  r_nedge  <= cfg_data[EW:0];
                R_ITER:   r_iter   <= cfg_data[4:0];
                R_OFFSET: r_off    <= cfg_data[4:0];
                R_ESTOP:  r_estop  <= cfg_data[0];
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
    reg            both_q;                     // the write is the load: both hard-decision memories
    reg            hpar_q;                     // else the one of this iteration parity
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

    // ---- the hard-decision memories, written with every APP word
    reg  [P-1:0]   hd0 [0:(1 << AW) - 1];
    reg  [P-1:0]   hd1 [0:(1 << AW) - 1];
    reg  [AW-1:0]  hd_raddr;
    reg  [P-1:0]   hd0_q, hd1_q;
    reg  [P-1:0]   hd_wdata;
    integer h;
    always @(*) begin
        for (h = 0; h < P; h = h + 1)
            hd_wdata[h] = app_wdata_q[h*A+A-1] || (app_wdata_q[h*A +: A] == {A{1'b0}});
    end
    always @(posedge clk) begin
        if (app_we_q != {P{1'b0}} && (both_q || !hpar_q))
            hd0[app_waddr_q] <= hd_wdata;
        hd0_q <= hd0[hd_raddr];
    end
    always @(posedge clk) begin
        if (app_we_q != {P{1'b0}} && (both_q || hpar_q))
            hd1[app_waddr_q] <= hd_wdata;
        hd1_q <= hd1[hd_raddr];
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
    function [P-1:0] rot_down1;                // the same on one bit a lane
        input [P-1:0] x;
        input [3:0]   r;
        integer m;
        reg   [3:0] s;
        begin
            for (m = 0; m < P; m = m + 1) begin
                s = m[3:0] + r;
                rot_down1[m] = x[s];
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
    localparam S_IDLE = 3'd0, S_LOAD = 3'd1, S_READ = 3'd2, S_DRAIN = 3'd3, S_ABORT = 3'd4, S_OUT = 3'd5,
               S_DONE = 3'd6;
    reg [2:0]    state;
    reg [4:0]    it;                           // the iteration being decoded
    reg [EW:0]   ep, row_start;
    reg [4:0]    w;
    reg [AW-1:0] cnt;                          // LOAD / OUT word
    reg [4:0]    pw;                           // its word in the column
    reg          ld_v;                         // LOAD: the word read last clock arrives
    reg [AW-1:0] ld_addr;
    reg          out_bank;                     // the hard decisions that come out
    reg [LW-1:0] r_base;

    assign busy = (state != S_IDLE);
    assign out_req = (state == S_OUT);

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
    reg          s1_v, s1_first, s1_last;
    reg [3:0]    s1_rot;
    reg [AW-1:0] s1_addr;
    reg [4:0]    s1_eidx;
    // stage 2: tt registered
    reg          s2_v, s2_first, s2_last;
    reg [3:0]    s2_rot;
    reg [AW-1:0] s2_addr;
    reg [4:0]    s2_eidx;
    reg [P*A-1:0] s2_tt;

    // stage 1b: the RAM outputs registered (the BRAMs' own output registers)
    reg          s1b_v, s1b_first, s1b_last;
    reg [3:0]    s1b_rot;
    reg [AW-1:0] s1b_addr;
    reg [4:0]    s1b_eidx;
    reg [P*A-1:0] app_q;
    reg [P*M-1:0] c2v_q;
    always @(posedge clk) begin
        s1b_v     <= s1_v && rst_n;
        s1b_first <= s1_first;
        s1b_last  <= s1_last;
        s1b_rot   <= s1_rot;
        s1b_addr  <= s1_addr;
        s1b_eidx  <= s1_eidx;
        app_q     <= app_dout;
        c2v_q     <= c2v_dout;
    end
    wire [P*A-1:0] v_rot = rot_down(app_q, s1b_rot);
    reg  [P*A-1:0] tt_c;
    reg  [A+1:0]   vv, cv;
    integer j;
    always @(*) begin
        for (j = 0; j < P; j = j + 1) begin
            vv = {{2{v_rot[j*A+A-1]}}, v_rot[j*A +: A]};
            cv = (it == 5'd0) ? {(A+2){1'b0}} : {{4{c2v_q[j*M+M-1]}}, c2v_q[j*M +: M]};
            tt_c[j*A +: A] = sat_app(vv - cv);
        end
    end

    // the group's running minimum (stage 2 -> accumulators)
    reg [5*P-1:0] min1, min2, idx;
    reg [P-1:0]   par;                         // sign parity
    reg [5*P-1:0] n_min1, n_min2, n_idx;
    reg [P-1:0]   n_par;
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
            end else begin
                n_par[j]  = par[j] ^ tv[A-1];
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
    wire          f_push = s2_v;
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
    reg  [P*M-1:0] w_msg;
    reg  [4:0]     w_new, w_m;
    always @(*) begin
        for (j = 0; j < P; j = j + 1) begin
            w_m = (w_eidx == w1_idx[j*5 +: 5]) ? w1_min2[j*5 +: 5] : w1_min1[j*5 +: 5];
            w_new = (w_m > r_off) ? w_m - r_off : 5'd0;
            w_msg[j*M +: M] = (w1_par[j] ^ w_tt[j*A+A-1]) ? -{1'b0, w_new} : {1'b0, w_new};
        end
    end

    // W1b: the messages registered (a clock of their own for 100 MHz with two decoders)
    reg            wb_v;
    reg  [P*A-1:0] wb_tt;
    reg  [P*M-1:0] wb_msg;
    reg  [AW-1:0]  wb_addr;
    reg  [3:0]     wb_rot;
    always @(posedge clk) begin
        wb_v    <= w1_v && rst_n;
        wb_tt   <= w_tt;
        wb_msg  <= w_msg;
        wb_addr <= w1_addr;
        wb_rot  <= w1_rot;
    end
    reg  [P*A-1:0] w_tn;
    always @(*) begin
        for (j = 0; j < P; j = j + 1)
            w_tn[j*A +: A] = sat_app({{2{wb_tt[j*A+A-1]}}, wb_tt[j*A +: A]} +
                                     {{4{wb_msg[j*M+M-1]}}, wb_msg[j*M +: M]});
    end

    // W2: the new APPs and messages registered; their rotation feeds the write registers
    reg            w2_v;
    reg  [P*A-1:0] w2_tn;
    reg  [P*M-1:0] w2_msg;
    reg  [AW-1:0]  w2_addr;
    reg  [3:0]     w2_rot;
    always @(posedge clk) begin
        w2_v    <= wb_v && rst_n;
        w2_tn   <= w_tn;
        w2_msg  <= wb_msg;
        w2_addr <= wb_addr;
        w2_rot  <= wb_rot;
    end

    wire writes_idle = !wr_act && !w1_v && !wb_v && !w2_v && (app_we_q == {P{1'b0}}) && !c2v_we_q;
    wire pipe_empty = !s1_v && !s1b_v && !s2_v && (f_wp == f_rp) && writes_idle;

    // ---- LOAD: per lane, the variable v = c z + j Q + w of the word decides
    // its start value. fl[j] follows v of lane j of word cnt.
    reg  [13:0]    fl [0:P-1];
    wire [13:0]    fstep = (pw == r_q - 5'd1) ? {5'd0, r_z} - {9'd0, r_q} + 14'd1 : 14'd1;
    reg  [1:0]     ld_cat [0:P-1];             // 0: zero, 1: -sat(llr), 2: +127
    reg  [P*A-1:0] ld_word;
    reg  [5:0]     lv;
    integer jl;
    always @(posedge clk) begin
        for (jl = 0; jl < P; jl = jl + 1) begin
            if (state == S_IDLE || (state == S_DRAIN && ep == r_nedge))
                fl[jl] <= r_q * jl[3:0];       // word 0, for the LOAD and the OUT
            else if (state == S_LOAD || (state == S_OUT && out_ready))
                fl[jl] <= fl[jl] + fstep;
            if (state == S_LOAD)
                ld_cat[jl] <= (fl[jl] < r_z2 || fl[jl] >= r_vend) ? 2'd0 :
                              (fl[jl] >= r_k && fl[jl] < r_kldpc) ? 2'd2 : 2'd1;
        end
    end
    always @(*) begin
        for (j = 0; j < P; j = j + 1) begin
            lv = (llr_data[j*6 +: 6] == 6'h20) ? 6'h21 : llr_data[j*6 +: 6];
            case (ld_cat[j])
                2'd1:    ld_word[j*A +: A] = -{{2{lv[5]}}, lv};
                2'd2:    ld_word[j*A +: A] = 8'd127;
                default: ld_word[j*A +: A] = 8'd0;
            endcase
        end
    end
    assign llr_re = (state == S_LOAD) && (cnt >= r_col2);
    always @(*) llr_addr = r_base + cnt[LW-1:0] - r_col2[LW-1:0];

    // ---- the syndrome unit: a walk of the edges over one hard-decision
    // memory, one word a clock, no drain (it only reads)
    reg          sy_run, sy_bank, sy_fail;
    reg          sy_done, sy_ok;               // sy_done: one clock, with the result
    reg [EW:0]   sy_ep, sy_rs;
    reg [4:0]    sy_w;
    wire [AW+9:0] sy_ent = etab[sy_ep[EW-1:0]];
    wire          sy_last = sy_ent[AW+9];
    wire [5:0]    sy_wa = {1'b0, sy_w} + {1'b0, sy_ent[8:4]};
    wire          sy_wrap = (sy_wa >= {1'b0, r_q});
    wire [4:0]    sy_wsel = sy_wrap ? sy_wa[4:0] - r_q : sy_wa[4:0];
    wire [AW-1:0] sy_addr = sy_ent[AW+8:9] + {{(AW-5){1'b0}}, sy_wsel};
    reg           y1_v, y1_first, y1_last, y1_end;
    reg  [3:0]    y1_rot;
    reg  [P-1:0]  sy_acc;
    wire [P-1:0]  y1_hd = rot_down1(sy_bank ? hd1_q : hd0_q, y1_rot);
    wire [P-1:0]  sy_next = (y1_first ? {P{1'b0}} : sy_acc) ^ y1_hd;
    reg           sy_start;
    reg           sy_first;                    // the next read starts a group

    always @(*) hd_raddr = (state == S_OUT) ? cnt : sy_addr;

    always @(posedge clk) begin
        if (!rst_n) begin
            sy_run <= 1'b0;
            y1_v <= 1'b0;
            sy_done <= 1'b0;
        end else begin
            sy_done <= 1'b0;
            y1_v     <= sy_run;
            y1_first <= sy_first;
            y1_last  <= sy_last;
            y1_rot   <= sy_wrap ? sy_ent[3:0] + 4'd1 : sy_ent[3:0];
            y1_end   <= sy_run && sy_last && sy_w == r_q - 5'd1 && sy_ep + 1'b1 == r_nedge;
            if (sy_start) begin
                sy_run <= 1'b1;
                sy_fail <= 1'b0;
                sy_ep <= {(EW+1){1'b0}};
                sy_rs <= {(EW+1){1'b0}};
                sy_w <= 5'd0;
                sy_first <= 1'b1;
            end else if (sy_run) begin
                sy_first <= sy_last;
                if (sy_last) begin
                    if (sy_w == r_q - 5'd1) begin
                        sy_w <= 5'd0;
                        sy_ep <= sy_ep + 1'b1;
                        sy_rs <= sy_ep + 1'b1;
                        if (sy_ep + 1'b1 == r_nedge)
                            sy_run <= 1'b0;
                    end else begin
                        sy_w <= sy_w + 5'd1;
                        sy_ep <= sy_rs;
                    end
                end else
                    sy_ep <= sy_ep + 1'b1;
            end
            if (y1_v && !sy_fail) begin
                sy_acc <= sy_next;
                if (y1_last && sy_next != {P{1'b0}}) begin
                    sy_fail <= 1'b1;
                    sy_run <= 1'b0;            // one failing check is enough
                    sy_done <= 1'b1;
                    sy_ok <= 1'b0;
                end else if (y1_end) begin
                    sy_done <= 1'b1;
                    sy_ok <= 1'b1;
                end
            end
            if (state == S_IDLE)
                sy_run <= 1'b0;
        end
    end
    wire sy_busy = sy_run || y1_v || sy_start;
    wire sy_pass = sy_done && sy_ok;

    // ---- the memories' ports, from the state
    always @(*) begin
        app_raddr = rd_addr;
        app_waddr = w2_addr;
        app_we    = {P{1'b0}};
        app_wdata = rot_up(w2_tn, w2_rot);
        c2v_we    = 1'b0;
        c2v_wdata = w2_msg;
        if (ld_v) begin
            app_waddr = ld_addr;
            app_we    = {P{1'b1}};
            app_wdata = ld_word;
        end else if (state == S_READ || state == S_DRAIN) begin
            app_we = w2_v ? {P{1'b1}} : {P{1'b0}};
            c2v_we = w2_v;
        end
    end

    // OUT: the word read in the previous clock, and its lanes below k
    reg          o_v;
    reg [AW-1:0] o_addr;
    reg [P-1:0]  o_mask;
    integer jo;

    always @(posedge clk) begin
        both_q <= ld_v;
        hpar_q <= it[0];
        if (!rst_n) begin
            state <= S_IDLE;
            s1_v <= 1'b0;
            s2_v <= 1'b0;
            o_v <= 1'b0;
            ld_v <= 1'b0;
            wr_rem <= 5'd0;
            f_wp <= 5'd0;
            f_rp <= 5'd0;
            out_valid <= 1'b0;
            loaded <= 1'b0;
            done <= 1'b0;
            iters <= 5'd0;
            sy_start <= 1'b0;
        end else begin
            done <= 1'b0;
            loaded <= 1'b0;
            out_valid <= 1'b0;
            sy_start <= 1'b0;

            // stage 1 <- issue
            s1_v     <= issue;
            s1_first <= (ep == row_start);
            s1_last  <= e_last;
            s1_rot   <= rd_rot;
            s1_addr  <= rd_addr;
            s1_eidx  <= eidx;
            // stage 2 <- stage 1
            s2_v     <= s1b_v;
            s2_first <= s1b_first;
            s2_last  <= s1b_last;
            s2_rot   <= s1b_rot;
            s2_addr  <= s1b_addr;
            s2_eidx  <= s1b_eidx;
            s2_tt    <= tt_c;
            // accumulators <- stage 2
            if (s2_v) begin
                min1 <= n_min1;
                min2 <= n_min2;
                idx  <= n_idx;
                par  <= n_par;
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
            if (s2_v && s2_last) begin
                q_min1 <= n_min1;
                q_min2 <= n_min2;
                q_idx  <= n_idx;
                q_par  <= n_par;
                wr_rem <= s2_eidx + 5'd1;
            end else if (wr_act)
                wr_rem <= wr_rem - 5'd1;

            // LOAD: the word read last clock
            ld_v <= 1'b0;
            ld_addr <= cnt;

            // OUT stage 2
            o_v <= 1'b0;
            if (o_v) begin
                out_valid <= 1'b1;
                out_word  <= out_bank ? hd1_q : hd0_q;
                out_addr  <= o_addr;
                out_mask  <= o_mask;
            end

            case (state)
                S_IDLE: if (start) begin
                    state  <= S_LOAD;
                    cnt    <= {AW{1'b0}};
                    pw     <= 5'd0;
                    r_base <= llr_base;
                end
                S_LOAD: begin
                    ld_v <= 1'b1;
                    cnt <= cnt + 1'b1;
                    pw <= (pw == r_q - 5'd1) ? 5'd0 : pw + 5'd1;
                    if (cnt == r_initw - 1'b1) begin
                        loaded <= 1'b1;
                        state <= S_DRAIN;            // the last write lands, then the first row
                        it <= 5'd0;
                        ep <= {(EW+1){1'b0}};
                        row_start <= {(EW+1){1'b0}};
                        w <= 5'd0;
                        c2v_rp <= {CW{1'b0}};
                        c2v_wp <= {CW{1'b0}};
                    end
                end
                S_READ: begin
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
                S_DRAIN: if (pipe_empty && !ld_v) begin
                    if (ep != r_nedge)
                        state <= S_READ;
                    else if (!sy_busy) begin
                        // iteration it is complete; the check of it - 1 failed
                        ep <= {(EW+1){1'b0}};
                        row_start <= {(EW+1){1'b0}};
                        c2v_rp <= {CW{1'b0}};
                        c2v_wp <= {CW{1'b0}};
                        if (it + 5'd1 == r_iter) begin
                            state <= S_OUT;
                            out_bank <= it[0];
                            iters <= it + 5'd1;
                            cnt <= {AW{1'b0}};
                            pw <= 5'd0;
                        end else begin
                            state <= S_READ;
                            it <= it + 5'd1;
                            if (r_estop) begin
                                sy_start <= 1'b1;
                                sy_bank <= it[0];
                            end
                        end
                    end
                end
                S_ABORT: if (!s1_v && !s1b_v && !s2_v && writes_idle) begin
                    // a group cut short left its reads in the FIFO
                    f_rp <= f_wp;
                    state <= S_OUT;
                    cnt <= {AW{1'b0}};
                    pw <= 5'd0;
                end
                S_OUT: if (out_ready) begin
                    o_v    <= 1'b1;
                    o_addr <= cnt;
                    for (jo = 0; jo < P; jo = jo + 1)
                        o_mask[jo] <= (fl[jo] < r_k);
                    cnt <= cnt + 1'b1;
                    pw <= (pw == r_q - 5'd1) ? 5'd0 : pw + 5'd1;
                    if (cnt == r_outw - 1'b1)
                        state <= S_DONE;
                end
                default: if (!o_v) begin              // S_DONE: the last word is out
                    state <= S_IDLE;
                    done  <= 1'b1;
                end
            endcase
            // the check of iteration it - 1 passed: its bits, without finishing it
            if (sy_pass && (state == S_READ || state == S_DRAIN)) begin
                state <= S_ABORT;
                out_bank <= sy_bank;
                iters <= it;
            end
        end
    end
endmodule
