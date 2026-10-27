`timescale 1ns / 1ps
// The frame transmitter (M9): the preamble and fxmodem.FxModem(true).tx in
// RTL, bit exact against twin.TwinSync.preamble, the LDPC encoder
// (modem/src/ldpc.zig, as rtlcfg.ldpc_enc_program) and twin.TwinModem.tx.
//
// While `go` is high, frames follow each other without gaps: the preamble
// (288 samples from a table), then S OFDM symbols of N + cp samples,
// 12-bit I/Q at out_* (out_ready paces them: the DAC's rate). Frame r's
// information bits come from outside: info_frame = r, info_addr =
// c OUTW + w (codeword c, word w of rtl/ldpc_dec's output layout), the
// word a clock later.
//
//   encoder  per codeword: the information words into the codeword memory,
//            then the program, one 16-lane word a clock: dst = XOR of
//            source vectors read shifted (the decoder's word layout and
//            rotations). Columns >= 2 live in a bank at the rx_frame LLR
//            layout, so the receiver's deinterleaver table (rtlcfg.rx_perm)
//            addresses the transmitted bits too; columns 0, 1 and the
//            accumulator in a scratch memory. Two banks: one is encoded
//            while the other is sent.
//   mapper   per symbol, one FFT bin a clock: zero, a pilot (sign times
//            tx_pilot), or a data RE's B bits through perm and the bank
//            (a clock a bit) into the I and Q levels; into one of two grids.
//   fft256   inverse, a grid -> one of two symbol buffers.
//   emitter  per frame the preamble, then each symbol buffer from N - cp
//            (the cyclic prefix, then the symbol), saturated to 12 bits.
//
// Configuration (cfg_*, rtlcfg.tx_cfg): registers 0 S, 1 F, 2 npsym,
// 3-5 pilot symbols, 6 B, 7 codewords, 8 LLR-layout words a codeword,
// 9 Q, 10 OUTW, 11 k, 12 z, 13 k_b Q, 14 tx_pilot, 15 cp, 16 guard_lo,
// 17 guard_hi, 18 dc_null, 19 program ops; 0x40 + i the levels; 0x400 + o
// the ops' {destination, first source}; 0x800 + e the sources {last of the
// op, a, b, vector}; vectors are {scratch, word base} (rtlcfg.tx_cfg);
// 0xc00 + p the pilot signs in order; 0x1000 + i the preamble {re, im};
// 0x4000 + i perm {discard, word, lane}.
module tx_frame (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               cfg_we,
    input  wire        [15:0] cfg_addr,
    input  wire        [31:0] cfg_data,
    input  wire               go,
    output wire               idle,
    output reg         [31:0] info_frame,
    output wire        [10:0] info_addr,
    input  wire        [15:0] info_data,
    output reg                out_valid,
    output reg  signed [11:0] out_re,
    output reg  signed [11:0] out_im,
    output reg                out_first,       // the first preamble sample of a frame
    input  wire               out_ready,
    output reg         [31:0] n_frames         // frames whose last sample went out
);
    localparam P = 16;

    // ---- registers
    reg [3:0]  r_nsym, r_psym0, r_psym1, r_psym2;
    reg [7:0]  r_nf, r_glo, r_ghi;
    reg [1:0]  r_npsym, r_ncw;
    reg [2:0]  r_bps;
    reg [9:0]  r_cww, r_outw, r_kbq;
    reg [4:0]  r_q;
    reg [13:0] r_k;
    reg [8:0]  r_z;
    reg [11:0] r_pilot;
    reg [5:0]  r_cp, r_nops;
    reg        r_dc;
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:6] == 10'd0) begin
            case (cfg_addr[5:0])
                6'd0:  r_nsym  <= cfg_data[3:0];
                6'd1:  r_nf    <= cfg_data[7:0];
                6'd2:  r_npsym <= cfg_data[1:0];
                6'd3:  r_psym0 <= cfg_data[3:0];
                6'd4:  r_psym1 <= cfg_data[3:0];
                6'd5:  r_psym2 <= cfg_data[3:0];
                6'd6:  r_bps   <= cfg_data[2:0];
                6'd7:  r_ncw   <= cfg_data[1:0];
                6'd8:  r_cww   <= cfg_data[9:0];
                6'd9:  r_q     <= cfg_data[4:0];
                6'd10: r_outw  <= cfg_data[9:0];
                6'd11: r_k     <= cfg_data[13:0];
                6'd12: r_z     <= cfg_data[8:0];
                6'd13: r_kbq   <= cfg_data[9:0];
                6'd14: r_pilot <= cfg_data[11:0];
                6'd15: r_cp    <= cfg_data[5:0];
                6'd16: r_glo   <= cfg_data[7:0];
                6'd17: r_ghi   <= cfg_data[7:0];
                6'd18: r_dc    <= cfg_data[0];
                6'd19: r_nops  <= cfg_data[5:0];
                default: ;
            endcase
        end
    end

    // ---- tables
    reg signed [11:0] t_lv [0:7];
    reg        [20:0] t_op [0:63];             // {destination [11:0], first source [8:0]}
    reg        [21:0] t_src [0:511];           // {last, a[4:0], b[3:0], vector [11:0]}
    reg        [1:0]  t_sg [0:1023];
    reg        [23:0] m_pre [0:511];
    reg        [14:0] m_perm [0:16383];
    reg        [23:0] pre_q;
    reg        [14:0] perm_q;
    reg        [8:0]  pre_ra;
    reg        [13:0] perm_ra;
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:3] == 13'h0008)
            t_lv[cfg_addr[2:0]] <= cfg_data[11:0];
        if (cfg_we && cfg_addr[15:6] == 10'h010)
            t_op[cfg_addr[5:0]] <= cfg_data[20:0];
        if (cfg_we && cfg_addr[15:9] == 7'h04)
            t_src[cfg_addr[8:0]] <= cfg_data[21:0];
        if (cfg_we && cfg_addr[15:10] == 6'h03)
            t_sg[cfg_addr[9:0]] <= cfg_data[1:0];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:9] == 7'h08)
            m_pre[cfg_addr[8:0]] <= cfg_data[23:0];
        pre_q <= m_pre[pre_ra];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:14] == 2'b01)
            m_perm[cfg_addr[13:0]] <= cfg_data[14:0];
        perm_q <= m_perm[perm_ra];
    end

    function [P-1:0] rot_down1;                // out lane j = in lane (j + r) mod P
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

    // ---- the codeword memories: two banks (columns >= 2 of all codewords
    // of a frame, the LLR layout) and the scratch (columns 0, 1 and the
    // accumulator of the codeword being encoded). A vector address is
    // {scratch, word}: the scratch word, or the word within the codeword.
    reg  [15:0] m_bk0 [0:1023];
    reg  [15:0] m_bk1 [0:1023];
    reg  [15:0] m_scr [0:127];
    reg         cw_we;
    reg  [11:0] cw_wa;                         // {scratch, word} in the bank being encoded
    reg  [15:0] cw_wd;
    reg  [9:0]  bk0_ra, bk1_ra;
    reg  [6:0]  scr_ra;
    reg  [15:0] bk0_q, bk1_q, scr_q;
    reg         ebank;                         // the bank being encoded
    reg         mbank;                         // the bank being sent
    always @(posedge clk) begin
        if (cw_we && !cw_wa[11] && !ebank)
            m_bk0[cw_wa[9:0]] <= cw_wd;
        bk0_q <= m_bk0[bk0_ra];
    end
    always @(posedge clk) begin
        if (cw_we && !cw_wa[11] && ebank)
            m_bk1[cw_wa[9:0]] <= cw_wd;
        bk1_q <= m_bk1[bk1_ra];
    end
    always @(posedge clk) begin
        if (cw_we && cw_wa[11])
            m_scr[cw_wa[6:0]] <= cw_wd;
        scr_q <= m_scr[scr_ra];
    end

    // ================= the encoder =================
    localparam E_IDLE = 3'd0, E_LOAD = 3'd1, E_DRAIN = 3'd2, E_OP = 3'd3, E_NEXT = 3'd4;
    reg  [2:0]  est;
    reg  [1:0]  ecw;
    reg  [9:0]  ecbase;                        // the codeword's base in the bank
    reg  [9:0]  ea;                            // LOAD: the word c Q + w
    reg  [4:0]  ew;                            // its word in the column
    reg  [13:0] fl [0:P-1];                    // LOAD: the variable of each lane
    reg  [5:0]  eop;
    reg  [8:0]  esrc;
    reg  [4:0]  eopw;                          // OP: the destination word
    reg  [1:0]  bank_full;                     // a bank holds an encoded frame
    reg         bank_free, bank_free_i;        // the mapper is done with bank bank_free_i
    reg  [31:0] enc_frame;
    wire [20:0] op = t_op[eop];
    wire [21:0] src = t_src[esrc];
    wire        src_last = src[21];
    wire [5:0]  wa_e = {1'b0, eopw} + {1'b0, src[20:16]};
    wire        wrap_e = (wa_e >= {1'b0, r_q});
    wire [4:0]  wsel_e = wrap_e ? wa_e[4:0] - r_q : wa_e[4:0];
    // a vector's word: in the scratch, or in the bank after the codeword's base
    // (bit 10 of a vector address is spare)
    /* verilator lint_off UNUSEDSIGNAL */
    function [11:0] vword;
        input [11:0] v;
        input [9:0]  base;
        input [4:0]  w;
        begin
            vword = v[11] ? {2'b10, v[9:0] + {5'd0, w}} : {2'b00, base + v[9:0] + {5'd0, w}};
        end
    endfunction
    wire [11:0] src_addr = vword(src[11:0], ecbase, wsel_e);
    // F is implied by the guards; the top configuration bits are not used
    wire        cfg_unused = &{1'b0, cfg_data[31:24], r_nf, cw_wa[10], src_addr[10]};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [9:0]  r_q2 = {4'd0, r_q, 1'b0};
    assign info_addr = ((ecw == 2'd0) ? 11'd0 : (ecw == 2'd1) ? {1'b0, r_outw} : {r_outw, 1'b0}) + {1'b0, ea};
    // L1: the info word / the source word arrives
    reg         ld_v, ld_ok;
    reg  [11:0] ld_wa;
    reg  [15:0] ld_mask;
    reg         op_v, op_first, op_last, op_scr;
    reg  [3:0]  op_rot;
    reg  [11:0] op_dst;
    reg  [15:0] x_acc;
    wire [15:0] e_word = op_scr ? scr_q : (ebank ? bk1_q : bk0_q);
    wire [15:0] x_new = (op_first ? 16'd0 : x_acc) ^ rot_down1(e_word, op_rot);
    reg         op_newgrp;                     // the next source read starts a destination word
    integer jl;
    always @(posedge clk) begin
        for (jl = 0; jl < P; jl = jl + 1) begin
            if (est != E_LOAD)
                fl[jl] <= r_q * jl[3:0];
            else
                fl[jl] <= fl[jl] + ((ew == r_q - 5'd1) ? {5'd0, r_z} - {9'd0, r_q} + 14'd1 : 14'd1);
            ld_mask[jl] <= (fl[jl] < r_k);
        end
        ld_ok <= (ea < r_outw);
        ld_wa <= (ea < r_q2) ? {2'b10, ea} : {2'b00, ecbase + ea - r_q2};
        op_first <= op_newgrp;
        op_last <= src_last;
        op_scr <= src_addr[11];
        op_rot <= wrap_e ? src[15:12] + 4'd1 : src[15:12];
        op_dst <= vword(op[20:9], ecbase, eopw);
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            est <= E_IDLE;
            ebank <= 1'b0;
            bank_full <= 2'b00;
            enc_frame <= 32'd0;
            info_frame <= 32'hffffffff;        // not a frame yet: frame 0 is a change
            ld_v <= 1'b0;
            op_v <= 1'b0;
            cw_we <= 1'b0;
        end else begin
            cw_we <= 1'b0;
            ld_v <= (est == E_LOAD);
            op_v <= (est == E_OP);
            if (ld_v) begin
                cw_we <= 1'b1;
                cw_wa <= ld_wa;
                cw_wd <= ld_ok ? info_data & ld_mask : 16'd0;
            end
            if (op_v) begin
                x_acc <= x_new;
                if (op_last) begin
                    cw_we <= 1'b1;
                    cw_wa <= op_dst;
                    cw_wd <= x_new;
                end
            end
            case (est)
                E_IDLE: if (go && !bank_full[ebank]) begin
                    est <= E_LOAD;
                    ecw <= 2'd0;
                    ecbase <= 10'd0;
                    ea <= 10'd0;
                    ew <= 5'd0;
                    info_frame <= enc_frame;
                end
                E_LOAD: begin
                    ea <= ea + 10'd1;
                    ew <= (ew == r_q - 5'd1) ? 5'd0 : ew + 5'd1;
                    if (ea == r_kbq - 10'd1) begin
                        est <= E_DRAIN;
                        eop <= 6'd0;
                    end
                end
                E_DRAIN: if (!ld_v && !op_v && !cw_we) begin   // the last write is in
                    if (eop == r_nops)
                        est <= E_NEXT;
                    else begin
                        est <= E_OP;
                        esrc <= op[8:0];
                        eopw <= 5'd0;
                        op_newgrp <= 1'b1;
                    end
                end
                E_OP: begin
                    op_newgrp <= src_last;
                    if (src_last) begin
                        if (eopw == r_q - 5'd1) begin
                            eop <= eop + 6'd1;
                            est <= E_DRAIN;
                        end else begin
                            eopw <= eopw + 5'd1;
                            esrc <= op[8:0];
                        end
                    end else
                        esrc <= esrc + 9'd1;
                end
                default: begin                         // E_NEXT
                    if (ecw == r_ncw - 2'd1) begin
                        bank_full[ebank] <= 1'b1;
                        ebank <= !ebank;
                        enc_frame <= enc_frame + 32'd1;
                        est <= E_IDLE;
                    end else begin
                        ecw <= ecw + 2'd1;
                        ecbase <= ecbase + r_cww;
                        ea <= 10'd0;
                        ew <= 5'd0;
                        est <= E_LOAD;
                    end
                end
            endcase
            if (bank_free)
                bank_full[bank_free_i] <= 1'b0;
        end
    end

    // ---- the hand-overs between the stages
    reg         gr;                            // the grid the FFT takes next
    reg         g_taken;                       // the FFT has read grid gr
    reg         orr;                           // the symbol buffer the emitter sends
    reg         o_free, o_free_i;              // the emitter is done with buffer o_free_i

    // ================= the mapper =================
    localparam M_IDLE = 2'd0, M_RUN = 2'd1, M_WAIT = 2'd2, M_HOLD = 2'd3;
    reg  [1:0]  mst;
    reg  [3:0]  ms;                            // symbol
    reg  [8:0]  mi;                            // natural index, to 256
    reg  [2:0]  mb;                            // a data RE's bit
    reg  [9:0]  mp;                            // pilot RE, in order
    reg  [13:0] mperm;                         // perm index d B + b
    reg  [1:0]  gfull;                         // the grids: full, waiting for the FFT
    reg         gfirst [0:1];                  // the grid is a frame's first symbol
    reg         glast [0:1];                   // or its last
    reg         gw;                            // the grid being filled
    wire [7:0]  mi8 = mi[7:0];
    wire        m_used = (mi8 >= r_glo) && (mi8 < 8'd0 - r_ghi) && !(r_dc && mi8 == 8'h80);
    wire        m_pilot = (ms == r_psym0) || (ms == r_psym1) || (r_npsym == 2'd3 && ms == r_psym2);
    wire        m_data = m_used && !m_pilot;
    wire [15:0] pil = {{4{r_pilot[11]}}, r_pilot};
    reg  [31:0] m_g [0:511];                   // two grids of 256 bins, {re, im}
    reg         g_we;
    reg  [8:0]  g_wa;
    reg  [31:0] g_wd;
    reg  [8:0]  g_ra;
    reg  [31:0] g_q;
    always @(posedge clk) begin
        if (g_we)
            m_g[g_wa] <= g_wd;
        g_q <= m_g[g_ra];
    end
    // the encoder reads its bank only while it runs: then it is not the mapper's
    wire        e_rd = (est == E_OP);
    always @(*) begin
        perm_ra = mperm;
        bk0_ra = (e_rd && ebank == 1'b0) ? src_addr[9:0] : perm_q[13:4];
        bk1_ra = (e_rd && ebank == 1'b1) ? src_addr[9:0] : perm_q[13:4];
        scr_ra = src_addr[6:0];
    end
    // the bit pipeline: B1 perm arrives (the bank read goes out), B2 the bank word
    reg        b1_v, b1_last, b2_v, b2_last;
    reg [2:0]  b1_b, b2_b;
    reg [7:0]  b1_bin, b2_bin;
    reg        b2_disc;
    reg [3:0]  b2_lane;
    reg [5:0]  rbits;                          // the RE's bits so far, bit b at b
    wire [15:0] m_word = mbank ? bk1_q : bk0_q;
    wire       bit_now = !b2_disc && m_word[b2_lane];
    reg  [5:0] all_bits;
    reg  [2:0] li, lq;
    integer bj;
    always @(*) begin
        all_bits = rbits;
        all_bits[b2_b] = bit_now;
        li = 3'd0;
        lq = 3'd0;
        for (bj = 0; bj < 3; bj = bj + 1)
            if (bj < {29'd0, r_bps[2:1]}) begin
                li = {li[1:0], all_bits[2 * bj]};
                lq = {lq[1:0], all_bits[2 * bj + 1]};
            end
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            mst <= M_IDLE;
            mbank <= 1'b0;
            gfull <= 2'b00;
            gw <= 1'b0;
            b1_v <= 1'b0;
            b2_v <= 1'b0;
            g_we <= 1'b0;
            bank_free <= 1'b0;
        end else begin
            g_we <= 1'b0;
            bank_free <= 1'b0;
            b1_v <= 1'b0;
            b2_v <= b1_v;
            b2_last <= b1_last;
            b2_b <= b1_b;
            b2_bin <= b1_bin;
            b2_disc <= perm_q[14];
            b2_lane <= perm_q[3:0];
            if (b2_v) begin
                rbits <= all_bits;
                if (b2_last) begin
                    g_we <= 1'b1;
                    g_wa <= {gw, b2_bin};
                    g_wd <= {{4{t_lv[li][11]}}, t_lv[li], {4{t_lv[lq][11]}}, t_lv[lq]};
                end
            end
            case (mst)
                M_IDLE: if (bank_full[mbank] && !gfull[gw]) begin
                    mst <= M_RUN;
                    ms <= 4'd0;
                    mi <= 9'd0;
                    mb <= 3'd0;
                    mp <= 10'd0;
                    mperm <= 14'd0;
                end
                M_RUN: if (mi == 9'd256) begin
                    if (!b1_v && !b2_v && !g_we)
                        mst <= M_WAIT;
                end else if (m_data) begin           // one bit a clock
                    b1_v <= 1'b1;
                    b1_b <= mb;
                    b1_last <= (mb == r_bps - 3'd1);
                    b1_bin <= mi8 ^ 8'h80;
                    mperm <= mperm + 14'd1;
                    if (mb == r_bps - 3'd1) begin
                        mb <= 3'd0;
                        mi <= mi + 9'd1;
                    end else
                        mb <= mb + 3'd1;
                end else if (!(b2_v && b2_last)) begin   // a pilot or an empty bin, not over a data RE's write
                    g_we <= 1'b1;
                    g_wa <= {gw, mi8 ^ 8'h80};
                    g_wd <= !m_used ? 32'd0 :
                            {t_sg[mp][1] ? -pil : pil, t_sg[mp][0] ? -pil : pil};
                    if (m_used)
                        mp <= mp + 10'd1;
                    mi <= mi + 9'd1;
                end
                M_WAIT: begin                        // the grid is complete
                    gfull[gw] <= 1'b1;
                    gfirst[gw] <= (ms == 4'd0);
                    glast[gw] <= (ms == r_nsym - 4'd1);
                    gw <= !gw;
                    if (ms == r_nsym - 4'd1) begin
                        mst <= M_IDLE;
                        bank_free <= 1'b1;
                        bank_free_i <= mbank;
                        mbank <= !mbank;
                    end else begin
                        ms <= ms + 4'd1;
                        mi <= 9'd0;
                        mst <= M_HOLD;
                    end
                end
                default: if (!gfull[gw])             // M_HOLD: the next grid is free
                    mst <= M_RUN;
            endcase
            if (g_taken)
                gfull[gr] <= 1'b0;
        end
    end

    // ================= the inverse FFT =================
    localparam F_IDLE = 2'd0, F_LOAD = 2'd1, F_OUT = 2'd2;
    reg  [1:0]  fst;
    reg  [8:0]  fcnt;
    reg         f_in;
    reg  [1:0]  ofull;                         // the symbol buffers: full, waiting for the emitter
    reg         ofirst [0:1];
    reg         olast [0:1];
    reg         ow;                            // the buffer the FFT fills next
    wire        fft_rdy, fft_ov;
    wire signed [15:0] fft_ore, fft_oim;
    wire [7:0]  fft_oidx;
    fft256 fft (
        .clk(clk), .rst_n(rst_n), .inverse(1'b1), .in_valid(f_in), .in_re(g_q[31:16]), .in_im(g_q[15:0]),
        .in_ready(fft_rdy), .out_valid(fft_ov), .out_re(fft_ore), .out_im(fft_oim), .out_idx(fft_oidx));
    reg  [31:0] m_o [0:511];                   // two symbol buffers
    reg  [8:0]  o_ra;
    reg  [31:0] o_q;
    always @(posedge clk) begin
        if (fft_ov)
            m_o[{ow, fft_oidx}] <= {fft_ore, fft_oim};
        o_q <= m_o[o_ra];
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            fst <= F_IDLE;
            gr <= 1'b0;
            ow <= 1'b0;
            ofull <= 2'b00;
            f_in <= 1'b0;
            g_taken <= 1'b0;
        end else begin
            f_in <= 1'b0;
            g_taken <= 1'b0;
            case (fst)
                F_IDLE: if (gfull[gr] && !ofull[ow]) begin
                    fst <= F_LOAD;
                    fcnt <= 9'd0;
                    ofirst[ow] <= gfirst[gr];
                    olast[ow] <= glast[gr];
                end
                F_LOAD: if (fft_rdy) begin            // read the bin, it goes in a clock later
                    if (fcnt != 9'd256) begin
                        g_ra <= {gr, fcnt[7:0]};
                        fcnt <= fcnt + 9'd1;
                    end
                    f_in <= (fcnt != 9'd0) && (fcnt != 9'd257);
                    if (fcnt == 9'd256) begin
                        fcnt <= 9'd257;
                    end
                    if (fcnt == 9'd257) begin
                        g_taken <= 1'b1;
                        fst <= F_OUT;
                    end
                end
                default: if (fft_ov && fft_oidx == 8'd255) begin   // F_OUT
                    ofull[ow] <= 1'b1;
                    ow <= !ow;
                    gr <= !gr;
                    fst <= F_IDLE;
                end
            endcase
            if (o_free)
                ofull[o_free_i] <= 1'b0;
        end
    end

    // ================= the emitter =================
    // a sample: its address registered (R1), the memory's word (R2), the
    // output register; one at a time, so the output register is free
    localparam X_IDLE = 2'd0, X_PRE = 2'd1, X_SYM = 2'd2;
    reg  [1:0]  xst;
    reg  [8:0]  xi;
    reg         rd_v, rd_v2;                   // a read is in flight
    reg         rd_first, rd_pre, rd_end, rd2_first, rd2_pre, rd2_end;
    wire        take = (!out_valid || out_ready) && !rd_v && !rd_v2;
    wire signed [15:0] o_re = o_q[31:16], o_im = o_q[15:0];
    function signed [11:0] sat12;
        input signed [15:0] v;
        begin
            sat12 = (v > 16'sd2047) ? 12'sd2047 : (v < -16'sd2048) ? -12'sd2048 : v[11:0];
        end
    endfunction
    wire [8:0]  sym_len = 9'd256 + {3'd0, r_cp};
    always @(posedge clk) begin
        if (!rst_n) begin
            xst <= X_IDLE;
            orr <= 1'b0;
            out_valid <= 1'b0;
            rd_v <= 1'b0;
            rd_v2 <= 1'b0;
            o_free <= 1'b0;
            n_frames <= 32'd0;
        end else begin
            o_free <= 1'b0;
            if (out_valid && out_ready)
                out_valid <= 1'b0;
            rd_v <= 1'b0;
            rd_v2 <= rd_v;
            rd2_first <= rd_first;
            rd2_pre <= rd_pre;
            rd2_end <= rd_end;
            if (rd_v2) begin
                out_valid <= 1'b1;
                out_first <= rd2_first;
                out_re <= rd2_pre ? pre_q[23:12] : sat12(o_re);
                out_im <= rd2_pre ? pre_q[11:0] : sat12(o_im);
                if (rd2_end)
                    n_frames <= n_frames + 32'd1;
            end
            case (xst)
                X_IDLE: if (ofull[orr]) begin
                    xst <= ofirst[orr] ? X_PRE : X_SYM;
                    xi <= 9'd0;
                end
                X_PRE: if (take) begin
                    pre_ra <= xi;
                    rd_v <= 1'b1;
                    rd_pre <= 1'b1;
                    rd_first <= (xi == 9'd0);
                    rd_end <= 1'b0;
                    xi <= xi + 9'd1;
                    if (xi == 9'd287) begin
                        xi <= 9'd0;
                        xst <= X_SYM;
                    end
                end
                default: if (take) begin             // X_SYM: the CP (from N - cp), then the symbol
                    o_ra <= {orr, xi[7:0] - {2'b00, r_cp}};
                    rd_v <= 1'b1;
                    rd_pre <= 1'b0;
                    rd_first <= 1'b0;
                    rd_end <= olast[orr] && (xi == sym_len - 9'd1);
                    xi <= xi + 9'd1;
                    if (xi == sym_len - 9'd1) begin
                        o_free <= 1'b1;

                        o_free_i <= orr;
                        orr <= !orr;
                        xst <= X_IDLE;
                    end
                end
            endcase
        end
    end

    assign idle = (est == E_IDLE) && bank_full == 2'b00 && mst == M_IDLE && gfull == 2'b00 && fst == F_IDLE &&
                  ofull == 2'b00 && xst == X_IDLE && !out_valid && !rd_v && !rd_v2;
endmodule
