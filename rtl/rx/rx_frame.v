`timescale 1ns / 1ps
// The frame receiver: fxmodem.FxModem(true).rx in RTL, bit exact against
// it and twin.TwinModem.rx. Aligned frames of S (256 + cp) ADC samples go
// in; the decoded bits of their codewords come out.
//
// Three stages work on three frames at once, so a frame needs the time of
// the slowest stage, not of all of them:
//
//   A  rx_fe     CP removal, FFT, timing phase, subcarrier map -> Y bank
//   B  rx_chest  pilot LS and frequency filter                 -> hf
//      rx_track  phase ramp, Y and hf derotated (if enabled)
//      rx_demap  time filter, max-log demapper                 -> LLR bank
//   C  ldpc_dec  NDEC decoders, each takes a codeword from an LLR bank
//
// Two Y banks between A and B, NLLR LLR banks between B and C. A frame
// starts (start) when stage A is idle and a Y bank is free (can_start);
// stage B takes the oldest full Y bank when an LLR bank is free; each
// codeword of the oldest full LLR bank goes to a free decoder, which copies
// it (the bank is free once its last codeword is copied). The decoders take
// turns at the output: words of 16 bits in the decoder's layout
// (rtlcfg.ldpc_out_bits) with the frame's tag and the codeword, and a
// cw_done per codeword with its iterations. Codewords can finish out of
// order across decoders. Every frame gets the next tag (start order).
//
// About 1,600 clocks an FFT symbol (A, the input stalls in between: real
// time needs a sample FIFO in front), npsym F (1 + L / 2) for the estimate,
// 2^(B/2) a data RE for the demapper, and the decoders' iterations; see
// harness/rtlcfg.ldpc_cycles.
//
// Configuration (cfg_*): registers below 0x40, then the tables, from
// harness/rtlcfg.rx_cfg; mant and shift (the LLR scale) change per frame
// and are taken with start.
//   0x0400 ldpc_dec  0x0800 at[s 4 + t]  0x0900 start[f]  0x0a00 levels
//   0x0c00 pilot signs[t F + f]  0x1000 r[d]  0x2000 taps[f 32 + j]  0x4000 perm[i]
module rx_frame #(
    parameter NDEC = 3,                        // decoders: 3 keep up with every frame failing (20 iterations)
    parameter NLLR = 2                         // LLR banks
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               cfg_we,
    input  wire        [15:0] cfg_addr,
    input  wire        [31:0] cfg_data,
    input  wire               start,
    output wire               can_start,
    output wire               idle,
    input  wire               in_valid,
    input  wire signed [15:0] in_re,
    input  wire signed [15:0] in_im,
    output wire               in_ready,
    output reg                out_valid,
    output reg         [15:0] out_word,
    output reg         [15:0] out_mask,
    output reg         [9:0]  out_addr,
    output reg         [1:0]  out_cw,
    output reg         [7:0]  out_tag,
    output reg                cw_done,
    output reg         [4:0]  cw_iters,
    // probes for the tests
    output wire               dbg_y_we,
    output wire        [11:0] dbg_y_addr,
    output wire        [31:0] dbg_y_data,
    output wire               dbg_hf_we,
    output wire        [9:0]  dbg_hf_addr,
    output wire        [31:0] dbg_hf_data,
    output wire               dbg_phi_valid,
    output wire        [3:0]  dbg_phi_idx,
    output wire signed [31:0] dbg_phi,
    output wire               dbg_h_valid,
    output wire        [31:0] dbg_h,
    output wire        [31:0] dbg_dy,
    output wire               dbg_llr_valid,
    output wire        [13:0] dbg_llr_idx,
    output wire signed [5:0]  dbg_llr,
    output wire        [7:0]  dbg_b_tag            // the frame stage B is on
);
    localparam LB = (NLLR > 2) ? 2 : 1;        // LLR bank index bits
    localparam DB = (NDEC > 2) ? 2 : 1;        // decoder index bits
    /* verilator lint_off WIDTHTRUNC */
    localparam [LB-1:0] LLAST = NLLR - 1;
    /* verilator lint_on WIDTHTRUNC */

    // ---- registers
    reg [3:0]  r_nsym, r_psym0, r_psym1, r_psym2;
    reg [7:0]  r_nf, r_glo, r_ghi, r_lmin;
    reg [1:0]  r_npsym, r_ncw;
    reg [5:0]  r_sump, r_taps, r_shift, r_cp;
    reg        r_dc, r_track;
    reg [2:0]  r_bps;
    reg [9:0]  r_cww;
    reg [16:0] r_aq, r_gain, r_mant;
    wire is_reg = cfg_we && (cfg_addr[15:6] == 10'd0);
    always @(posedge clk) begin
        if (is_reg) begin
            case (cfg_addr[5:0])
                6'd0:  r_nsym  <= cfg_data[3:0];
                6'd1:  r_nf    <= cfg_data[7:0];
                6'd2:  r_npsym <= cfg_data[1:0];
                6'd3:  r_psym0 <= cfg_data[3:0];
                6'd4:  r_psym1 <= cfg_data[3:0];
                6'd5:  r_psym2 <= cfg_data[3:0];
                6'd6:  r_sump  <= cfg_data[5:0];
                6'd7:  r_taps  <= cfg_data[5:0];
                6'd8:  r_glo   <= cfg_data[7:0];
                6'd9:  r_ghi   <= cfg_data[7:0];
                6'd10: r_dc    <= cfg_data[0];
                6'd11: r_lmin  <= cfg_data[7:0];
                6'd12: r_bps   <= cfg_data[2:0];
                6'd13: r_ncw   <= cfg_data[1:0];
                6'd14: r_cww   <= cfg_data[9:0];
                6'd16: r_aq    <= cfg_data[16:0];
                6'd17: r_track <= cfg_data[0];
                6'd18: r_gain  <= cfg_data[16:0];
                6'd19: r_mant  <= cfg_data[16:0];
                6'd20: r_shift <= cfg_data[5:0];
                6'd21: r_cp    <= cfg_data[5:0];
                default: ;
            endcase
        end
    end

    // ---- small tables, read without latency
    reg signed [15:0] t_at [0:63];
    reg        [7:0]  t_start [0:255];
    reg signed [3:0]  t_lv [0:7];
    reg        [1:0]  t_sg [0:1023];
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:6] == 10'h020)
            t_at[cfg_addr[5:0]] <= cfg_data[15:0];
        if (cfg_we && cfg_addr[15:8] == 8'h09)
            t_start[cfg_addr[7:0]] <= cfg_data[7:0];
        if (cfg_we && cfg_addr[15:3] == 13'h0140)
            t_lv[cfg_addr[2:0]] <= cfg_data[3:0];
        if (cfg_we && cfg_addr[15:10] == 6'h03)
            t_sg[cfg_addr[9:0]] <= cfg_data[1:0];
    end

    // ---- configuration memories: one write port, registered reads
    reg [31:0] m_tap0 [0:4095];                // taps j even at {f, j / 2}
    reg [31:0] m_tap1 [0:4095];                // and odd
    reg [15:0] m_r [0:4095];
    reg [14:0] m_perm [0:16383];
    reg [31:0] tap0_q, tap1_q;
    reg [15:0] r_q;
    reg [14:0] perm_q;
    wire [11:0] tap_ra, r_ra;
    wire [13:0] perm_ra;
    wire        tap_we = cfg_we && cfg_addr[15:13] == 3'b001;
    always @(posedge clk) begin
        if (tap_we && !cfg_addr[0])
            m_tap0[{cfg_addr[12:5], cfg_addr[4:1]}] <= cfg_data;
        tap0_q <= m_tap0[tap_ra];
    end
    always @(posedge clk) begin
        if (tap_we && cfg_addr[0])
            m_tap1[{cfg_addr[12:5], cfg_addr[4:1]}] <= cfg_data;
        tap1_q <= m_tap1[tap_ra];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:12] == 4'h1)
            m_r[cfg_addr[11:0]] <= cfg_data[15:0];
        r_q <= m_r[r_ra];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:14] == 2'b01)
            m_perm[cfg_addr[13:0]] <= cfg_data[14:0];
        perm_q <= m_perm[perm_ra];
    end

    // ================= stage A: the front end into a Y bank =================
    reg  [1:0] ystate [0:1];                   // 0 free, 1 filling, 2 full, 3 in stage B
    reg  [7:0] ytag [0:1];
    reg [16:0] ymant [0:1];                    // the LLR scale when the frame started
    reg  [5:0] yshift [0:1];
    reg        a_bank, b_bank;                 // the bank A fills next, the bank B takes next
    reg        a_run;
    reg  [7:0] tag_next;
    assign can_start = !a_run && ystate[a_bank] == 2'd0;

    wire        fe_we, fe_done;
    wire [11:0] fe_wa;
    wire [31:0] fe_wd;
    rx_fe fe (
        .clk(clk), .rst_n(rst_n), .cp_len(r_cp), .num_sym(r_nsym), .nf(r_nf), .guard_lo(r_glo), .guard_hi(r_ghi),
        .dc_null(r_dc), .l_min(r_lmin), .start(start && can_start), .in_valid(in_valid), .in_re(in_re),
        .in_im(in_im), .in_ready(in_ready), .y_we(fe_we), .y_addr(fe_wa), .y_data(fe_wd), .done(fe_done));

    // ================= stage B: estimate, tracker, demapper =================
    localparam P_IDLE = 2'd0, P_CH = 2'd1, P_TR = 2'd2, P_DM = 2'd3;
    reg [1:0]    ph;
    reg          go;
    reg [LB-1:0] b_lbank;                      // the LLR bank B writes
    reg [7:0]    b_tag;
    reg [16:0]   b_mant;
    reg [5:0]    b_shift;
    assign dbg_b_tag = b_tag;

    // the Y banks: A writes one, B reads and (the tracker) writes the other
    reg  [31:0] m_y0 [0:4095];
    reg  [31:0] m_y1 [0:4095];
    reg  [31:0] y0_q, y1_q;
    wire [31:0] y_q = b_bank ? y1_q : y0_q;
    wire [11:0] y_ra;
    wire        tr_ywe;
    wire [11:0] tr_ywa;
    wire [31:0] tr_ywd;
    wire        y0_we = (a_run && !a_bank) ? fe_we : (b_bank == 1'b0 && tr_ywe);
    wire        y1_we = (a_run && a_bank) ? fe_we : (b_bank == 1'b1 && tr_ywe);
    wire [11:0] y0_wa = (a_run && !a_bank) ? fe_wa : tr_ywa;
    wire [11:0] y1_wa = (a_run && a_bank) ? fe_wa : tr_ywa;
    wire [31:0] y0_wd = (a_run && !a_bank) ? fe_wd : tr_ywd;
    wire [31:0] y1_wd = (a_run && a_bank) ? fe_wd : tr_ywd;
    always @(posedge clk) begin
        if (y0_we)
            m_y0[y0_wa] <= y0_wd;
        y0_q <= m_y0[y_ra];
    end
    always @(posedge clk) begin
        if (y1_we)
            m_y1[y1_wa] <= y1_wd;
        y1_q <= m_y1[y_ra];
    end
    assign dbg_y_we = fe_we || tr_ywe;
    assign dbg_y_addr = fe_we ? fe_wa : tr_ywa;
    assign dbg_y_data = fe_we ? fe_wd : tr_ywd;

    // the pilot estimates, one bank per pilot symbol t, at {t, f}
    reg  [31:0] m_hf0 [0:255];
    reg  [31:0] m_hf1 [0:255];
    reg  [31:0] m_hf2 [0:255];
    reg  [31:0] hf0_q, hf1_q, hf2_q;
    reg         hf_we;
    reg  [9:0]  hf_wa;
    reg  [31:0] hf_wd;
    reg  [7:0]  hf_rf;
    reg  [1:0]  hf_rt;                         // the tracker's read: its pilot symbol, a clock later
    always @(posedge clk) begin
        if (hf_we && hf_wa[9:8] == 2'd0)
            m_hf0[hf_wa[7:0]] <= hf_wd;
        hf0_q <= m_hf0[hf_rf];
    end
    always @(posedge clk) begin
        if (hf_we && hf_wa[9:8] == 2'd1)
            m_hf1[hf_wa[7:0]] <= hf_wd;
        hf1_q <= m_hf1[hf_rf];
    end
    always @(posedge clk) begin
        if (hf_we && hf_wa[9:8] == 2'd2)
            m_hf2[hf_wa[7:0]] <= hf_wd;
        hf2_q <= m_hf2[hf_rf];
    end
    wire [31:0] hf_q = (hf_rt == 2'd0) ? hf0_q : (hf_rt == 2'd1) ? hf1_q : hf2_q;

    wire [7:0]  ch_sta;
    wire [9:0]  ch_sga, ch_hfa;
    wire [11:0] ch_tpa;
    wire [11:0] ch_ya;
    wire        ch_we, ch_done;
    wire [31:0] ch_hfd;
    rx_chest chest (
        .clk(clk), .rst_n(rst_n), .start(go && ph == P_CH), .nf(r_nf), .taps(r_taps), .npsym(r_npsym),
        .psym0(r_psym0), .psym1(r_psym1), .psym2(r_psym2), .st_addr(ch_sta), .st_data(t_start[ch_sta]),
        .sg_addr(ch_sga), .sg_data(t_sg[ch_sga]), .tp_addr(ch_tpa), .tp0_data(tap0_q), .tp1_data(tap1_q),
        .y_addr(ch_ya), .y_data(y_q), .hf_we(ch_we), .hf_addr(ch_hfa), .hf_data(ch_hfd), .done(ch_done));
    assign tap_ra = ch_tpa;

    wire [9:0]  tr_hra, tr_hwa;
    wire [11:0] tr_yra;
    wire        tr_hwe, tr_done;
    wire [31:0] tr_hwd;
    rx_track track (
        .clk(clk), .rst_n(rst_n), .start(go && ph == P_TR), .nf(r_nf), .num_sym(r_nsym), .npsym(r_npsym),
        .psym0(r_psym0), .psym1(r_psym1), .psym2(r_psym2), .sum_psym(r_sump), .gain(r_gain),
        .hf_raddr(tr_hra), .hf_rdata(hf_q), .hf_we(tr_hwe), .hf_waddr(tr_hwa), .hf_wdata(tr_hwd),
        .y_raddr(tr_yra), .y_rdata(y_q), .y_we(tr_ywe), .y_waddr(tr_ywa), .y_wdata(tr_ywd),
        .phi_valid(dbg_phi_valid), .phi_idx(dbg_phi_idx), .phi_val(dbg_phi), .done(tr_done));

    wire [3:0]  dm_ats;
    wire [2:0]  dm_lva;
    wire [7:0]  dm_hfa;
    wire [11:0] dm_ya;
    wire        dm_done;
    wire        llr_we;
    wire [13:0] llr_wa;
    wire [5:0]  llr_wd;
    rx_demap demap (
        .clk(clk), .rst_n(rst_n), .start(go && ph == P_DM), .nf(r_nf), .num_sym(r_nsym), .npsym(r_npsym),
        .psym0(r_psym0), .psym1(r_psym1), .psym2(r_psym2), .bps(r_bps), .a_q(r_aq), .mant(b_mant),
        .shift(b_shift), .at_s(dm_ats), .at_data({t_at[{dm_ats, 2'd2}], t_at[{dm_ats, 2'd1}], t_at[{dm_ats, 2'd0}]}),
        .lv_addr(dm_lva), .lv_data(t_lv[dm_lva]), .r_addr(r_ra), .r_data(r_q), .perm_addr(perm_ra),
        .perm_data(perm_q), .hf_addr(dm_hfa), .hf_data({hf2_q, hf1_q, hf0_q}), .y_addr(dm_ya), .y_data(y_q),
        .llr_we(llr_we), .llr_addr(llr_wa), .llr_data(llr_wd), .dbg_h_valid(dbg_h_valid), .dbg_h(dbg_h),
        .dbg_y(dbg_dy), .dbg_llr_valid(dbg_llr_valid), .dbg_llr_idx(dbg_llr_idx), .dbg_llr(dbg_llr),
        .done(dm_done));

    assign y_ra = (ph == P_CH) ? ch_ya : (ph == P_TR) ? tr_yra : dm_ya;
    always @(*) begin
        hf_rf = (ph == P_TR) ? tr_hra[7:0] : dm_hfa;
        hf_we = (ph == P_CH) ? ch_we : tr_hwe;
        hf_wa = (ph == P_CH) ? ch_hfa : tr_hwa;
        hf_wd = (ph == P_CH) ? ch_hfd : tr_hwd;
    end
    always @(posedge clk)
        hf_rt <= tr_hra[9:8];
    assign dbg_hf_we = hf_we;
    assign dbg_hf_addr = hf_wa;
    assign dbg_hf_data = hf_wd;

    // ================= the LLR banks: 16 lanes of 8 bits a word =================
    reg  [1:0]    lstate [0:NLLR-1];          // 0 free, 1 being written, 2 full
    reg  [7:0]    ltag [0:NLLR-1];
    reg  [127:0]  m_llr [0:NLLR*1024-1];
    reg  [127:0]  llr_q;
    wire [LB+9:0] llr_ra;
    wire [LB+9:0] llr_wa_full = {b_lbank, llr_wa[13:4]};
    integer bl;
    always @(posedge clk) begin
        for (bl = 0; bl < 16; bl = bl + 1)
            if (llr_we && llr_wa[3:0] == bl[3:0])
                m_llr[llr_wa_full][bl*8 +: 8] <= {{2{llr_wd[5]}}, llr_wd};
        llr_q <= m_llr[llr_ra];
    end
    reg [95:0] llr_q6;
    integer bj;
    always @(*) begin
        for (bj = 0; bj < 16; bj = bj + 1)
            llr_q6[bj*6 +: 6] = llr_q[bj*8 +: 6];
    end

    // ================= stage C: the decoders =================
    reg  [LB-1:0] c_lbank;                     // the LLR bank being handed out
    reg  [1:0]    c_cw;                        // its next codeword
    reg           c_wait;                      // a decoder is copying it
    reg  [DB-1:0] c_dec;                       // that decoder
    reg  [NDEC-1:0] d_start;
    reg  [9:0]    d_base;
    reg  [7:0]    d_tag [0:NDEC-1];
    reg  [1:0]    d_cw [0:NDEC-1];
    wire [NDEC-1:0] d_busy, d_loaded, d_req, d_valid, d_done;
    wire [9:0]    d_ra [0:NDEC-1];
    wire [9:0]    d_oaddr [0:NDEC-1];
    wire [15:0]   d_oword [0:NDEC-1];
    wire [15:0]   d_omask [0:NDEC-1];
    wire [4:0]    d_iters [0:NDEC-1];
    reg           o_own;                       // a decoder has the output
    reg  [DB-1:0] o_dec;
    genvar gi;
    generate
        for (gi = 0; gi < NDEC; gi = gi + 1) begin : dec
            /* verilator lint_off UNUSEDSIGNAL */
            wire re;
            /* verilator lint_on UNUSEDSIGNAL */
            ldpc_dec u (
                .clk(clk), .rst_n(rst_n), .cfg_we(cfg_we && cfg_addr[15:10] == 6'h01), .cfg_addr(cfg_addr[9:0]),
                .cfg_data(cfg_data), .start(d_start[gi]), .llr_base(d_base), .busy(d_busy[gi]),
                .llr_addr(d_ra[gi]), .llr_re(re), .llr_data(llr_q6), .loaded(d_loaded[gi]), .out_req(d_req[gi]),
                .out_ready(o_own && o_dec == gi), .out_valid(d_valid[gi]), .out_addr(d_oaddr[gi]),
                .out_word(d_oword[gi]), .out_mask(d_omask[gi]), .done(d_done[gi]), .iters(d_iters[gi]));
        end
    endgenerate
    assign llr_ra = {c_lbank, d_ra[c_dec]};

    // a free decoder (the lowest), and a decoder that wants the output
    reg          free_any, req_any;
    reg [DB-1:0] free_i, req_i;
    integer di;
    always @(*) begin
        free_any = 1'b0;
        free_i = {DB{1'b0}};
        req_any = 1'b0;
        req_i = {DB{1'b0}};
        for (di = NDEC - 1; di >= 0; di = di - 1) begin
            if (!d_busy[di] && !d_start[di]) begin
                free_any = 1'b1;
                free_i = di[DB-1:0];
            end
            if (d_req[di]) begin
                req_any = 1'b1;
                req_i = di[DB-1:0];
            end
        end
    end

    reg lstate_all_free;
    integer li;
    always @(*) begin
        lstate_all_free = 1'b1;
        for (li = 0; li < NLLR; li = li + 1)
            if (lstate[li] != 2'd0)
                lstate_all_free = 1'b0;
    end
    assign idle = !a_run && ystate[0] == 2'd0 && ystate[1] == 2'd0 && ph == P_IDLE && !go && !c_wait &&
                  d_busy == {NDEC{1'b0}} && d_start == {NDEC{1'b0}} && lstate_all_free && !o_own && !cw_done;

    integer ii;
    always @(posedge clk) begin
        if (!rst_n) begin
            ystate[0] <= 2'd0;
            ystate[1] <= 2'd0;
            for (ii = 0; ii < NLLR; ii = ii + 1)
                lstate[ii] <= 2'd0;
            a_bank <= 1'b0;
            b_bank <= 1'b0;
            a_run <= 1'b0;
            tag_next <= 8'd0;
            ph <= P_IDLE;
            go <= 1'b0;
            b_lbank <= {LB{1'b0}};
            c_lbank <= {LB{1'b0}};
            c_cw <= 2'd0;
            c_wait <= 1'b0;
            d_start <= {NDEC{1'b0}};
            o_own <= 1'b0;
            out_valid <= 1'b0;
            cw_done <= 1'b0;
        end else begin
            go <= 1'b0;
            d_start <= {NDEC{1'b0}};
            out_valid <= 1'b0;
            cw_done <= 1'b0;

            // stage A
            if (start && can_start) begin
                a_run <= 1'b1;
                ystate[a_bank] <= 2'd1;
                ytag[a_bank] <= tag_next;
                ymant[a_bank] <= r_mant;
                yshift[a_bank] <= r_shift;
                tag_next <= tag_next + 8'd1;
            end
            if (a_run && fe_done) begin
                a_run <= 1'b0;
                ystate[a_bank] <= 2'd2;
                a_bank <= !a_bank;
            end

            // stage B
            case (ph)
                P_IDLE: if (ystate[b_bank] == 2'd2 && lstate[b_lbank] == 2'd0) begin
                    ph <= P_CH;
                    go <= 1'b1;
                    ystate[b_bank] <= 2'd3;
                    lstate[b_lbank] <= 2'd1;
                    b_tag <= ytag[b_bank];
                    b_mant <= ymant[b_bank];
                    b_shift <= yshift[b_bank];
                end
                P_CH: if (ch_done) begin
                    ph <= r_track ? P_TR : P_DM;
                    go <= 1'b1;
                end
                P_TR: if (tr_done) begin
                    ph <= P_DM;
                    go <= 1'b1;
                end
                default: if (dm_done) begin       // P_DM
                    ph <= P_IDLE;
                    ystate[b_bank] <= 2'd0;
                    b_bank <= !b_bank;
                    lstate[b_lbank] <= 2'd2;
                    ltag[b_lbank] <= b_tag;
                    b_lbank <= (b_lbank == LLAST) ? {LB{1'b0}} : b_lbank + 1'b1;
                end
            endcase

            // stage C: hand the codewords out, one copy at a time
            if (c_wait) begin
                if (d_loaded[c_dec]) begin
                    c_wait <= 1'b0;
                    if (c_cw == r_ncw - 2'd1) begin
                        c_cw <= 2'd0;
                        lstate[c_lbank] <= 2'd0;
                        c_lbank <= (c_lbank == LLAST) ? {LB{1'b0}} : c_lbank + 1'b1;
                    end else
                        c_cw <= c_cw + 2'd1;
                end
            end else if (lstate[c_lbank] == 2'd2 && free_any) begin
                c_wait <= 1'b1;
                c_dec <= free_i;
                d_start[free_i] <= 1'b1;
                d_base <= (c_cw == 2'd0) ? 10'd0 : (c_cw == 2'd1) ? r_cww : {r_cww[8:0], 1'b0};
                d_tag[free_i] <= ltag[c_lbank];
                d_cw[free_i] <= c_cw;
            end

            // the output: one decoder at a time, until its done
            if (!o_own && req_any) begin
                o_own <= 1'b1;
                o_dec <= req_i;
            end
            if (o_own && d_valid[o_dec]) begin
                out_valid <= 1'b1;
                out_word <= d_oword[o_dec];
                out_mask <= d_omask[o_dec];
                out_addr <= d_oaddr[o_dec];
                out_cw <= d_cw[o_dec];
                out_tag <= d_tag[o_dec];
            end
            if (o_own && d_done[o_dec]) begin
                o_own <= 1'b0;
                cw_done <= 1'b1;
                cw_iters <= d_iters[o_dec];
                out_cw <= d_cw[o_dec];
                out_tag <= d_tag[o_dec];
            end
        end
    end
endmodule
