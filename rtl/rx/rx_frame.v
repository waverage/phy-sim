`timescale 1ns / 1ps
// The frame receiver: fxmodem.FxModem(true).rx in RTL, bit exact against
// it and twin.TwinModem.rx. An aligned frame of S (256 + cp) ADC samples
// goes in; the decoded bits of its codewords come out.
//
//   rx_fe     CP removal, FFT, timing phase, subcarrier map -> Y [S F]
//   rx_chest  pilot LS and frequency filter                -> hf [npsym F]
//   rx_track  phase ramp, Y and hf derotated (if enabled)
//   rx_demap  time filter, max-log demapper                -> deinterleaver
//   ldpc_dec  each codeword from the deinterleaver         -> bits
//
// The phases run one after another on one frame (not overlapped): about
// 36k clocks for the FFTs, npsym F L for the estimate, 20 a data RE for
// the demapper and the decoder's iterations; see harness/rtlcfg.rx_cycles.
//
// Configuration (cfg_*): registers below 0x40, then the tables, from
// harness/rtlcfg.rx_cfg; mant and shift (the LLR scale) change per frame.
//   0x0400 ldpc_dec  0x0800 at[s 4 + t]  0x0900 start[f]  0x0a00 levels
//   0x0c00 pilot signs[t F + f]  0x1000 r[d]  0x2000 taps[f 32 + j]  0x4000 perm[i]
module rx_frame (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               cfg_we,
    input  wire        [15:0] cfg_addr,
    input  wire        [31:0] cfg_data,
    input  wire               start,
    output wire               busy,
    input  wire               in_valid,
    input  wire signed [15:0] in_re,
    input  wire signed [15:0] in_im,
    output wire               in_ready,
    output wire               out_valid,
    output wire               out_bit,
    output wire        [13:0] out_idx,
    output reg         [1:0]  out_cw,
    output reg                done,
    output reg         [9:0]  iters,          // sum over the frame's codewords
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
    output wire signed [5:0]  dbg_llr
);
    // ---- registers
    reg [3:0]  r_nsym, r_psym0, r_psym1, r_psym2;
    reg [7:0]  r_nf, r_glo, r_ghi, r_lmin;
    reg [1:0]  r_npsym, r_ncw;
    reg [5:0]  r_sump, r_taps, r_shift, r_cp;
    reg        r_dc, r_track;
    reg [2:0]  r_bps;
    reg [13:0] r_n;
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
                6'd14: r_n     <= cfg_data[13:0];
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

    // ---- block RAMs: one write port, one registered read port each
    reg [31:0] m_taps [0:8191];
    reg [15:0] m_r [0:4095];
    reg [13:0] m_perm [0:16383];
    reg [31:0] m_y [0:4095];
    reg [31:0] m_hf [0:1023];
    reg [5:0]  m_llr [0:16383];
    reg [31:0] taps_q, y_q, hf_q;
    reg [15:0] r_q;
    reg [13:0] perm_q;
    reg [5:0]  llr_q;
    wire [12:0] taps_ra;
    wire [11:0] r_ra, y_ra;
    wire [13:0] perm_ra;
    reg  [13:0] llr_ra;
    wire [9:0]  hf_ra;
    reg         y_we, hf_we;
    reg  [11:0] y_wa;
    reg  [9:0]  hf_wa;
    reg  [31:0] y_wd, hf_wd;
    wire        llr_we;
    wire [13:0] llr_wa;
    wire [5:0]  llr_wd;
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:13] == 3'b001)
            m_taps[cfg_addr[12:0]] <= cfg_data;
        taps_q <= m_taps[taps_ra];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:12] == 4'h1)
            m_r[cfg_addr[11:0]] <= cfg_data[15:0];
        r_q <= m_r[r_ra];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_addr[15:14] == 2'b01)
            m_perm[cfg_addr[13:0]] <= cfg_data[13:0];
        perm_q <= m_perm[perm_ra];
    end
    always @(posedge clk) begin
        if (y_we)
            m_y[y_wa] <= y_wd;
        y_q <= m_y[y_ra];
    end
    always @(posedge clk) begin
        if (hf_we)
            m_hf[hf_wa] <= hf_wd;
        hf_q <= m_hf[hf_ra];
    end
    always @(posedge clk) begin
        if (llr_we)
            m_llr[llr_wa] <= llr_wd;
        llr_q <= m_llr[llr_ra];
    end

    // ---- the phases
    localparam P_IDLE = 3'd0, P_FE = 3'd1, P_CH = 3'd2, P_TR = 3'd3, P_DM = 3'd4, P_DEC = 3'd5;
    reg [2:0] ph;
    reg       go;                              // start pulse of the current phase
    assign busy = (ph != P_IDLE);

    wire        fe_we, fe_done;
    wire [11:0] fe_wa;
    wire [31:0] fe_wd;
    rx_fe fe (
        .clk(clk), .rst_n(rst_n), .cp_len(r_cp), .num_sym(r_nsym), .nf(r_nf), .guard_lo(r_glo), .guard_hi(r_ghi),
        .dc_null(r_dc), .l_min(r_lmin), .start(go && ph == P_FE), .in_valid(in_valid), .in_re(in_re),
        .in_im(in_im), .in_ready(in_ready), .y_we(fe_we), .y_addr(fe_wa), .y_data(fe_wd), .done(fe_done));

    wire [7:0]  ch_sta;
    wire [9:0]  ch_sga, ch_hfa;
    wire [12:0] ch_tpa;
    wire [11:0] ch_ya;
    wire        ch_we, ch_done;
    wire [31:0] ch_hfd;
    rx_chest chest (
        .clk(clk), .rst_n(rst_n), .start(go && ph == P_CH), .nf(r_nf), .taps(r_taps), .npsym(r_npsym),
        .psym0(r_psym0), .psym1(r_psym1), .psym2(r_psym2), .st_addr(ch_sta), .st_data(t_start[ch_sta]),
        .sg_addr(ch_sga), .sg_data(t_sg[ch_sga]), .tp_addr(ch_tpa), .tp_data(taps_q), .y_addr(ch_ya),
        .y_data(y_q), .hf_we(ch_we), .hf_addr(ch_hfa), .hf_data(ch_hfd), .done(ch_done));
    assign taps_ra = ch_tpa;

    wire [9:0]  tr_hra, tr_hwa;
    wire [11:0] tr_yra, tr_ywa;
    wire        tr_hwe, tr_ywe, tr_done;
    wire [31:0] tr_hwd, tr_ywd;
    rx_track track (
        .clk(clk), .rst_n(rst_n), .start(go && ph == P_TR), .nf(r_nf), .num_sym(r_nsym), .npsym(r_npsym),
        .psym0(r_psym0), .psym1(r_psym1), .psym2(r_psym2), .sum_psym(r_sump), .gain(r_gain),
        .hf_raddr(tr_hra), .hf_rdata(hf_q), .hf_we(tr_hwe), .hf_waddr(tr_hwa), .hf_wdata(tr_hwd),
        .y_raddr(tr_yra), .y_rdata(y_q), .y_we(tr_ywe), .y_waddr(tr_ywa), .y_wdata(tr_ywd),
        .phi_valid(dbg_phi_valid), .phi_idx(dbg_phi_idx), .phi_val(dbg_phi), .done(tr_done));

    wire [5:0]  dm_ata;
    wire [2:0]  dm_lva;
    wire [9:0]  dm_hfa;
    wire [11:0] dm_ya;
    wire        dm_done;
    rx_demap demap (
        .clk(clk), .rst_n(rst_n), .start(go && ph == P_DM), .nf(r_nf), .num_sym(r_nsym), .npsym(r_npsym),
        .psym0(r_psym0), .psym1(r_psym1), .psym2(r_psym2), .bps(r_bps), .a_q(r_aq), .mant(r_mant),
        .shift(r_shift), .at_addr(dm_ata), .at_data(t_at[dm_ata]), .lv_addr(dm_lva), .lv_data(t_lv[dm_lva]),
        .r_addr(r_ra), .r_data(r_q), .perm_addr(perm_ra), .perm_data(perm_q), .hf_addr(dm_hfa), .hf_data(hf_q),
        .y_addr(dm_ya), .y_data(y_q), .llr_we(llr_we), .llr_addr(llr_wa), .llr_data(llr_wd),
        .dbg_h_valid(dbg_h_valid), .dbg_h(dbg_h), .dbg_y(dbg_dy), .dbg_llr_valid(dbg_llr_valid),
        .dbg_llr_idx(dbg_llr_idx), .dbg_llr(dbg_llr), .done(dm_done));

    assign y_ra  = (ph == P_CH) ? ch_ya : (ph == P_TR) ? tr_yra : dm_ya;
    assign hf_ra = (ph == P_TR) ? tr_hra : dm_hfa;
    always @(*) begin
        y_we  = (ph == P_FE) ? fe_we : tr_ywe;
        y_wa  = (ph == P_FE) ? fe_wa : tr_ywa;
        y_wd  = (ph == P_FE) ? fe_wd : tr_ywd;
        hf_we = (ph == P_CH) ? ch_we : tr_hwe;
        hf_wa = (ph == P_CH) ? ch_hfa : tr_hwa;
        hf_wd = (ph == P_CH) ? ch_hfd : tr_hwd;
    end
    assign dbg_y_we = y_we;
    assign dbg_y_addr = y_wa;
    assign dbg_y_data = y_wd;
    assign dbg_hf_we = hf_we;
    assign dbg_hf_addr = hf_wa;
    assign dbg_hf_data = hf_wd;

    // ---- the decoder, fed from the deinterleaver
    wire        dec_start = go && ph == P_DEC;
    wire        dec_ready, dec_done;
    /* verilator lint_off UNUSEDSIGNAL */
    wire        dec_busy;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [4:0]  dec_iters;
    reg         fv;
    reg  [13:0] fcnt;
    ldpc_dec dec (
        .clk(clk), .rst_n(rst_n), .cfg_we(cfg_we && cfg_addr[15:10] == 6'h01), .cfg_addr(cfg_addr[9:0]),
        .cfg_data(cfg_data), .start(dec_start), .busy(dec_busy), .in_valid(fv), .in_llr(llr_q),
        .in_ready(dec_ready), .out_valid(out_valid), .out_bit(out_bit), .out_idx(out_idx), .done(dec_done),
        .iters(dec_iters));
    wire feed = (ph == P_DEC) && dec_ready && (fcnt != r_n) && !go;

    always @(posedge clk) begin
        if (!rst_n) begin
            ph <= P_IDLE;
            go <= 1'b0;
            done <= 1'b0;
            fv <= 1'b0;
        end else begin
            go <= 1'b0;
            done <= 1'b0;
            fv <= feed;
            if (feed) begin
                llr_ra <= llr_ra + 14'd1;
                fcnt <= fcnt + 14'd1;
            end
            case (ph)
                P_IDLE: if (start) begin
                    ph <= P_FE;
                    go <= 1'b1;
                end
                P_FE: if (fe_done) begin
                    ph <= P_CH;
                    go <= 1'b1;
                end
                P_CH: if (ch_done) begin
                    ph <= r_track ? P_TR : P_DM;
                    go <= 1'b1;
                end
                P_TR: if (tr_done) begin
                    ph <= P_DM;
                    go <= 1'b1;
                end
                P_DM: if (dm_done) begin
                    ph <= P_DEC;
                    go <= 1'b1;
                    llr_ra <= 14'd0;
                    fcnt <= 14'd0;
                    out_cw <= 2'd0;
                    iters <= 10'd0;
                end
                default: if (dec_done) begin       // P_DEC
                    iters <= iters + {5'd0, dec_iters};
                    fcnt <= 14'd0;
                    if (out_cw == r_ncw - 2'd1) begin
                        ph <= P_IDLE;
                        done <= 1'b1;
                    end else begin
                        out_cw <= out_cw + 2'd1;
                        go <= 1'b1;
                    end
                end
            endcase
        end
    end
endmodule
