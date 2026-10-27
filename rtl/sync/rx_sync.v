`timescale 1ns / 1ps
// The synchroniser on an endless stream of ADC samples (M9):
// fxsync.Sync.runOne and alignOne in RTL, bit exact against
// twin.TwinSync.stream (harness/twin.py), which defines the stream rules:
//
//   the search starts at a base (the first sample after enable) as if a
//   buffer began there: stf_corr finds the crossing and the coarse peak d;
//   then the coarse CFO (angle30 of P), the first pilot symbol's window
//   through the coarse NCO, the FFT, the conjugate pilot and the IFFT (the
//   CIR), its first path (tau) and the CIR check; the medium CFO from the
//   cyclic prefixes; the frame aligned and derotated, S slot samples from
//   start - backoff (the NCO counts samples from the base). A failed CIR
//   check moves the base to base + d + pre; a synchronised frame moves it
//   to start + S slot - rearm, so the search sees the next preamble.
//
// The samples go into a buffer of 2^BW (12-bit I/Q); everything reads from
// it through one read port that the detector (even clocks) and the
// sequencer (odd clocks) share, so the detector catches up at half a
// sample per clock after a re-arm in the past. The sequencer serves one
// detection at a time (the next one waits, latched): about 3,700 clocks for
// the CIR, 900 for the prefixes, then the 4,032 output samples, paced by
// the receiver. A sample read after the buffer wrapped over it counts as
// an overrun (the frame is then wrong).
//
// Configuration (cfg_*): registers 0 threshold_q8, 1 first_path_shift,
// 2 noise_floor, 3 cir_min_ratio, 4 backoff, 5 rearm, 6 S, 7 pilot_symbol,
// 8 cp_cfo, 9 guard_lo, 10 guard_hi, 11 dc_null, 12 enable (setting it
// starts the search at the next sample, clearing it drops the frame in
// progress and the output not yet taken); the first pilot symbol's signs at
// 256 + f, {re < 0, im < 0}. N = 256, cp = 32, period 32, 9 repetitions.
module rx_sync #(
    parameter BW = 13                          // buffer: 2^BW samples
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               cfg_we,
    input  wire        [9:0]  cfg_addr,
    input  wire        [31:0] cfg_data,
    input  wire               in_valid,
    input  wire signed [11:0] in_re,
    input  wire signed [11:0] in_im,
    // the aligned frames
    output wire               out_valid,
    output wire signed [15:0] out_re,
    output wire signed [15:0] out_im,
    output wire               out_first,
    input  wire               out_ready,
    // per synchronised frame, a clock before its alignment starts
    output reg                info_valid,
    output reg         [31:0] info_base,
    output reg         [31:0] info_start,     // absolute
    output reg         [31:0] info_peak,      // absolute
    output reg  signed [8:0]  info_tau,
    output reg  signed [17:0] info_ang_coarse,
    output reg  signed [17:0] info_ang_cp,
    output reg         [31:0] info_inc,
    output reg         [32:0] info_cir_peak,
    output reg         [40:0] info_cir_sum,
    // counters
    output reg         [31:0] n_detect,       // crossings
    output reg         [31:0] n_reject,       // failed CIR checks
    output reg         [31:0] n_sync,         // synchronised frames
    output reg         [31:0] n_overrun
);
    localparam [31:0] PRE32 = 288, SLOT32 = 288, CP32 = 32, N32 = 256;

    // ---- configuration
    reg [8:0] r_thr;
    reg [4:0] r_fps;
    reg [7:0] r_nf, r_ratio, r_glo, r_ghi;
    reg [5:0] r_backoff;
    reg [9:0] r_rearm;
    reg [3:0] r_nsym, r_psym;
    reg       r_cpcfo, r_dc, r_en;
    reg [1:0] t_sg [0:255];
    always @(posedge clk) begin
        if (!rst_n)
            r_en <= 1'b0;
        else if (cfg_we && !cfg_addr[8]) begin
            case (cfg_addr[3:0])
                4'd0:  r_thr     <= cfg_data[8:0];
                4'd1:  r_fps     <= cfg_data[4:0];
                4'd2:  r_nf      <= cfg_data[7:0];
                4'd3:  r_ratio   <= cfg_data[7:0];
                4'd4:  r_backoff <= cfg_data[5:0];
                4'd5:  r_rearm   <= cfg_data[9:0];
                4'd6:  r_nsym    <= cfg_data[3:0];
                4'd7:  r_psym    <= cfg_data[3:0];
                4'd8:  r_cpcfo   <= cfg_data[0];
                4'd9:  r_glo     <= cfg_data[7:0];
                4'd10: r_ghi     <= cfg_data[7:0];
                4'd11: r_dc      <= cfg_data[0];
                4'd12: r_en      <= cfg_data[0];
                default: ;
            endcase
        end
        if (cfg_we && cfg_addr[8])
            t_sg[cfg_addr[7:0]] <= cfg_data[1:0];
    end
    wire [31:0] flen = {28'd0, r_nsym} * SLOT32;   // S slot

    // ---- the buffer: written with every sample, one read port
    reg  [23:0]   m_buf [0:(1 << BW) - 1];
    reg  [31:0]   wn;                          // samples written
    wire [BW-1:0] rd_a;
    reg  [23:0]   rd_q;
    always @(posedge clk) begin
        if (in_valid)
            m_buf[wn[BW-1:0]] <= {in_re, in_im};
        rd_q <= m_buf[rd_a];
    end
    wire signed [11:0] q_re = rd_q[23:12], q_im = rd_q[11:0];
    reg ph;                                    // 0: the detector may read, 1: the sequencer
    always @(posedge clk) begin
        if (!rst_n) begin
            wn <= 32'd0;
            ph <= 1'b0;
        end else begin
            ph <= !ph;
            if (in_valid)
                wn <= wn + 32'd1;
        end
    end
    // a sample is there once written, and gone once the buffer wrapped over it
    function avail;
        input [31:0] a;
        reg   [31:0] dd;
        begin
            dd = wn - a;
            avail = !dd[31] && dd != 32'd0;
        end
    endfunction
    function gone;
        input [31:0] a;
        begin
            gone = (wn - a) > ((32'd1 << BW) - 32'd8);
        end
    endfunction

    // ================= the detector =================
    reg         det_run;
    reg  [31:0] dp;                            // the next sample it reads
    reg  [31:0] det_base;
    reg         det_restart;
    wire        det_issue = !ph && det_run && avail(dp);
    reg         det_v;
    wire        sc_done, sc_det;
    wire [31:0] sc_peak;
    wire signed [33:0] sc_pr, sc_pi;
    /* verilator lint_off PINCONNECTEMPTY */
    stf_corr #(.DW(32)) corr (
        .clk(clk), .rst_n(rst_n), .restart(det_restart), .threshold_q8(r_thr), .in_valid(det_v), .in_re(q_re),
        .in_im(q_im), .in_last(1'b0), .trace_valid(), .trace_d(), .trace_pr(), .trace_pi(), .trace_e(),
        .done(sc_done), .detected(sc_det), .peak(sc_peak), .peak_pr(), .peak_pi(), .peak_e(),
        .peak_pr_full(sc_pr), .peak_pi_full(sc_pi));
    /* verilator lint_on PINCONNECTEMPTY */
    reg         pend;                          // a detection waits for the sequencer
    reg  [31:0] pend_base, pend_d;
    reg signed [33:0] pend_pr, pend_pi;
    reg         en_q;

    // ================= the sequencer =================
    localparam Q_IDLE = 4'd0, Q_ANG = 4'd1, Q_PRD = 4'd2, Q_FFT = 4'd3, Q_IFFT = 4'd4, Q_CIRO = 4'd5,
               Q_NEAR = 4'd6, Q_FIRST = 4'd7, Q_DEC = 4'd8, Q_CP = 4'd9, Q_CPANG = 4'd10, Q_INFO = 4'd11,
               Q_ALIGN = 4'd12, Q_DRAIN = 4'd13;
    reg  [3:0]  sq;
    reg  [31:0] base, cd;                      // the frame's base and its coarse peak d
    reg signed [17:0] a1;
    reg  [31:0] inc1, inc;
    reg  [12:0] cnt;                           // reads issued, or a step counter
    reg         nco_first;                     // the next NCO sample is the run's first
    reg  [31:0] start_rel;
    reg         seq_rd;                        // the sequencer's read issued last clock: its data is here
    reg         seq_issue;

    // the angles: one angle30 for P (coarse) and for the prefix sum
    reg               ang_in;
    reg signed [47:0] ang_re, ang_im;
    wire              ang_ov;
    wire signed [17:0] ang_out;
    /* verilator lint_off UNUSEDSIGNAL */
    wire              ang_tag;
    /* verilator lint_on UNUSEDSIGNAL */
    angle30 #(.W(48), .TAGW(1)) a30 (
        .clk(clk), .rst_n(rst_n), .in_valid(ang_in), .in_re(ang_re), .in_im(ang_im), .in_tag(1'b0),
        .out_valid(ang_ov), .out_ang(ang_out), .out_tag(ang_tag));

    // the NCO: the pilot window into the FFT, then the frame to the output
    wire        nco_in = seq_rd && (sq == Q_PRD || sq == Q_ALIGN);
    wire        nco_start = nco_in && nco_first;
    reg  [31:0] nco_ph0, nco_inc;
    wire        nco_ov;
    wire signed [15:0] nco_re, nco_im;
    nco_rot nco (
        .clk(clk), .rst_n(rst_n), .start(nco_start), .phase0(nco_ph0), .inc(nco_inc), .in_valid(nco_in),
        .in_re({{4{q_re[11]}}, q_re}), .in_im({{4{q_im[11]}}, q_im}), .out_valid(nco_ov), .out_re(nco_re),
        .out_im(nco_im));

    // the FFT: forward on the pilot window, inverse on z
    reg         fft_inv, fft_in;
    reg signed [15:0] fft_ire, fft_iim;
    wire        fft_rdy, fft_ov;
    wire signed [15:0] fft_ore, fft_oim;
    wire [7:0]  fft_oidx;
    fft256 fft (
        .clk(clk), .rst_n(rst_n), .inverse(fft_inv), .in_valid(fft_in), .in_re(fft_ire), .in_im(fft_iim),
        .in_ready(fft_rdy), .out_valid(fft_ov), .out_re(fft_ore), .out_im(fft_oim), .out_idx(fft_oidx));

    // z: the pilot FFT times the conjugate pilot, halved, zero off the subcarriers
    reg  [31:0] m_z [0:255];
    reg         z_we;
    reg  [7:0]  z_wa;
    reg  [31:0] z_wd;
    reg  [31:0] z_q;
    always @(posedge clk) begin
        if (z_we)
            m_z[z_wa] <= z_wd;
        z_q <= m_z[cnt[7:0]];
    end
    wire [7:0] zi = fft_oidx ^ 8'h80;          // natural index of the bin
    wire       z_used = (zi >= r_glo) && (zi < 8'd0 - r_ghi) && !(r_dc && zi == 8'h80);
    wire [7:0] z_f = zi - r_glo - ((r_dc && zi > 8'h80) ? 8'd1 : 8'd0);
    wire [1:0] z_sg = t_sg[z_f];
    wire signed [17:0] yr = {{2{fft_ore[15]}}, fft_ore}, yi = {{2{fft_oim[15]}}, fft_oim};
    wire signed [17:0] z_r2 = (z_sg[1] ? -yr : yr) + (z_sg[0] ? -yi : yi);     // sa yr + sb yi
    wire signed [17:0] z_i2 = (z_sg[1] ? -yi : yi) - (z_sg[0] ? -yr : yr);     // sa yi - sb yr
    function signed [15:0] half_sat;           // sat16(rshr(x, 1))
        input signed [17:0] x;
        reg signed [17:0] q;
        begin
            q = (x + 18'sd1) >>> 1;
            half_sat = (q > 18'sd32767) ? 16'sh7fff : (q < -18'sd32768) ? 16'sh8000 : q[15:0];
        end
    endfunction

    // the CIR's power, its sum and its first peak
    reg  [32:0] m_pw [0:255];
    reg  [7:0]  pw_ra;
    reg  [32:0] pw_q;
    reg         pw_we;
    reg  [7:0]  pw_wa;
    reg  [32:0] pw_wd;
    always @(posedge clk) begin
        if (pw_we)
            m_pw[pw_wa] <= pw_wd;
        pw_q <= m_pw[pw_ra];
    end
    wire signed [31:0] sq_re = fft_ore * fft_ore, sq_im = fft_oim * fft_oim;
    wire [32:0] pw_now = {1'b0, sq_re} + {1'b0, sq_im};
    reg  [40:0] total, near;
    reg  [32:0] pmax;
    reg  [7:0]  pk, first;
    reg  [47:0] nf_floor;                      // noise_floor (total - near)
    reg         nr1, nr2;                      // NEAR: a read in flight, its word next
    reg  [5:0]  j;
    reg  [32:0] cand_p;
    reg  [7:0]  cand_t;
    reg         cand_last;

    // the prefixes: r[n + N] conj(r[n]) summed
    reg signed [11:0] cp_bre, cp_bim;          // r[n], waiting for its partner
    reg signed [39:0] qr, qi;
    reg               cp_odd;                  // the arriving word is r[n + N]

    // the output FIFO: 32 deep, with room for the NCO's samples in flight
    reg  [32:0] m_o [0:31];
    reg  [4:0]  o_wp, o_rp;
    reg  [5:0]  o_cnt;
    reg  [5:0]  o_fly;                         // align reads not yet out of the NCO
    reg  [12:0] o_m;                           // aligned samples pushed
    wire        o_pop = out_valid && out_ready;
    wire        o_push = nco_ov && (sq == Q_ALIGN || sq == Q_DRAIN);
    assign out_valid = (o_cnt != 6'd0);
    assign {out_first, out_re, out_im} = m_o[o_rp];

    // the sequencer's read address (relative to base), from the state
    wire [31:0] ws0 = cd + PRE32 + {28'd0, r_psym} * SLOT32 + CP32;    // the pilot window
    reg  [31:0] want;
    always @(*) begin
        case (sq)
            Q_PRD:   want = ws0 + {19'd0, cnt};
            Q_CP:    want = start_rel + {28'd0, cnt[9:6]} * SLOT32 + {27'd0, cnt[5:1]} + (cnt[0] ? N32 : 32'd0);
            default: want = start_rel - {26'd0, r_backoff} + {19'd0, cnt};
        endcase
    end
    wire [31:0] want_abs = base + want;
    assign rd_a = det_issue ? dp[BW-1:0] : want_abs[BW-1:0];
    wire        cp_total = (cnt == {3'd0, r_nsym, 6'd0});               // 2 cp S reads
    wire        o_room = (o_cnt + o_fly) < 6'd28;
    always @(*) begin
        seq_issue = 1'b0;
        if (ph && avail(want_abs)) begin
            case (sq)
                Q_PRD:   seq_issue = (cnt != 13'd256);
                Q_CP:    seq_issue = !cp_total;
                Q_ALIGN: seq_issue = (cnt != flen[12:0]) && o_room;
                default: seq_issue = 1'b0;
            endcase
        end
    end

    // the decision's products
    wire [48:0] c1_lhs = {16'd0, cand_p} << r_fps;
    wire [40:0] c2_lhs = cand_p * 8'd191;      // floor_n = N - (2 cp + 1)
    wire        cand_ok = (c1_lhs >= {16'd0, pmax}) && ({7'd0, c2_lhs} >= nf_floor);
    wire [48:0] pm256 = {8'd0, pmax, 8'd0};
    wire [48:0] ratio_total = r_ratio * total;
    wire [31:0] tau32 = {{24{first[7]}}, first};                        // first - N if first >= N / 2
    wire [31:0] st_abs = base + cd + PRE32 + tau32;
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [21:0] dcp = {{4{ang_out[17]}}, ang_out} - {{1{a1[17]}}, a1, 3'd0};   // acp - (N / PER) a1
    wire        [22:0] cfg_unused = {cfg_addr[9], cfg_data[31:10]};
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [16:0] delta17 = (dcp[15:0] == 16'h8000) ? 17'sd32768 : {dcp[15], dcp[15:0]};   // wrapped
    wire [31:0] inc_coarse = {{3{ang_out[17]}}, ang_out, 11'd0};       // angle 2^32 / (65536 PER)

    always @(posedge clk) begin
        if (!rst_n) begin
            det_run <= 1'b0;
            det_restart <= 1'b0;
            det_v <= 1'b0;
            en_q <= 1'b0;
            pend <= 1'b0;
            sq <= Q_IDLE;
            seq_rd <= 1'b0;
            ang_in <= 1'b0;
            fft_in <= 1'b0;
            z_we <= 1'b0;
            pw_we <= 1'b0;
            nr1 <= 1'b0;
            nr2 <= 1'b0;
            info_valid <= 1'b0;
            o_wp <= 5'd0;
            o_rp <= 5'd0;
            o_cnt <= 6'd0;
            o_fly <= 6'd0;
            n_detect <= 32'd0;
            n_reject <= 32'd0;
            n_sync <= 32'd0;
            n_overrun <= 32'd0;
        end else begin
            det_restart <= 1'b0;
            ang_in <= 1'b0;
            fft_in <= 1'b0;
            z_we <= 1'b0;
            pw_we <= 1'b0;
            info_valid <= 1'b0;
            en_q <= r_en;

            // ---- the detector: one sample a read, until its peak
            det_v <= det_issue;
            if (det_issue) begin
                dp <= dp + 32'd1;
                if (gone(dp))
                    n_overrun <= n_overrun + 32'd1;
            end
            if (sc_done && sc_det && det_run) begin
                det_run <= 1'b0;
                pend <= 1'b1;
                pend_base <= det_base;
                pend_d <= sc_peak;
                pend_pr <= sc_pr;
                pend_pi <= sc_pi;
                n_detect <= n_detect + 32'd1;
            end
            if (!r_en)
                det_run <= 1'b0;
            else if (!en_q) begin              // enabled: search from the next sample
                det_run <= 1'b1;
                det_restart <= 1'b1;
                det_base <= wn;
                dp <= wn;
            end

            // ---- the sequencer's reads
            seq_rd <= seq_issue;
            if (seq_issue) begin
                cnt <= cnt + 13'd1;
                if (gone(want_abs))
                    n_overrun <= n_overrun + 32'd1;
            end
            if (nco_in)
                nco_first <= 1'b0;

            // ---- the output FIFO
            if (o_push) begin
                m_o[o_wp] <= {o_m == 13'd0, nco_re, nco_im};
                o_wp <= o_wp + 5'd1;
                o_m <= o_m + 13'd1;
            end
            if (o_pop)
                o_rp <= o_rp + 5'd1;
            o_cnt <= o_cnt + {5'd0, o_push} - {5'd0, o_pop};
            o_fly <= o_fly + {5'd0, seq_issue && sq == Q_ALIGN} - {5'd0, o_push};

            // ---- the FFT's input in the forward pass: the NCO's output
            if ((sq == Q_PRD || sq == Q_FFT) && nco_ov) begin
                fft_in <= 1'b1;
                fft_inv <= 1'b0;
                fft_ire <= nco_re;
                fft_iim <= nco_im;
            end

            case (sq)
                Q_IDLE: if (pend) begin
                    pend <= 1'b0;
                    base <= pend_base;
                    cd <= pend_d;
                    ang_in <= 1'b1;
                    ang_re <= {{14{pend_pr[33]}}, pend_pr};
                    ang_im <= {{14{pend_pi[33]}}, pend_pi};
                    sq <= Q_ANG;
                end
                Q_ANG: if (ang_ov) begin
                    a1 <= ang_out;
                    inc1 <= inc_coarse;
                    nco_inc <= inc_coarse;
                    nco_ph0 <= -(ws0 * inc_coarse);
                    cnt <= 13'd0;
                    nco_first <= 1'b1;
                    sq <= Q_PRD;
                end
                Q_PRD: if (cnt == 13'd256 && !seq_rd)          // the window is in the NCO
                    sq <= Q_FFT;
                Q_FFT: if (fft_ov) begin
                    z_we <= 1'b1;
                    z_wa <= fft_oidx;
                    z_wd <= z_used ? {half_sat(z_r2), half_sat(z_i2)} : 32'd0;
                    if (fft_oidx == 8'd255) begin
                        sq <= Q_IFFT;
                        cnt <= 13'd0;
                    end
                end
                Q_IFFT: begin                   // z -> the inverse FFT: read at cnt, the word a clock later
                    if (fft_rdy && cnt != 13'd257) begin
                        cnt <= cnt + 13'd1;
                        if (cnt != 13'd0) begin
                            fft_in <= 1'b1;
                            fft_inv <= 1'b1;
                            {fft_ire, fft_iim} <= z_q;
                        end
                    end
                    if (cnt == 13'd257) begin
                        sq <= Q_CIRO;
                        total <= 41'd0;
                        pmax <= 33'd0;
                        pk <= 8'd0;
                    end
                end
                Q_CIRO: if (fft_ov) begin       // the CIR comes out in natural order
                    pw_we <= 1'b1;
                    pw_wa <= fft_oidx;
                    pw_wd <= pw_now;
                    total <= total + {8'd0, pw_now};
                    if (fft_oidx == 8'd0 || pw_now > pmax) begin
                        pmax <= pw_now;
                        pk <= fft_oidx;
                    end
                    if (fft_oidx == 8'd255) begin
                        sq <= Q_NEAR;
                        near <= 41'd0;
                        cnt <= 13'd0;
                    end
                end
                Q_NEAR: begin                   // sum pw over pk - cp .. pk + cp
                    nr1 <= (cnt < 13'd65);
                    nr2 <= nr1;
                    if (cnt < 13'd65) begin
                        pw_ra <= pk - 8'd32 + cnt[7:0];
                        cnt <= cnt + 13'd1;
                    end
                    if (nr2)
                        near <= near + {8'd0, pw_q};
                    if (cnt == 13'd65 && !nr1 && !nr2) begin
                        nf_floor <= r_nf * {7'd0, total - near};
                        sq <= Q_FIRST;
                        j <= 6'd0;
                        cnt <= 13'd0;
                    end
                end
                Q_FIRST: begin                  // the earliest candidate within a cp before the peak
                    case (cnt[1:0])             // one every four clocks: address, read, word, test
                        2'd0: begin
                            pw_ra <= pk - 8'd31 + {2'd0, j};
                            cnt <= 13'd1;
                        end
                        2'd1: cnt <= 13'd2;
                        2'd2: begin
                            cand_p <= pw_q;
                            cand_t <= pk - 8'd31 + {2'd0, j};
                            cand_last <= (j == 6'd31);
                            cnt <= 13'd3;
                        end
                        default: begin
                            if (cand_ok || cand_last) begin
                                first <= cand_t;
                                sq <= Q_DEC;
                            end else begin
                                j <= j + 6'd1;
                                cnt <= 13'd0;
                            end
                        end
                    endcase
                end
                Q_DEC: begin
                    if (pm256 < ratio_total) begin
                        // the CIR check failed: search again past this preamble
                        n_reject <= n_reject + 32'd1;
                        det_run <= r_en;
                        det_restart <= 1'b1;
                        det_base <= base + cd + PRE32;
                        dp <= base + cd + PRE32;
                        sq <= Q_IDLE;
                    end else begin
                        start_rel <= cd + PRE32 + tau32;
                        det_run <= r_en;
                        det_restart <= 1'b1;
                        det_base <= st_abs + flen - {22'd0, r_rearm};
                        dp <= st_abs + flen - {22'd0, r_rearm};
                        n_sync <= n_sync + 32'd1;
                        info_tau <= {first[7], first};
                        info_cir_peak <= pmax;
                        info_cir_sum <= total;
                        cnt <= 13'd0;
                        qr <= 40'sd0;
                        qi <= 40'sd0;
                        cp_odd <= 1'b0;
                        if (r_cpcfo)
                            sq <= Q_CP;
                        else begin
                            inc <= inc1;
                            info_ang_cp <= 18'sd0;
                            sq <= Q_INFO;
                        end
                    end
                end
                Q_CP: begin                     // r[n], then r[n + N]: their product into q
                    if (seq_rd) begin
                        cp_odd <= !cp_odd;
                        if (!cp_odd) begin
                            cp_bre <= q_re;
                            cp_bim <= q_im;
                        end else begin
                            qr <= qr + q_re * cp_bre + q_im * cp_bim;
                            qi <= qi + q_im * cp_bre - q_re * cp_bim;
                        end
                    end
                    if (cp_total && !seq_rd) begin
                        ang_in <= 1'b1;
                        ang_re <= {{8{qr[39]}}, qr};
                        ang_im <= {{8{qi[39]}}, qi};
                        sq <= Q_CPANG;
                    end
                end
                Q_CPANG: if (ang_ov) begin
                    info_ang_cp <= {delta17[16], delta17};
                    inc <= inc1 + {{7{delta17[16]}}, delta17, 8'd0};        // delta 2^32 / (65536 N)
                    sq <= Q_INFO;
                end
                Q_INFO: begin
                    info_valid <= 1'b1;
                    info_base <= base;
                    info_start <= base + start_rel;
                    info_peak <= base + cd;
                    info_ang_coarse <= a1;
                    info_inc <= inc;
                    nco_inc <= inc;
                    nco_ph0 <= -((start_rel - {26'd0, r_backoff}) * inc);
                    cnt <= 13'd0;
                    nco_first <= 1'b1;
                    o_m <= 13'd0;
                    sq <= Q_ALIGN;
                end
                Q_ALIGN: if (cnt == flen[12:0] && !seq_rd)
                    sq <= Q_DRAIN;
                default: if (o_fly == 6'd0)      // Q_DRAIN: the last samples are in the FIFO
                    sq <= Q_IDLE;
            endcase
            // disabled: drop whatever is in progress (once the NCO is empty)
            if (!r_en && o_fly == 6'd0 && !seq_rd) begin
                sq <= Q_IDLE;
                pend <= 1'b0;
                o_wp <= 5'd0;
                o_rp <= 5'd0;
                o_cnt <= 6'd0;
            end
        end
    end
endmodule
