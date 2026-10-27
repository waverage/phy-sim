`timescale 1ns / 1ps
// The whole radio on the chip (M9): rtl/tx, an emulated air between two
// radios, rtl/sync and rtl/rx, with the error counters, behind one register
// bus (onchip_axi puts it on the Zynq's AXI GP0). harness/onchip.py is its
// software model, bit exact.
//
// A run of NFRAMES frames r = 0, 1, ...: segment r is GAP samples of
// silence, then TX frame r (its preamble and S symbols, the information
// bits of slot r mod NTX); a last segment of silence follows. The emulator
// turns the stream by CFO_INC a sample (an NCO from the run's first
// sample), then chan_emu adds noise, segment r with seed SEED0 + r. One
// sample every PACE clocks on average (10: 10 MS/s at 100 MHz, the radio's
// rate; 6 the fastest the receiver keeps up with: nothing holds the air
// back, so a faster stream overruns the synchroniser; 0, as fast as the
// FIFO takes them, is for tests only) goes to the synchroniser; its aligned frames
// go to the receiver, whose decoded words are compared with the slot's bits.
// Nothing waits for the receiver but the synchroniser's output: a receiver
// too slow shows as synchroniser overruns.
//
// A synchronised frame is TX frame r when its start is within 64 samples
// of r's true start; frames passed over are missed, one matching none is
// spurious (decoded, not counted).
//
// Word addresses (bus_addr):
//   0x00000 registers: 0 CTRL (w: bit 0 start a run, bit 1 clear counters)
//           1 STATUS (r: bit 0 busy)  2 NFRAMES  3 SEED0  4 GX  5 GN  6 NTX
//           7 FLEN (TX samples a frame: preamble and S symbols)  8 NCW
//           9 OUTW  10 PACE  11 CFO_INC  12 GAP
//           16.. counters (r): 16 frames decoded 17 their frame errors
//           18 codeword errors 19 bit errors 20 iterations 21 clocks (low)
//           22 clocks (high) 23 missed 24 spurious 25 sync overruns
//           26 sync rejections
//   0x10000 rx_frame configuration (16-bit cfg_addr, write only)
//   0x20000 tx_frame configuration (15-bit cfg_addr, write only)
//   0x28000 information bits in rx_frame's output layout: slot f, codeword
//           c, word w at f 1024 + c OUTW + w (16 bits)
//   0x2C000 rx_sync configuration (10-bit cfg_addr, write only; its enable
//           is the run's)
module onchip (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        bus_we,
    input  wire [17:0] bus_addr,
    input  wire [31:0] bus_wdata,
    output reg  [31:0] bus_rdata
);
    // ---- registers
    reg [31:0] r_nframes, r_seed0, r_gx, r_gn, r_cfo;
    reg [3:0]  r_ntx;
    reg [12:0] r_flen;
    reg [1:0]  r_ncw;
    reg [9:0]  r_outw, r_gap;
    reg [7:0]  r_pace;
    wire reg_we = bus_we && bus_addr[17:6] == 12'd0;
    wire start_run = reg_we && bus_addr[5:0] == 6'd0 && bus_wdata[0];
    wire clear = reg_we && bus_addr[5:0] == 6'd0 && bus_wdata[1];
    always @(posedge clk) begin
        if (reg_we) begin
            case (bus_addr[5:0])
                6'd2:  r_nframes <= bus_wdata;
                6'd3:  r_seed0   <= bus_wdata;
                6'd4:  r_gx      <= bus_wdata;
                6'd5:  r_gn      <= bus_wdata;
                6'd6:  r_ntx     <= bus_wdata[3:0];
                6'd7:  r_flen    <= bus_wdata[12:0];
                6'd8:  r_ncw     <= bus_wdata[1:0];
                6'd9:  r_outw    <= bus_wdata[9:0];
                6'd10: r_pace    <= bus_wdata[7:0];
                6'd11: r_cfo     <= bus_wdata;
                6'd12: r_gap     <= bus_wdata[9:0];
                default: ;
            endcase
        end
    end
    wire [12:0] seg_len = {3'd0, r_gap} + r_flen;

    // ---- the frames' information bits: the TX reads them, the comparison too
    reg  [15:0] m_ref [0:8191];
    wire [12:0] tx_ia, cmp_ia;
    reg  [15:0] tx_iq, cmp_iq;
    always @(posedge clk) begin
        if (bus_we && bus_addr[17:13] == 5'b10100)
            m_ref[bus_addr[12:0]] <= bus_wdata[15:0];
        tx_iq <= m_ref[tx_ia];
    end
    always @(posedge clk)
        cmp_iq <= m_ref[cmp_ia];

    // ================= the transmitter =================
    reg         tx_go;
    wire        tx_ov;
    /* verilator lint_off UNUSEDSIGNAL */
    wire        tx_idle, tx_first;
    wire [31:0] tx_ifr, tx_nfr;
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [11:0] tx_re, tx_im;
    wire        tx_take;
    wire [10:0] tx_iaddr;
    tx_frame tx (
        .clk(clk), .rst_n(rst_n), .cfg_we(bus_we && bus_addr[17:15] == 3'b100), .cfg_addr({1'b0, bus_addr[14:0]}),
        .cfg_data(bus_wdata), .go(tx_go), .idle(tx_idle), .info_frame(tx_ifr), .info_addr(tx_iaddr),
        .info_data(tx_iq), .out_valid(tx_ov), .out_re(tx_re), .out_im(tx_im), .out_first(tx_first),
        .out_ready(tx_take), .n_frames(tx_nfr));
    // the slot of the frame the TX encodes: a new frame number (with its
    // first info read) moves it on; the TX is asked for NFRAMES frames
    reg  [31:0] tx_fr_seen, tx_enc;
    reg  [3:0]  tx_slot;
    wire        tx_new = (tx_ifr != tx_fr_seen);
    wire [3:0]  tx_slot_now = !tx_new ? tx_slot : (tx_slot == r_ntx - 4'd1) ? 4'd0 : tx_slot + 4'd1;
    assign tx_ia = {tx_slot_now[2:0], 10'd0} + {2'd0, tx_iaddr};

    // ================= the source =================
    localparam S_IDLE = 2'd0, S_GAP = 2'd1, S_FRAME = 2'd2;
    reg  [1:0]  sst;
    reg  [31:0] seg;                           // segment being sent
    reg  [12:0] scnt;
    reg         tail;                          // the last segment: silence
    reg  [7:0]  pc;
    reg  [15:0] credit;
    wire        paced = (r_pace != 8'd0);
    // the NCO and the FIFO behind it must have room (the unpaced source waits)
    reg  [5:0]  q_n, fly;
    wire        room = ({1'b0, q_n} + {1'b0, fly}) < 7'd28;
    wire        slot_ok = paced ? (credit != 16'd0) : room;
    wire        src_issue = slot_ok && ((sst == S_GAP) || (sst == S_FRAME && (tail || tx_ov)));
    assign      tx_take = (sst == S_FRAME) && !tail && slot_ok;
    wire signed [11:0] s_re = (sst == S_FRAME && !tail) ? tx_re : 12'sd0;
    wire signed [11:0] s_im = (sst == S_FRAME && !tail) ? tx_im : 12'sd0;
    reg         nco_first;
    wire        nco_ov;
    wire signed [15:0] nco_re, nco_im;
    nco_rot cfo (
        .clk(clk), .rst_n(rst_n), .start(src_issue && nco_first), .phase0(32'd0), .inc(32'd0 - r_cfo),
        .in_valid(src_issue), .in_re({{4{s_re[11]}}, s_re}), .in_im({{4{s_im[11]}}, s_im}), .out_valid(nco_ov),
        .out_re(nco_re), .out_im(nco_im));

    // NCO -> a FIFO -> chan_emu: each segment's seed loads (8 clocks) before
    // its first sample goes in, after the last sample of the segment before
    reg  [31:0] m_q [0:31];
    reg  [4:0]  q_w, q_r;
    reg  [12:0] e_left;                        // samples of the current segment still to go into chan_emu
    reg  [31:0] e_seg;                         // its number
    reg         e_wait, e_started;
    reg         emu_start, e_in;
    reg signed [15:0] e_re, e_im;
    wire        emu_ready, emu_ov;
    wire signed [15:0] emu_re, emu_im;
    wire        q_pop = !e_wait && !emu_start && q_n != 6'd0 && e_left != 13'd0;
    chan_emu emu (
        .clk(clk), .rst_n(rst_n), .start(emu_start), .seed(r_seed0 + e_seg), .gx(r_gx), .gn(r_gn),
        .ready(emu_ready), .in_valid(e_in), .in_re(e_re), .in_im(e_im), .out_valid(emu_ov), .out_re(emu_re),
        .out_im(emu_im));

    // ================= the receiver side =================
    reg         sync_we;
    reg  [9:0]  sync_a;
    reg  [31:0] sync_d;
    reg         en_now, en_val;
    always @(*) begin
        sync_we = bus_we && bus_addr[17:12] == 6'b101100;
        sync_a = bus_addr[9:0];
        sync_d = bus_wdata;
        if (en_now) begin
            sync_we = 1'b1;
            sync_a = 10'd12;
            sync_d = {31'd0, en_val};
        end
    end
    wire        sy_ov, sy_first, sy_info;
    wire signed [15:0] sy_re, sy_im;
    wire [31:0] sy_start, sy_nover, sy_nrej;
    reg         rx_feed;
    reg  [12:0] rx_fed;
    wire        rx_ready, rx_can, rx_idle;
    wire        sy_take = rx_feed && rx_ready && sy_ov;
    reg  [31:0] fed_total;                     // samples into the synchroniser since reset (its count)
    /* verilator lint_off PINCONNECTEMPTY */
    rx_sync sync (
        .clk(clk), .rst_n(rst_n), .cfg_we(sync_we), .cfg_addr(sync_a), .cfg_data(sync_d),
        .in_valid(emu_ov), .in_re(emu_re[11:0]), .in_im(emu_im[11:0]), .out_valid(sy_ov), .out_re(sy_re),
        .out_im(sy_im), .out_first(sy_first), .out_ready(sy_take), .info_valid(sy_info), .info_base(),
        .info_start(sy_start), .info_peak(), .info_tau(), .info_ang_coarse(), .info_ang_cp(), .info_inc(),
        .info_cir_peak(), .info_cir_sum(), .n_detect(), .n_reject(sy_nrej), .n_sync(), .n_overrun(sy_nover));
    /* verilator lint_on PINCONNECTEMPTY */

    reg         rx_start;
    wire        o_v, o_done;
    wire [15:0] o_word, o_mask;
    wire [9:0]  o_addr;
    wire [1:0]  o_cw;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [7:0]  o_tag;
    wire [12:0] emu_hi = {emu_re[15:12], emu_im[15:12], 5'd0};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [4:0]  o_iters;
    /* verilator lint_off PINCONNECTEMPTY */
    rx_frame rx (
        .clk(clk), .rst_n(rst_n), .cfg_we(bus_we && bus_addr[17:16] == 2'b01), .cfg_addr(bus_addr[15:0]),
        .cfg_data(bus_wdata), .start(rx_start), .can_start(rx_can), .idle(rx_idle), .in_valid(rx_feed && sy_ov),
        .in_re(sy_re), .in_im(sy_im), .in_ready(rx_ready), .out_valid(o_v), .out_word(o_word), .out_mask(o_mask),
        .out_addr(o_addr), .out_cw(o_cw), .out_tag(o_tag), .cw_done(o_done), .cw_iters(o_iters),
        .dbg_y_we(), .dbg_y_addr(), .dbg_y_data(), .dbg_hf_we(), .dbg_hf_addr(), .dbg_hf_data(),
        .dbg_phi_valid(), .dbg_phi_idx(), .dbg_phi(), .dbg_h_valid(), .dbg_h(), .dbg_dy(),
        .dbg_llr_valid(), .dbg_llr_idx(), .dbg_llr(), .dbg_b_tag());
    /* verilator lint_on PINCONNECTEMPTY */
    wire [12:0] rx_len = r_flen - 13'd288;     // the aligned frame: S slot samples

    // the matcher: each synchronised frame's start against the true starts, in order
    reg  [31:0] m_r;                           // the TX frame it may be next
    reg  [31:0] m_true;                        // its true start (the synchroniser's count)
    reg  [3:0]  m_slot;
    reg         m_busy;
    reg  [31:0] m_start;
    reg  [3:0]  mq [0:3];                      // results waiting for the receiver: the slot, 15 spurious
    reg  [1:0]  mq_w, mq_r;
    reg  [3:0]  tagmap [0:7];                  // the slot of a tag in flight
    reg  [7:0]  rtag;

    // the comparison (rtl/loopback's)
    wire [9:0]  cwbase = (o_cw == 2'd0) ? 10'd0 : (o_cw == 2'd1) ? r_outw : {r_outw[8:0], 1'b0};
    assign      cmp_ia = {tagmap[o_tag[2:0]][2:0], cwbase + o_addr};
    reg         c1_v, c2_v;
    reg  [15:0] c1_word, c1_mask;
    reg  [4:0]  c2_n;
    reg  [15:0] cw_bits;
    reg  [4:0]  pop16;
    reg  [15:0] diff;
    integer pb;
    always @(*) begin
        diff = (c1_word ^ cmp_iq) & c1_mask;
        pop16 = 5'd0;
        for (pb = 0; pb < 16; pb = pb + 1)
            pop16 = pop16 + {4'd0, diff[pb]};
    end
    reg         pend;
    reg  [2:0]  pend_tag;
    reg  [4:0]  pend_iters;
    reg  [1:0]  slot_n [0:7];
    reg         slot_e [0:7];
    wire        cw_end = pend && !c1_v && !c2_v;
    wire        pend_spur = (tagmap[pend_tag] == 4'd15);
    wire        frame_end = cw_end && slot_n[pend_tag] == r_ncw - 2'd1;

    // the run's end: the source is done, every TX frame matched or passed
    // over, the receiver idle; or, after the source, a long quiet time
    reg         run, src_done;
    reg  [19:0] quiet;
    wire        settled = src_done && rx_idle && !rx_feed && !m_busy && mq_w == mq_r && !pend && !c1_v && !c2_v &&
                          !o_v && !rx_start;

    // ---- counters
    reg [31:0] n_frames, n_ferr, n_cwerr, n_biterr, n_iters, n_missed, n_spur;
    reg [63:0] n_clk;

    integer si;
    always @(posedge clk) begin
        if (!rst_n) begin
            run <= 1'b0;
            src_done <= 1'b0;
            tx_go <= 1'b0;
            sst <= S_IDLE;
            emu_start <= 1'b0;
            e_in <= 1'b0;
            e_wait <= 1'b0;
            e_started <= 1'b0;
            q_w <= 5'd0;
            q_r <= 5'd0;
            q_n <= 6'd0;
            fly <= 6'd0;
            en_now <= 1'b0;
            rx_feed <= 1'b0;
            rx_start <= 1'b0;
            rtag <= 8'd0;
            mq_w <= 2'd0;
            mq_r <= 2'd0;
            m_busy <= 1'b0;
            c1_v <= 1'b0;
            c2_v <= 1'b0;
            pend <= 1'b0;
            cw_bits <= 16'd0;
            fed_total <= 32'd0;
            for (si = 0; si < 8; si = si + 1) begin
                slot_n[si] <= 2'd0;
                slot_e[si] <= 1'b0;
            end
        end else begin
            emu_start <= 1'b0;
            rx_start <= 1'b0;
            en_now <= 1'b0;
            e_in <= 1'b0;
            if (emu_ov)
                fed_total <= fed_total + 32'd1;

            // ---- the pace
            if (sst == S_IDLE || pc == r_pace - 8'd1)
                pc <= 8'd0;
            else
                pc <= pc + 8'd1;
            credit <= (sst == S_IDLE) ? 16'd0 : credit + {15'd0, paced && pc == r_pace - 8'd1} -
                                                          {15'd0, src_issue && paced};

            // ---- the TX's slots, and no more frames than the run's
            if (tx_new) begin
                tx_fr_seen <= tx_ifr;
                tx_slot <= tx_slot_now;
                tx_enc <= tx_enc + 32'd1;
                if (tx_enc + 32'd1 == r_nframes)
                    tx_go <= 1'b0;
            end

            // ---- the source's segments
            if (src_issue)
                nco_first <= 1'b0;
            case (sst)
                S_IDLE: if (start_run && !run) begin
                    run <= 1'b1;
                    src_done <= 1'b0;
                    quiet <= 20'd0;
                    tx_go <= 1'b1;
                    en_now <= 1'b1;            // the synchroniser searches from the run's first sample
                    en_val <= 1'b1;
                    seg <= 32'd0;
                    tail <= 1'b0;
                    scnt <= 13'd0;
                    nco_first <= 1'b1;
                    e_seg <= 32'd0;
                    e_left <= 13'd0;
                    e_started <= 1'b0;
                    tx_fr_seen <= tx_ifr;
                    tx_slot <= r_ntx - 4'd1;   // the TX's next frame number moves it to slot 0
                    tx_enc <= 32'd0;
                    m_r <= 32'd0;
                    m_true <= fed_total + {22'd0, r_gap} + 32'd288;
                    m_slot <= 4'd0;
                    sst <= S_GAP;
                end
                S_GAP: if (src_issue) begin
                    scnt <= scnt + 13'd1;
                    if (scnt == {3'd0, r_gap} - 13'd1) begin
                        scnt <= 13'd0;
                        sst <= S_FRAME;
                    end
                end
                default: if (src_issue) begin  // S_FRAME: the TX frame, or the tail's silence
                    scnt <= scnt + 13'd1;
                    if (scnt == r_flen - 13'd1) begin
                        scnt <= 13'd0;
                        if (tail) begin
                            sst <= S_IDLE;
                            src_done <= 1'b1;
                        end else begin
                            seg <= seg + 32'd1;
                            if (seg + 32'd1 == r_nframes)
                                tail <= 1'b1;
                            sst <= S_GAP;
                        end
                    end
                end
            endcase
            fly <= fly + {5'd0, src_issue} - {5'd0, nco_ov};

            // ---- NCO -> FIFO -> chan_emu
            if (nco_ov) begin
                m_q[q_w] <= {nco_re, nco_im};
                q_w <= q_w + 5'd1;
            end
            if (q_pop) begin
                e_in <= 1'b1;
                {e_re, e_im} <= m_q[q_r];
                q_r <= q_r + 5'd1;
                e_left <= e_left - 13'd1;
            end
            q_n <= q_n + {5'd0, nco_ov} - {5'd0, q_pop};
            if (!e_wait && !emu_start && e_left == 13'd0 && q_n != 6'd0) begin
                // the next segment's first sample is at the FIFO's head: its seed
                emu_start <= 1'b1;
                e_wait <= 1'b1;
                if (e_started)
                    e_seg <= e_seg + 32'd1;
                e_started <= 1'b1;
            end
            if (e_wait && !emu_start && emu_ready) begin
                e_wait <= 1'b0;
                e_left <= seg_len;
            end

            // ---- the matcher
            if (sy_info && run) begin
                m_start <= sy_start;
                m_busy <= 1'b1;
            end
            if (m_busy) begin
                if (m_r < r_nframes && m_start > m_true + 32'd64) begin
                    m_r <= m_r + 32'd1;        // TX frame m_r was passed over
                    m_true <= m_true + {19'd0, seg_len};
                    m_slot <= (m_slot == r_ntx - 4'd1) ? 4'd0 : m_slot + 4'd1;
                end else begin
                    m_busy <= 1'b0;
                    mq_w <= mq_w + 2'd1;
                    if (m_r < r_nframes && m_start + 32'd64 >= m_true) begin
                        mq[mq_w] <= m_slot;
                        m_r <= m_r + 32'd1;
                        m_true <= m_true + {19'd0, seg_len};
                        m_slot <= (m_slot == r_ntx - 4'd1) ? 4'd0 : m_slot + 4'd1;
                    end else
                        mq[mq_w] <= 4'd15;
                end
            end

            // ---- synchronised frames into the receiver, each with its match
            if (!rx_feed && !rx_start && sy_ov && sy_first && mq_w != mq_r && rx_can) begin
                rx_start <= 1'b1;
                rx_feed <= 1'b1;
                rx_fed <= 13'd0;
                tagmap[rtag[2:0]] <= mq[mq_r];
                rtag <= rtag + 8'd1;
                mq_r <= mq_r + 2'd1;
            end
            if (sy_take) begin
                rx_fed <= rx_fed + 13'd1;
                if (rx_fed == rx_len - 13'd1)
                    rx_feed <= 1'b0;
            end

            // ---- comparison
            c1_v <= o_v;
            c1_word <= o_word;
            c1_mask <= o_mask;
            c2_v <= c1_v;
            c2_n <= pop16;
            if (o_done) begin
                pend <= 1'b1;
                pend_tag <= o_tag[2:0];
                pend_iters <= o_iters;
            end
            if (c2_v)
                cw_bits <= cw_bits + {11'd0, c2_n};
            if (cw_end) begin
                pend <= 1'b0;
                cw_bits <= 16'd0;
                if (slot_n[pend_tag] == r_ncw - 2'd1) begin
                    slot_n[pend_tag] <= 2'd0;
                    slot_e[pend_tag] <= 1'b0;
                end else begin
                    slot_n[pend_tag] <= slot_n[pend_tag] + 2'd1;
                    slot_e[pend_tag] <= slot_e[pend_tag] || (cw_bits != 16'd0);
                end
            end

            // ---- the end of the run
            if (run && src_done) begin
                quiet <= (sy_ov || o_v || rx_feed) ? 20'd0 : quiet + 20'd1;
                if (settled && (m_r == r_nframes || quiet == 20'hfffff)) begin
                    run <= 1'b0;
                    en_now <= 1'b1;            // the synchroniser stops (a detection in the tail is dropped)
                    en_val <= 1'b0;
                end
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            n_frames <= 32'd0;
            n_ferr <= 32'd0;
            n_cwerr <= 32'd0;
            n_biterr <= 32'd0;
            n_iters <= 32'd0;
            n_missed <= 32'd0;
            n_spur <= 32'd0;
            n_clk <= 64'd0;
        end else begin
            if (run)
                n_clk <= n_clk + 64'd1;
            if (cw_end && !pend_spur) begin
                n_cwerr <= n_cwerr + {31'd0, cw_bits != 16'd0};
                n_biterr <= n_biterr + {16'd0, cw_bits};
                n_iters <= n_iters + {27'd0, pend_iters};
            end
            if (frame_end && !pend_spur) begin
                n_frames <= n_frames + 32'd1;
                n_ferr <= n_ferr + {31'd0, slot_e[pend_tag] || (cw_bits != 16'd0)};
            end
            if (m_busy && m_r < r_nframes && m_start > m_true + 32'd64)
                n_missed <= n_missed + 32'd1;
            if (m_busy && !(m_r < r_nframes && m_start > m_true + 32'd64) &&
                !(m_r < r_nframes && m_start + 32'd64 >= m_true))
                n_spur <= n_spur + 32'd1;
            if (run && src_done && settled && quiet == 20'hfffff && m_r != r_nframes)
                n_missed <= n_missed + (r_nframes - m_r);   // never synchronised at all
        end
    end

    always @(posedge clk) begin
        case (bus_addr[5:0])
            6'd1:  bus_rdata <= {31'd0, run};
            6'd2:  bus_rdata <= r_nframes;
            6'd3:  bus_rdata <= r_seed0;
            6'd16: bus_rdata <= n_frames;
            6'd17: bus_rdata <= n_ferr;
            6'd18: bus_rdata <= n_cwerr;
            6'd19: bus_rdata <= n_biterr;
            6'd20: bus_rdata <= n_iters;
            6'd21: bus_rdata <= n_clk[31:0];
            6'd22: bus_rdata <= n_clk[63:32];
            6'd23: bus_rdata <= n_missed;
            6'd24: bus_rdata <= n_spur;
            6'd25: bus_rdata <= sy_nover;
            6'd26: bus_rdata <= sy_nrej;
            default: bus_rdata <= 32'h4e505859;      // "NPXY"
        endcase
    end
endmodule
