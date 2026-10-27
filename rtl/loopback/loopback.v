`timescale 1ns / 1ps
// The board loopback (M8, real time in M9): transmitted frames from the
// ARM, the channel emulator, the frame receiver and the error counters,
// behind one register bus (loopback_axi puts it on the Zynq's AXI GP0).
// harness/loopback.py is its software model, bit exact.
//
// A run of NFRAMES frames r = 0, 1, ...: TX frame r mod NTX goes through
// chan_emu with seed SEED0 + r, frame after frame without gaps, into a
// sample FIFO and from it into rx_frame, which works on three frames at
// once; every decoded word is compared with the frame's information bits.
//
// PACE sets the source: 0 as fast as the receiver takes the samples (the
// emulator waits for FIFO space), n > 0 one sample every n clocks on
// average like a radio (n = 10 is 10 MS/s at 100 MHz), which never waits:
// a frame that finds the receiver busy is dropped, a sample that finds the
// FIFO full is lost. A real-time run must end with no drops and no losses.
//
// The counters (frames decoded, frame errors, codeword errors, bit errors,
// decoder iterations, clocks of the run, frames dropped, samples lost, the
// FIFO's highest fill) accumulate until CTRL bit 1 clears them.
//
// Word addresses (bus_addr):
//   0x00000 registers: 0 CTRL (w: bit 0 start a run, bit 1 clear counters)
//           1 STATUS (r: bit 0 busy)  2 NFRAMES  3 SEED0  4 GX  5 GN  6 NTX
//           7 TXLEN (samples a frame)  8 NCW (codewords a frame)
//           9 OUTW (output words a codeword)  10 PACE
//           16.. counters (r): 16 frames 17 frame errors 18 codeword errors
//           19 bit errors 20 iterations 21 clocks (low) 22 clocks (high)
//           23 frames dropped 24 samples lost 25 FIFO high water
//           26 clocks a frame's start waited on an unmarked sample (must be 0)
//   0x10000 rx_frame configuration (cfg_addr = bus_addr[15:0], write only)
//   0x20000 TX frames: frame f sample i at f 4096 + i, {re[11:0], im[11:0]}
//   0x28000 information bits in rx_frame's output layout: frame f,
//           codeword c, output word w at f 1024 + c OUTW + w (16 bits)
module loopback (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        bus_we,
    input  wire [17:0] bus_addr,
    input  wire [31:0] bus_wdata,
    output reg  [31:0] bus_rdata
);
    // ---- registers
    reg [31:0] r_nframes, r_seed0, r_gx, r_gn;
    reg [3:0]  r_ntx;
    reg [12:0] r_txlen;
    reg [1:0]  r_ncw;
    reg [9:0]  r_outw;
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
                6'd7:  r_txlen   <= bus_wdata[12:0];
                6'd8:  r_ncw     <= bus_wdata[1:0];
                6'd9:  r_outw    <= bus_wdata[9:0];
                6'd10: r_pace    <= bus_wdata[7:0];
                default: ;
            endcase
        end
    end

    // ---- memories
    reg [23:0] m_tx [0:32767];
    reg [15:0] m_ref [0:8191];
    reg [14:0] tx_ra;
    reg [12:0] ref_ra;
    reg [23:0] tx_q;
    reg [15:0] ref_q;
    always @(posedge clk) begin
        if (bus_we && bus_addr[17:15] == 3'b100)
            m_tx[bus_addr[14:0]] <= bus_wdata[23:0];
        tx_q <= m_tx[tx_ra];
    end
    always @(posedge clk) begin
        if (bus_we && bus_addr[17:13] == 5'b10100)
            m_ref[bus_addr[12:0]] <= bus_wdata[15:0];
        ref_q <= m_ref[ref_ra];
    end

    // ================= the source: TX frames through the emulator =================
    localparam S_IDLE = 2'd0, S_SEED = 2'd1, S_EMU = 2'd2;
    reg [1:0]  st;
    reg        run;                            // the run is on (until its last frame is counted)
    reg [31:0] r;                              // the source's frame
    reg [3:0]  tf;                             // its TX frame
    reg [12:0] cnt;
    reg        emu_start, emu_in;
    wire       emu_ready, emu_ov;
    wire signed [15:0] emu_re, emu_im;
    wire [11:0] txr = tx_q[23:12], txi = tx_q[11:0];
    chan_emu emu (
        .clk(clk), .rst_n(rst_n), .start(emu_start), .seed(r_seed0 + r), .gx(r_gx), .gn(r_gn), .ready(emu_ready),
        .in_valid(emu_in), .in_re({{4{txr[11]}}, txr}), .in_im({{4{txi[11]}}, txi}),
        .out_valid(emu_ov), .out_re(emu_re), .out_im(emu_im));

    // the pace: a credit every PACE clocks, a sample spends one
    reg  [7:0]  pc;
    reg  [15:0] credit;
    // the FIFO: 1024 samples with a mark on each frame's first
    reg  [32:0] m_fifo [0:1023];
    reg  [9:0]  fw, fr;
    reg  [10:0] fcount;
    reg  [6:0]  inflight;                      // samples inside the emulator
    reg  [12:0] ocnt;                          // emulator outputs of the frame so far
    wire        paced = (r_pace != 8'd0);
    wire        room = (fcount + {4'd0, inflight}) < 11'd1000;
    wire        issue = (st == S_EMU) && (paced ? (credit != 16'd0) : room);
    always @(*) tx_ra = {tf[2:0], cnt[11:0]};

    // ================= the receiver side =================
    localparam C_WAIT = 2'd0, C_FEED = 2'd1, C_DROP = 2'd2;
    reg  [1:0]  cst;
    reg  [31:0] cr;                            // the frame at the FIFO's head
    reg  [3:0]  ctf;
    reg  [12:0] fed;
    reg  [32:0] f_head;
    wire        f_mark = f_head[32];           // a frame's first sample: the count must agree
    wire        f_any = (fcount != 11'd0);
    wire        rx_ready, rx_can, rx_idle;
    wire        rx_in = (cst == C_FEED) && f_any;
    wire        pop = (rx_in && rx_ready) || (cst == C_DROP && f_any);
    reg         rx_start;
    reg  [7:0]  rtag;                          // rx_frame's next tag
    reg  [3:0]  tagmap [0:7];                  // TX frame of a tag in flight
    wire        rx_cfg_we = bus_we && bus_addr[17:16] == 2'b01;
    wire        o_v, o_done;
    wire [15:0] o_word, o_mask;
    wire [9:0]  o_addr;
    wire [1:0]  o_cw;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [7:0]  o_tag;                         // eight frames in flight at most: the low bits suffice
    /* verilator lint_on UNUSEDSIGNAL */
    wire [4:0]  o_iters;
    // the head, registered: the word at the next read position, or the word
    // written there in the same clock (an empty FIFO)
    wire [9:0]  fr_next = pop ? fr + 10'd1 : fr;
    wire        f_wr = emu_ov && (fcount != 11'd1024 || pop);
    wire [32:0] f_wd = {ocnt == 13'd0, emu_re, emu_im};
    reg  [32:0] f_mem_q, f_byp_d;
    reg         f_byp;
    always @(posedge clk) begin
        if (f_wr)
            m_fifo[fw] <= f_wd;
        f_mem_q <= m_fifo[fr_next];
    end
    always @(posedge clk) begin
        f_byp <= f_wr && fw == fr_next;
        f_byp_d <= f_wd;
    end
    always @(*) f_head = f_byp ? f_byp_d : f_mem_q;
    /* verilator lint_off PINCONNECTEMPTY */
    rx_frame rx (
        .clk(clk), .rst_n(rst_n), .cfg_we(rx_cfg_we), .cfg_addr(bus_addr[15:0]), .cfg_data(bus_wdata),
        .start(rx_start), .can_start(rx_can), .idle(rx_idle), .in_valid(rx_in), .in_re(f_head[31:16]),
        .in_im(f_head[15:0]), .in_ready(rx_ready), .out_valid(o_v), .out_word(o_word), .out_mask(o_mask),
        .out_addr(o_addr), .out_cw(o_cw), .out_tag(o_tag), .cw_done(o_done), .cw_iters(o_iters),
        .dbg_y_we(), .dbg_y_addr(), .dbg_y_data(), .dbg_hf_we(), .dbg_hf_addr(), .dbg_hf_data(),
        .dbg_phi_valid(), .dbg_phi_idx(), .dbg_phi(), .dbg_h_valid(), .dbg_h(), .dbg_dy(),
        .dbg_llr_valid(), .dbg_llr_idx(), .dbg_llr(), .dbg_b_tag());
    /* verilator lint_on PINCONNECTEMPTY */

    // ---- comparison: the reference word, a clock later; its errors the next
    wire [9:0]  cwbase = (o_cw == 2'd0) ? 10'd0 : (o_cw == 2'd1) ? r_outw : {r_outw[8:0], 1'b0};
    always @(*) ref_ra = {tagmap[o_tag[2:0]][2:0], cwbase + o_addr};
    reg         c1_v, c2_v;
    reg  [15:0] c1_word, c1_mask;
    reg  [4:0]  c2_n;
    reg  [15:0] cw_bits;                       // the codeword's bit errors so far
    reg  [4:0]  pop16;
    reg  [15:0] diff;
    integer pb;
    always @(*) begin
        diff = (c1_word ^ ref_q) & c1_mask;
        pop16 = 5'd0;
        for (pb = 0; pb < 16; pb = pb + 1)
            pop16 = pop16 + {4'd0, diff[pb]};
    end
    // a codeword's end, counted once its last word is through the comparison
    reg         pend;
    reg  [2:0]  pend_tag;
    reg  [4:0]  pend_iters;
    reg  [1:0]  slot_n [0:7];                  // codewords of a tag done
    reg         slot_e [0:7];                  // and whether one failed

    // ---- counters
    reg [31:0] n_frames, n_ferr, n_cwerr, n_biterr, n_iters, n_drop, n_lost, n_mark;
    reg [63:0] n_clk;
    reg [10:0] n_high;

    integer si;
    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE;
            run <= 1'b0;
            emu_start <= 1'b0;
            emu_in <= 1'b0;
            cst <= C_WAIT;
            rx_start <= 1'b0;
            fw <= 10'd0;
            fr <= 10'd0;
            fcount <= 11'd0;
            inflight <= 7'd0;
            c1_v <= 1'b0;
            c2_v <= 1'b0;
            pend <= 1'b0;
            rtag <= 8'd0;
            cw_bits <= 16'd0;
            for (si = 0; si < 8; si = si + 1) begin
                slot_n[si] <= 2'd0;
                slot_e[si] <= 1'b0;
            end
        end else begin
            emu_start <= 1'b0;
            rx_start <= 1'b0;

            // ---- the source
            emu_in <= issue;
            inflight <= inflight + {6'd0, issue} - {6'd0, emu_ov};
            if (st == S_IDLE || pc == r_pace - 8'd1)
                pc <= 8'd0;
            else
                pc <= pc + 8'd1;
            credit <= (st == S_IDLE) ? 16'd0 : credit + {15'd0, paced && pc == r_pace - 8'd1} - {15'd0, issue && paced};
            case (st)
                S_IDLE: if (start_run) begin
                    st <= S_SEED;
                    run <= 1'b1;
                    r <= 32'd0;
                    tf <= 4'd0;
                    cr <= 32'd0;
                    ctf <= 4'd0;
                    ocnt <= 13'd0;
                    emu_start <= 1'b1;
                end
                S_SEED: if (emu_ready && !emu_start) begin       // base = h(seed) is ready
                    st <= S_EMU;
                    cnt <= 13'd0;
                end
                default: if (issue) begin       // S_EMU: read at cnt, the sample a clock later
                    cnt <= cnt + 13'd1;
                    if (cnt == r_txlen - 13'd1) begin
                        r <= r + 32'd1;
                        tf <= (tf == r_ntx - 4'd1) ? 4'd0 : tf + 4'd1;
                        if (r + 32'd1 == r_nframes)
                            st <= S_IDLE;
                        else begin
                            st <= S_SEED;
                            emu_start <= 1'b1;
                        end
                    end
                end
            endcase

            // ---- the FIFO
            if (emu_ov)
                ocnt <= (ocnt == r_txlen - 13'd1) ? 13'd0 : ocnt + 13'd1;
            if (f_wr)
                fw <= fw + 10'd1;
            if (pop)
                fr <= fr + 10'd1;
            fcount <= fcount + {10'd0, f_wr} - {10'd0, pop};

            // ---- the receiver side: a frame starts at its marked first sample.
            // The receiver may still be busy with the frame before (its last
            // FFT); a paced frame waits while the FIFO has room, then drops.
            case (cst)
                C_WAIT: if (f_any && run) begin
                    fed <= 13'd0;
                    if (rx_can) begin
                        rx_start <= 1'b1;
                        tagmap[rtag[2:0]] <= ctf;
                        rtag <= rtag + 8'd1;
                        cst <= C_FEED;
                    end else if (paced && fcount >= 11'd960)
                        cst <= C_DROP;
                end
                default: if (pop) begin          // C_FEED, C_DROP: the frame's samples
                    fed <= fed + 13'd1;
                    if (fed == r_txlen - 13'd1) begin
                        cst <= C_WAIT;
                        cr <= cr + 32'd1;
                        ctf <= (ctf == r_ntx - 4'd1) ? 4'd0 : ctf + 4'd1;
                    end
                end
            endcase

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
            if (pend && !c1_v && !c2_v) begin
                // the codeword is through; a new one's words are three clocks away at least
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
            // the run is over when every frame is decoded or dropped and the receiver is idle
            if (run && st == S_IDLE && !start_run && cr == r_nframes && rx_idle && !pend && !c1_v && !c2_v &&
                !o_v && !rx_start)
                run <= 1'b0;
        end
    end

    wire cw_end = pend && !c1_v && !c2_v;
    wire frame_end = cw_end && slot_n[pend_tag] == r_ncw - 2'd1;
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            n_frames <= 32'd0;
            n_ferr <= 32'd0;
            n_cwerr <= 32'd0;
            n_biterr <= 32'd0;
            n_iters <= 32'd0;
            n_clk <= 64'd0;
            n_drop <= 32'd0;
            n_lost <= 32'd0;
            n_mark <= 32'd0;
            n_high <= 11'd0;
        end else begin
            if (cst == C_WAIT && f_any && run && !f_mark)
                n_mark <= n_mark + 32'd1;
            if (run)
                n_clk <= n_clk + 64'd1;
            if (cw_end) begin
                n_cwerr <= n_cwerr + {31'd0, cw_bits != 16'd0};
                n_biterr <= n_biterr + {16'd0, cw_bits};
                n_iters <= n_iters + {27'd0, pend_iters};
            end
            if (frame_end) begin
                n_frames <= n_frames + 32'd1;
                n_ferr <= n_ferr + {31'd0, slot_e[pend_tag] || (cw_bits != 16'd0)};
            end
            if (cst == C_DROP && pop && fed == r_txlen - 13'd1)
                n_drop <= n_drop + 32'd1;
            if (emu_ov && fcount == 11'd1024 && !pop)
                n_lost <= n_lost + 32'd1;
            if (fcount > n_high)
                n_high <= fcount;
        end
    end

    always @(posedge clk) begin
        case (bus_addr[5:0])
            6'd1:  bus_rdata <= {31'd0, run};
            6'd2:  bus_rdata <= r_nframes;
            6'd3:  bus_rdata <= r_seed0;
            6'd4:  bus_rdata <= r_gx;
            6'd5:  bus_rdata <= r_gn;
            6'd16: bus_rdata <= n_frames;
            6'd17: bus_rdata <= n_ferr;
            6'd18: bus_rdata <= n_cwerr;
            6'd19: bus_rdata <= n_biterr;
            6'd20: bus_rdata <= n_iters;
            6'd21: bus_rdata <= n_clk[31:0];
            6'd22: bus_rdata <= n_clk[63:32];
            6'd23: bus_rdata <= n_drop;
            6'd24: bus_rdata <= n_lost;
            6'd25: bus_rdata <= {21'd0, n_high};
            6'd26: bus_rdata <= n_mark;
            default: bus_rdata <= 32'h4e505859;      // "NPXY"
        endcase
    end
endmodule
