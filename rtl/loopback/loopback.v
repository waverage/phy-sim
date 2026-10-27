`timescale 1ns / 1ps
// The board loopback (M8): transmitted frames from the ARM, the channel
// emulator, the frame receiver and the error counters, behind one register
// bus (loopback_axi puts it on the Zynq's AXI GP0). harness/loopback.py is
// its software model, bit exact.
//
// A run of NFRAMES frames r = 0, 1, ...: TX frame r mod NTX through
// chan_emu with seed SEED0 + r into the noisy-frame memory, then rx_frame
// on it; every decoded bit is compared with the frame's information bits.
// The counters (frames, frame errors, codeword errors, bit errors, decoder
// iterations, clocks) accumulate until CTRL bit 1 clears them.
//
// Word addresses (bus_addr):
//   0x00000 registers: 0 CTRL (w: bit 0 start a run, bit 1 clear counters)
//           1 STATUS (r: bit 0 busy)  2 NFRAMES  3 SEED0  4 GX  5 GN  6 NTX
//           7 TXLEN (samples a frame)  8 K (information bits a codeword)
//           16.. counters (r): 16 frames 17 frame errors 18 codeword errors
//           19 bit errors 20 iterations 21 clocks (low) 22 clocks (high)
//   0x10000 rx_frame configuration (cfg_addr = bus_addr[15:0], write only)
//   0x20000 TX frames: frame f sample i at f 4096 + i, {re[11:0], im[11:0]}
//   0x28000 information bits: frame f bit b at word f 512 + b / 32, bit b mod 32
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
    reg [13:0] r_k;
    wire reg_we = bus_we && bus_addr[17:6] == 12'd0;
    wire start_run = reg_we && bus_addr[5:0] == 6'd0 && bus_wdata[0];
    wire clear = reg_we && bus_addr[5:0] == 6'd0 && bus_wdata[1];
    always @(posedge clk) begin
        if (reg_we) begin
            case (bus_addr[5:0])
                6'd2: r_nframes <= bus_wdata;
                6'd3: r_seed0   <= bus_wdata;
                6'd4: r_gx      <= bus_wdata;
                6'd5: r_gn      <= bus_wdata;
                6'd6: r_ntx     <= bus_wdata[3:0];
                6'd7: r_txlen   <= bus_wdata[12:0];
                6'd8: r_k       <= bus_wdata[13:0];
                default: ;
            endcase
        end
    end

    // ---- memories
    reg [23:0] m_tx [0:32767];
    reg [31:0] m_ref [0:4095];
    reg [31:0] m_noisy [0:4095];
    reg [14:0] tx_ra;
    reg [11:0] ref_ra;
    reg [11:0] nz_ra, nz_wa;
    reg        nz_we;
    reg [31:0] nz_wd;
    reg [23:0] tx_q;
    reg [31:0] ref_q, nz_q;
    always @(posedge clk) begin
        if (bus_we && bus_addr[17:15] == 3'b100)
            m_tx[bus_addr[14:0]] <= bus_wdata[23:0];
        tx_q <= m_tx[tx_ra];
    end
    always @(posedge clk) begin
        if (bus_we && bus_addr[17:12] == 6'b101000)
            m_ref[bus_addr[11:0]] <= bus_wdata;
        ref_q <= m_ref[ref_ra];
    end
    always @(posedge clk) begin
        if (nz_we)
            m_noisy[nz_wa] <= nz_wd;
        nz_q <= m_noisy[nz_ra];
    end

    // ---- the run
    localparam R_IDLE = 3'd0, R_SEED = 3'd1, R_EMU = 3'd2, R_EDRAIN = 3'd3, R_RX = 3'd4, R_NEXT = 3'd5;
    reg [2:0]  st;
    reg [31:0] r;                              // frame of the run
    reg [3:0]  tf;                             // its TX frame
    reg [12:0] cnt;
    reg        emu_start;
    wire       emu_ready;
    reg        emu_in;
    wire       emu_ov;
    wire signed [15:0] emu_re, emu_im;
    wire [11:0] txr = tx_q[23:12], txi = tx_q[11:0];
    chan_emu emu (
        .clk(clk), .rst_n(rst_n), .start(emu_start), .seed(r_seed0 + r), .gx(r_gx), .gn(r_gn), .ready(emu_ready),
        .in_valid(emu_in), .in_re({{4{txr[11]}}, txr}), .in_im({{4{txi[11]}}, txi}),
        .out_valid(emu_ov), .out_re(emu_re), .out_im(emu_im));

    // the noisy frame -> rx_frame, through a 4-entry FIFO (rx_frame stalls its input)
    reg  [31:0] fq [0:3];
    reg  [2:0]  fcount;
    reg  [1:0]  fw, fr;
    reg         rd_v;                          // a noisy-memory read in flight
    reg  [12:0] fed;
    wire        rx_ready;
    wire        rx_in = (fcount != 3'd0);
    wire        pop = rx_in && rx_ready;
    wire        issue = (st == R_RX) && (fed != r_txlen) && (fcount + {2'd0, rd_v} < 3'd3);
    reg         rx_start;
    wire        rx_ov, rx_obit, rx_done;
    wire [13:0] rx_oidx;
    wire [1:0]  rx_ocw;
    wire [9:0]  rx_iters;
    wire        rx_cfg_we = bus_we && bus_addr[17:16] == 2'b01;
    /* verilator lint_off PINCONNECTEMPTY */
    rx_frame rx (
        .clk(clk), .rst_n(rst_n), .cfg_we(rx_cfg_we), .cfg_addr(bus_addr[15:0]), .cfg_data(bus_wdata),
        .start(rx_start), .busy(), .in_valid(rx_in), .in_re(fq[fr][31:16]), .in_im(fq[fr][15:0]),
        .in_ready(rx_ready), .out_valid(rx_ov), .out_bit(rx_obit), .out_idx(rx_oidx), .out_cw(rx_ocw),
        .done(rx_done), .iters(rx_iters),
        .dbg_y_we(), .dbg_y_addr(), .dbg_y_data(), .dbg_hf_we(), .dbg_hf_addr(), .dbg_hf_data(),
        .dbg_phi_valid(), .dbg_phi_idx(), .dbg_phi(), .dbg_h_valid(), .dbg_h(), .dbg_dy(),
        .dbg_llr_valid(), .dbg_llr_idx(), .dbg_llr());
    /* verilator lint_on PINCONNECTEMPTY */

    // ---- comparison: the reference word of each output bit, a clock later
    wire [13:0] k2 = {r_k[12:0], 1'b0};
    wire [13:0] bit_ix = (rx_ocw == 2'd0) ? rx_oidx : (rx_ocw == 2'd1) ? rx_oidx + r_k : rx_oidx + k2;
    reg        c_v;
    reg        c_bit;
    reg [4:0]  c_sel;
    reg [1:0]  c_cw;
    reg [3:0]  cw_err;
    reg [15:0] frame_bits_err;
    wire       c_err = c_v && (c_bit != ref_q[c_sel]);

    // ---- counters
    reg [31:0] n_frames, n_ferr, n_cwerr, n_biterr, n_iters;
    reg [63:0] n_clk;

    always @(*) begin
        tx_ra  = {tf[2:0], cnt[11:0]};
        nz_ra  = fed[11:0];
        ref_ra = {tf[2:0], bit_ix[13:5]};
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= R_IDLE;
            emu_start <= 1'b0;
            emu_in <= 1'b0;
            nz_we <= 1'b0;
            rx_start <= 1'b0;
            rd_v <= 1'b0;
            fcount <= 3'd0;
            fw <= 2'd0;
            fr <= 2'd0;
            c_v <= 1'b0;
        end else begin
            emu_start <= 1'b0;
            rx_start <= 1'b0;
            emu_in <= (st == R_EMU);
            // emulator output -> noisy memory, in order
            nz_we <= emu_ov;
            nz_wd <= {emu_re, emu_im};
            if (nz_we)
                nz_wa <= nz_wa + 12'd1;
            // FIFO: the read issued last clock lands now
            rd_v <= issue;
            if (issue)
                fed <= fed + 13'd1;
            if (rd_v) begin
                fq[fw] <= nz_q;
                fw <= fw + 2'd1;
            end
            if (pop)
                fr <= fr + 2'd1;
            fcount <= fcount + {2'd0, rd_v} - {2'd0, pop};
            // comparison
            c_v <= rx_ov;
            c_bit <= rx_obit;
            c_sel <= bit_ix[4:0];
            c_cw <= rx_ocw;
            if (c_err) begin
                cw_err[c_cw] <= 1'b1;
                frame_bits_err <= frame_bits_err + 16'd1;
            end

            case (st)
                R_IDLE: if (start_run) begin
                    st <= R_SEED;
                    r <= 32'd0;
                    tf <= 4'd0;
                    emu_start <= 1'b1;
                end
                R_SEED: if (emu_ready && !emu_start) begin       // base = h(seed) is ready
                    st <= R_EMU;
                    cnt <= 13'd0;
                    nz_wa <= 12'd0;
                end
                R_EMU: begin                    // TX frame -> emulator: read at cnt, the sample a clock later
                    cnt <= cnt + 13'd1;
                    if (cnt == r_txlen - 13'd1)
                        st <= R_EDRAIN;
                end
                R_EDRAIN: if (nz_wa == r_txlen[11:0] && !nz_we && !emu_ov) begin
                    st <= R_RX;
                    fed <= 13'd0;
                    rx_start <= 1'b1;
                    cw_err <= 4'd0;
                    frame_bits_err <= 16'd0;
                end
                R_RX: if (rx_done) begin
                    st <= R_NEXT;
                end
                default: if (!c_v) begin        // R_NEXT: count the frame
                    r <= r + 32'd1;
                    tf <= (tf == r_ntx - 4'd1) ? 4'd0 : tf + 4'd1;
                    if (r + 32'd1 == r_nframes)
                        st <= R_IDLE;
                    else begin
                        st <= R_SEED;
                        emu_start <= 1'b1;
                    end
                end
            endcase
        end
    end

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            n_frames <= 32'd0;
            n_ferr <= 32'd0;
            n_cwerr <= 32'd0;
            n_biterr <= 32'd0;
            n_iters <= 32'd0;
            n_clk <= 64'd0;
        end else begin
            if (st != R_IDLE)
                n_clk <= n_clk + 64'd1;
            if (st == R_NEXT && !c_v) begin
                n_frames <= n_frames + 32'd1;
                n_ferr <= n_ferr + {31'd0, cw_err != 4'd0};
                n_cwerr <= n_cwerr + {31'd0, cw_err[0]} + {31'd0, cw_err[1]} + {31'd0, cw_err[2]};
                n_biterr <= n_biterr + {16'd0, frame_bits_err};
            end
            if (rx_done)
                n_iters <= n_iters + {22'd0, rx_iters};
        end
    end

    always @(posedge clk) begin
        case (bus_addr[5:0])
            6'd1:  bus_rdata <= {31'd0, st != R_IDLE};
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
            default: bus_rdata <= 32'h4e505859;      // "NPXY"
        endcase
    end
endmodule
