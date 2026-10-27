`timescale 1ns / 1ps
// Receiver front end (fxmodem.rxFrame step 1, twin.TwinModem.rx): per OFDM
// symbol, drop the cyclic prefix, FFT the 256 samples (fft256), rotate bin b
// by the timing phase, twiddle (l_min b mod 256) rounded half up by 16 bits
// and saturated, and write the used subcarriers to the frame memory at
// s F + f. l_min = 0 rotates by exactly 1.
//
// The input is the aligned frame, S (256 + cp) samples (in_valid /
// in_ready); the FFT takes a symbol while its cyclic prefix is skipped,
// then stalls the input for the butterflies and the unload.
module rx_fe (
    input  wire               clk,
    input  wire               rst_n,
    input  wire        [5:0]  cp_len,
    input  wire        [3:0]  num_sym,
    input  wire        [7:0]  nf,
    input  wire        [7:0]  guard_lo,
    input  wire        [7:0]  guard_hi,
    input  wire               dc_null,
    input  wire        [7:0]  l_min,
    input  wire               start,
    input  wire               in_valid,
    input  wire signed [15:0] in_re,
    input  wire signed [15:0] in_im,
    output wire               in_ready,
    output reg                y_we,
    output reg         [11:0] y_addr,
    output reg         [31:0] y_data,
    output reg                done
);
    localparam I_IDLE = 2'd0, I_CP = 2'd1, I_DATA = 2'd2, I_END = 2'd3;
    reg [1:0] ist;
    reg [7:0] icnt;
    reg [3:0] isym;

    wire fft_ready;
    assign in_ready = (ist == I_CP) || (ist == I_DATA && fft_ready);
    wire fft_in = (ist == I_DATA) && in_valid && fft_ready;

    always @(posedge clk) begin
        if (!rst_n)
            ist <= I_IDLE;
        else case (ist)
            I_IDLE: if (start) begin
                ist  <= (cp_len == 6'd0) ? I_DATA : I_CP;
                icnt <= 8'd0;
                isym <= 4'd0;
            end
            I_CP: if (in_valid) begin
                icnt <= icnt + 8'd1;
                if (icnt == {2'b00, cp_len} - 8'd1) begin
                    ist  <= I_DATA;
                    icnt <= 8'd0;
                end
            end
            I_DATA: if (fft_in) begin
                icnt <= icnt + 8'd1;
                if (icnt == 8'd255) begin
                    icnt <= 8'd0;
                    isym <= isym + 4'd1;
                    ist  <= (isym == num_sym - 4'd1) ? I_END : ((cp_len == 6'd0) ? I_DATA : I_CP);
                end
            end
            default: if (done)
                ist <= I_IDLE;
        endcase
    end

    wire               fo_v;
    wire signed [15:0] fo_re, fo_im;
    wire        [7:0]  fo_idx;
    fft256 fft (
        .clk(clk), .rst_n(rst_n), .inverse(1'b0), .in_valid(fft_in), .in_re(in_re), .in_im(in_im),
        .in_ready(fft_ready), .out_valid(fo_v), .out_re(fo_re), .out_im(fo_im), .out_idx(fo_idx));

    // stage A, three clocks: j = l_min b mod 256, its twiddle, the products
    reg                j_v, w_v, a_v;
    reg         [7:0]  j, j_b, w_b, a_b;
    reg signed  [15:0] j_re, j_im, w_re, w_im;
    reg signed  [17:0] wr, wi;
    reg signed  [34:0] a_re, a_im;
    wire signed [17:0] twr, twi;
    fft256_twiddle rom (.j(j[6:0]), .wr(twr), .wi(twi));
    always @(posedge clk) begin
        j_v  <= fo_v & rst_n;
        j    <= l_min * fo_idx;
        j_b  <= fo_idx;
        j_re <= fo_re;
        j_im <= fo_im;
        w_v  <= j_v & rst_n;
        wr   <= j[7] ? -twr : twr;
        wi   <= j[7] ? -twi : twi;
        w_b  <= j_b;
        w_re <= j_re;
        w_im <= j_im;
        a_v  <= w_v & rst_n;
        a_b  <= w_b;
        a_re <= w_re * wr - w_im * wi;
        a_im <= w_re * wi + w_im * wr;
    end

    function signed [15:0] rnd_sat16;          // rshr by 16 (half up), sat to 16 bits
        input signed [34:0] x;
        reg signed [34:0] q;
        begin
            q = (x + 35'sd32768) >>> 16;
            if (q > 35'sd32767)
                rnd_sat16 = 16'sh7fff;
            else if (q < -35'sd32768)
                rnd_sat16 = 16'sh8000;
            else
                rnd_sat16 = q[15:0];
        end
    endfunction

    // stage B: the subcarrier of bin b, the write
    wire [7:0] i = a_b ^ 8'h80;                // natural index of the bin
    wire       used = (i >= guard_lo) && (i < 8'd0 - guard_hi) && !(dc_null && i == 8'h80);
    wire [7:0] f = i - guard_lo - ((dc_null && i > 8'h80) ? 8'd1 : 8'd0);
    reg [11:0] sbase;
    reg [3:0]  osym;
    always @(posedge clk) begin
        if (!rst_n || start) begin
            y_we  <= 1'b0;
            done  <= 1'b0;
            sbase <= 12'd0;
            osym  <= 4'd0;
        end else begin
            y_we   <= a_v && used;
            y_addr <= sbase + {4'd0, f};
            y_data <= {rnd_sat16(a_re), rnd_sat16(a_im)};
            done   <= 1'b0;
            if (a_v && a_b == 8'd255) begin
                sbase <= sbase + {4'd0, nf};
                osym  <= osym + 4'd1;
                if (osym == num_sym - 4'd1)
                    done <= 1'b1;
            end
        end
    end
endmodule
