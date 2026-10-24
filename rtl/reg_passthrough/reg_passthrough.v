`timescale 1ns / 1ps
// M0 placeholder block: one-cycle registered pass-through with valid.
// Exists only to prove the cocotb + Verilator flow before real DSP blocks.
module reg_passthrough (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [15:0] in_data,
    output reg         out_valid,
    output reg  [15:0] out_data
);
    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= 16'h0;
        end else begin
            out_valid <= in_valid;
            out_data  <= in_data;
        end
    end
endmodule
