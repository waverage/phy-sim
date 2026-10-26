`timescale 1ns / 1ps
// Test top: the three CORDIC cores side by side, each with its own stream
// and a 20-bit tag that carries the input's index to the output.
module cordic_tb (
    input  wire               clk,
    input  wire               rst_n,
    // vectoring
    input  wire               v_valid,
    input  wire signed [31:0] v_x,
    input  wire signed [31:0] v_y,
    input  wire        [19:0] v_tag,
    output wire               vo_valid,
    output wire signed [33:0] vo_mag,
    output wire signed [17:0] vo_ang,
    output wire        [19:0] vo_tag,
    // rotation
    input  wire               r_valid,
    input  wire        [15:0] r_ang,
    input  wire        [19:0] r_tag,
    output wire               ro_valid,
    output wire signed [17:0] ro_cos,
    output wire signed [17:0] ro_sin,
    output wire        [19:0] ro_tag,
    // angle of a wide sum
    input  wire               a_valid,
    input  wire signed [47:0] a_re,
    input  wire signed [47:0] a_im,
    input  wire        [19:0] a_tag,
    output wire               ao_valid,
    output wire signed [17:0] ao_ang,
    output wire        [19:0] ao_tag
);
    cordic_vec #(.W(32), .TAGW(20)) cv (
        .clk(clk), .rst_n(rst_n), .in_valid(v_valid), .in_x(v_x), .in_y(v_y), .in_tag(v_tag),
        .out_valid(vo_valid), .out_mag(vo_mag), .out_ang(vo_ang), .out_tag(vo_tag));
    cordic_rot #(.TAGW(20)) cr (
        .clk(clk), .rst_n(rst_n), .in_valid(r_valid), .in_ang(r_ang), .in_tag(r_tag),
        .out_valid(ro_valid), .out_cos(ro_cos), .out_sin(ro_sin), .out_tag(ro_tag));
    angle30 #(.W(48), .TAGW(20)) a30 (
        .clk(clk), .rst_n(rst_n), .in_valid(a_valid), .in_re(a_re), .in_im(a_im), .in_tag(a_tag),
        .out_valid(ao_valid), .out_ang(ao_ang), .out_tag(ao_tag));
endmodule
