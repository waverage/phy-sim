`timescale 1ns / 1ps
// rtl/loopback on an AXI4-Lite slave for the Zynq's M_AXI_GP0: 1 MB of
// byte addresses, word address = byte address / 4. One transaction at a
// time; reads take three clocks.
module loopback_axi (
    input  wire        s_axi_aclk,
    input  wire        s_axi_aresetn,
    input  wire [19:0] s_axi_awaddr,
    input  wire [2:0]  s_axi_awprot,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [19:0] s_axi_araddr,
    input  wire [2:0]  s_axi_arprot,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready
);
    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    localparam A_IDLE = 3'd0, A_WRESP = 3'd1, A_R1 = 3'd2, A_R2 = 3'd3, A_RDATA = 3'd4;
    reg  [2:0]  st;
    reg         bus_we;
    reg  [17:0] bus_addr;
    reg  [31:0] bus_wdata;
    wire [31:0] bus_rdata;

    loopback lb (.clk(s_axi_aclk), .rst_n(s_axi_aresetn), .bus_we(bus_we), .bus_addr(bus_addr),
                 .bus_wdata(bus_wdata), .bus_rdata(bus_rdata));

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            st <= A_IDLE;
            s_axi_awready <= 1'b0;
            s_axi_wready <= 1'b0;
            s_axi_bvalid <= 1'b0;
            s_axi_arready <= 1'b0;
            s_axi_rvalid <= 1'b0;
            bus_we <= 1'b0;
        end else begin
            bus_we <= 1'b0;
            s_axi_awready <= 1'b0;
            s_axi_wready <= 1'b0;
            s_axi_arready <= 1'b0;
            case (st)
                A_IDLE:
                    if (s_axi_awvalid && s_axi_wvalid) begin
                        s_axi_awready <= 1'b1;
                        s_axi_wready <= 1'b1;
                        bus_we <= 1'b1;
                        bus_addr <= s_axi_awaddr[19:2];
                        bus_wdata <= s_axi_wdata;
                        s_axi_bvalid <= 1'b1;
                        st <= A_WRESP;
                    end else if (s_axi_arvalid) begin
                        s_axi_arready <= 1'b1;
                        bus_addr <= s_axi_araddr[19:2];
                        st <= A_R1;
                    end
                A_WRESP: if (s_axi_bready) begin
                    s_axi_bvalid <= 1'b0;
                    st <= A_IDLE;
                end
                A_R1: st <= A_R2;                    // bus_rdata follows bus_addr by a clock
                A_R2: begin
                    s_axi_rdata <= bus_rdata;
                    s_axi_rvalid <= 1'b1;
                    st <= A_RDATA;
                end
                default: if (s_axi_rready) begin     // A_RDATA
                    s_axi_rvalid <= 1'b0;
                    st <= A_IDLE;
                end
            endcase
        end
    end

    // the protection, strobes and byte offsets are not used: every access is a full word
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused = &{1'b0, s_axi_awprot, s_axi_arprot, s_axi_wstrb, s_axi_awaddr[1:0], s_axi_araddr[1:0]};
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
