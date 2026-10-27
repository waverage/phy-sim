# The board design for the Arty Z7-20: the Zynq PS (Digilent board preset,
# FCLK0 100 MHz by default) with rtl/loopback on M_AXI_GP0 at 0x43c00000 (1 MB).
# Writes the bitstream, the hardware platform (.xsa, with ps7_init.tcl) and
# the utilisation and timing reports to <out dir>.
#   vivado -mode batch -source fpga/build.tcl -tclargs <rtl dir> <out dir> [FCLK0 MHz, 100] [top, loopback_axi]
# The top is rtl/loopback (M8, the receiver behind an emulator) or
# rtl/onchip (M9, the whole radio: onchip_axi), at the same address.
# M8 closed timing at 100 MHz with 0.2 ns to spare (0.6 ns at 90.9 MHz).
lassign $argv rtl out mhz top
if {$mhz eq ""} { set mhz 100 }
if {$top eq ""} { set top loopback_axi }
source [file join [file dirname [info script]] sources.tcl]
file mkdir $out
set proj [file join $out proj]
create_project -force loopback $proj -part $PART
set_property board_part digilentinc.com:arty-z7-20:part0:1.1 [current_project]
foreach f $RTL_SOURCES { add_files -norecurse [file normalize [file join $rtl $f]] }
update_compile_order -fileset sources_1

create_bd_design sys
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells ps7]
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $mhz CONFIG.PCW_USE_M_AXI_GP0 {1}] [get_bd_cells ps7]
create_bd_cell -type module -reference $top lb
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/ps7/M_AXI_GP0" Clk_master "Auto" Clk_slave "Auto" Clk_xbar "Auto" intc_ip "New AXI SmartConnect" master_apm "0"} \
    [get_bd_intf_pins lb/s_axi]
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data]] {
    set_property offset 0x43C00000 $seg
    set_property range 1M $seg
}
validate_bd_design
set fclk [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ [get_bd_cells ps7]]
set fh [open [file join $out fclk_mhz.txt] w]; puts $fh $fclk; close $fh
save_bd_design
make_wrapper -files [get_files sys.bd] -top -import
set_property top sys_wrapper [current_fileset]
update_compile_order -fileset sources_1

# the whole radio fills 70 % of the LUTs: the explore strategies and a
# physical optimisation after routing win the last few hundred ps
if {$top eq "onchip_axi"} {
    set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
    set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
}
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1
report_utilization -file [file join $out sys_util.txt]
report_utilization -hierarchical -hierarchical_depth 4 -file [file join $out sys_util_hier.txt]
report_timing_summary -max_paths 20 -file [file join $out sys_timing.txt]
set bit [glob [file join $proj loopback.runs impl_1 *.bit]]
file copy -force $bit [file join $out sys.bit]
write_hw_platform -fixed -include_bit -force [file join $out sys.xsa]
puts "WNS [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]"
