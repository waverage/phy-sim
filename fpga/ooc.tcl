# Out-of-context implementation of one RTL top for the Arty Z7-20's
# XC7Z020 at 100 MHz: utilisation and timing without the Zynq around it.
#   vivado -mode batch -source fpga/ooc.tcl -tclargs <rtl dir> <top> <out dir>
lassign $argv rtl top out
source [file join [file dirname [info script]] sources.tcl]
file mkdir $out
foreach f $RTL_SOURCES { read_verilog [file join $rtl $f] }
synth_design -top $top -part $PART -mode out_of_context -flatten_hierarchy rebuilt
set clk [lindex [get_ports -quiet {clk s_axi_aclk}] 0]
create_clock -name clk -period 10.0 $clk
report_utilization -file [file join $out ${top}_synth_util.txt]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file [file join $out ${top}_util.txt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $out ${top}_util_hier.txt]
report_timing_summary -max_paths 20 -file [file join $out ${top}_timing.txt]
report_timing -max_paths 2000 -nworst 1 -file [file join $out ${top}_paths.txt]
puts "WNS [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]"
