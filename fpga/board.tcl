# Run a loopback job on the Arty Z7-20 over JTAG (xsdb): configure the
# FPGA, initialise the PS from the design's ps7_init.tcl, load the job and
# the ARM program, wait for it, read the results back.
#   xsdb fpga/board.tcl <sys.bit> <ps7_init.tcl> <loopback.elf> <job.bin> <results.bin> <points>
lassign $argv bit ps7init elf job res npoints
set JOB 0x10000000
set RES 0x0F000000
connect
targets -set -nocase -filter {name =~ "APU*"}
rst -system
after 2000
targets -set -nocase -filter {name =~ "xc7z020*"}
fpga -file $bit
targets -set -nocase -filter {name =~ "APU*"}
source $ps7init
ps7_init
ps7_post_config
targets -set -nocase -filter {name =~ "*A9*#0"}
rst -processor
dow -data $job $JOB
dow $elf
set t0 [clock seconds]
con
while {1} {
    after 5000
    stop
    set status [mrd -value [expr {$JOB + 8}]]
    set point [mrd -value [expr {$JOB + 12}]]
    puts "[expr {[clock seconds] - $t0}] s: status $status, point $point of $npoints"
    if {$status == 2 || $status == 0xBAD} break
    con
}
mrd -bin -file $res $RES [expr {16 * $npoints}]
puts "done in [expr {[clock seconds] - $t0}] s"
