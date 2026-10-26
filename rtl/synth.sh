#!/bin/sh
# Resource estimate of the RTL blocks for the Arty A7's XC7A35T (Yosys
# synth_xilinx, no timing): rtl/synth.sh [block ...]. Prints LUTs, flip-flops,
# DSP48E1s and block RAMs per block. Yosys is an estimate, not Vivado.
set -e
here=$(cd "$(dirname "$0")" && pwd)
blocks=${*:-"fft256 cordic_vec cordic_rot stf_detect nco_rot"}
for b in $blocks; do
  case $b in
    fft256)     src="fft256/fft256.v fft256/fft256_twiddle.v" ;;
    cordic_vec|cordic_rot) src="cordic/cordic.v" ;;
    stf_detect) src="stf_corr/stf_corr.v cordic/cordic.v" ;;
    nco_rot)    src="nco_rot/nco_rot.v cordic/cordic.v" ;;
    *) echo "unknown block $b"; exit 1 ;;
  esac
  st=$(mktemp)
  (cd "$here" && yosys -q -p "read_verilog $src; synth_xilinx -family xc7 -flatten -top $b; tee -q -o $st stat" >/dev/null 2>&1)
  awk -v b="$b" '$1 ~ /^(LUT[1-6]|INV)$/ {lut += $2} $1 ~ /^FD[RSCP]E$/ {ff += $2} $1 ~ /^SRL/ {srl += $2}
                 $1 == "DSP48E1" {dsp += $2} $1 ~ /^RAMB/ {bram += $2} $1 ~ /^RAM[0-9]+X/ {lutram += $2}
                 END {printf "%-11s LUT %6d  FF %6d  SRL %4d  LUTRAM %5d  DSP48 %3d  BRAM %3d\n", b, lut, ff, srl, lutram, dsp, bram}' "$st"
  rm -f "$st"
done
