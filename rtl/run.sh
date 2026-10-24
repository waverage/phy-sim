#!/bin/sh
# Run one RTL block's cocotb test: rtl/run.sh <block> [SIM=verilator|icarus] [clean]
# Puts the project venv first on PATH, which cocotb's makefiles need and which
# macOS GNU make 3.81 cannot do from inside a makefile.
set -e
here=$(cd "$(dirname "$0")" && pwd)
block=$1; shift
export PATH="$here/../.venv/bin:$HOME/.local/bin:$PATH"
exec make -C "$here/$block" "$@"
