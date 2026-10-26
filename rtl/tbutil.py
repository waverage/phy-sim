"""Shared cocotb helpers for the RTL blocks: the clock and reset, signed
reads, and the golden vectors (vectors/, written by tools/make_vectors.py)."""
import json
import pathlib
import sys

import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

ROOT = pathlib.Path(__file__).resolve().parents[1]
VECTORS = ROOT / "vectors"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def start(dut, cycles: int = 3):
    """Start a 100 MHz clock and hold reset for a few cycles."""
    import cocotb
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def s(handle) -> int:
    """A signal as a signed integer."""
    return handle.value.to_signed()


def load(name: str) -> dict:
    d = np.load(VECTORS / f"{name}.npz")
    out = {k: d[k] for k in d.files}
    if "meta" in out:
        out["meta"] = json.loads(str(out["meta"]))
    return out


FRAME_FILES = ("frame_awgn_qpsk12", "frame_tdl600_16qam12", "frame_tdl600_cfo200_track_64qam34",
               "frame_p3_tdl2000_16qam12")
SYNC_FILES = ("sync_awgn_qpsk12", "sync_tdlc300_16qam12", "sync_p3_tdl2000_qpsk12")


def bins(num: dict) -> np.ndarray:
    """FFT bin of each effective subcarrier (twin.TwinModem.bin)."""
    n = num["fft_size"]
    lo, hi = num["num_guard"]
    idx = [i for i in range(lo, n - hi) if not (num["dc_null"] and i == n // 2)]
    return np.array([(i + n // 2) % n for i in idx])
