"""cocotb test: stream a vector through the block and diff against the model."""
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def passthrough_matches_model(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    rng = np.random.default_rng(1)
    vec = rng.integers(0, 2**16, size=200, dtype=np.int64)
    got = []
    for i in range(len(vec) + 1):
        if i < len(vec):
            dut.in_valid.value = 1
            dut.in_data.value = int(vec[i])
        else:
            dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        if dut.out_valid.value == 1:
            got.append(int(dut.out_data.value))
    assert np.array_equal(np.array(got), vec), f"mismatch, got {len(got)} samples"
