"""cocotb test of the board design without the Zynq (rtl/loopback behind
rtl/loopback_axi): the ARM's AXI4-Lite writes, a run of noisy frames, and
the error counters, which must equal harness/loopback.EmuLink's exactly."""
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

import tbutil  # noqa: F401  (puts the repository on sys.path)
from harness.loopback import (LB_COUNTERS, LB_CTRL, LB_NFRAMES, LB_STATUS, REAL_TIME_PACE, EmuLink, board_words,
                              frame_seconds)
from harness.waveform import MCS_TABLE, NUM_10MHZ

MCS = {m.name: m for m in MCS_TABLE}
ICARUS = os.environ.get("SIM", "verilator") == "icarus"


async def axi_write(dut, word_addr: int, data: int):
    dut.s_axi_awaddr.value = word_addr << 2
    dut.s_axi_wdata.value = data
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wvalid.value = 1
    while True:
        await RisingEdge(dut.s_axi_aclk)
        await ReadOnly()
        if int(dut.s_axi_awready.value):
            break
    await FallingEdge(dut.s_axi_aclk)
    dut.s_axi_awvalid.value = 0
    dut.s_axi_wvalid.value = 0
    while not int(dut.s_axi_bvalid.value):
        await FallingEdge(dut.s_axi_aclk)
    await RisingEdge(dut.s_axi_aclk)          # bready is high: the response is taken
    await FallingEdge(dut.s_axi_aclk)


async def axi_read(dut, word_addr: int) -> int:
    dut.s_axi_araddr.value = word_addr << 2
    dut.s_axi_arvalid.value = 1
    while True:
        await RisingEdge(dut.s_axi_aclk)
        await ReadOnly()
        if int(dut.s_axi_arready.value):
            break
    await FallingEdge(dut.s_axi_aclk)
    dut.s_axi_arvalid.value = 0
    while not int(dut.s_axi_rvalid.value):
        await FallingEdge(dut.s_axi_aclk)
    v = int(dut.s_axi_rdata.value)
    await RisingEdge(dut.s_axi_aclk)
    await FallingEdge(dut.s_axi_aclk)
    return v


async def run_point(dut, emu: EmuLink, esno: float, nframes: int, pace: int = 0) -> dict:
    w = board_words(emu, esno, pace)
    for a, d in w["rx"] + w["tx"] + w["ref"] + w["regs"] + [(LB_NFRAMES, nframes), (LB_CTRL, 2), (LB_CTRL, 1)]:
        await axi_write(dut, a, d)
    while await axi_read(dut, LB_STATUS) & 1:
        for _ in range(5000):
            await RisingEdge(dut.s_axi_aclk)
    return {k: await axi_read(dut, a) for k, a in LB_COUNTERS.items()}


async def setup(dut):
    for sig in ("awaddr", "awprot", "awvalid", "wdata", "wvalid", "araddr", "arprot", "arvalid"):
        getattr(dut, f"s_axi_{sig}").value = 0
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_bready.value = 1
    dut.s_axi_rready.value = 1
    cocotb.start_soon(Clock(dut.s_axi_aclk, 10, unit="ns").start())
    dut.s_axi_aresetn.value = 0
    for _ in range(3):
        await RisingEdge(dut.s_axi_aclk)
    dut.s_axi_aresetn.value = 1
    await FallingEdge(dut.s_axi_aclk)


async def check_point(dut, name, esno, nframes, pace=0, mixed=True):
    emu = EmuLink(NUM_10MHZ, MCS[name], num_frames=3)
    got = await run_point(dut, emu, esno, nframes, pace)
    want = emu.run(nframes, esno)
    clocks = got["clocks_lo"] + (got["clocks_hi"] << 32)
    rate = nframes / (clocks / 100e6)
    dut._log.info(f"{name} at {esno} dB, pace {pace}: board {got}, model {want}; {rate:.0f} frames/s at 100 MHz")
    assert got["dropped"] == 0 and got["lost"] == 0 and got["unmarked"] == 0, f"{name}: {got}"
    for k in ("frames", "frame_errors", "codeword_errors", "bit_errors", "iterations"):
        assert got[k] == want[k], f"{name}: {k} {got[k]} != {want[k]}"
    if mixed:
        assert 0 < want["frame_errors"] < nframes, "the point should have good and bad frames"
    return rate, got


@cocotb.test()
async def counters_match_the_model(dut):
    """Two MCS on their waterfall (some frames decode, some fail), frames
    cycling through three TX frames, as fast as the receiver goes."""
    await setup(dut)
    assert await axi_read(dut, 63) == 0x4E505859          # the ID word
    await axi_write(dut, 3, 0x12345678)
    assert await axi_read(dut, 3) == 0x12345678
    points = (("qpsk-1/2", 1.0, 3),) if ICARUS else (("qpsk-1/2", 1.0, 7), ("64qam-3/4", 15.75, 7))
    for name, esno, nframes in points:
        await check_point(dut, name, esno, nframes)


@cocotb.test(skip=ICARUS)
async def real_time(dut):
    """The emulator at 10 MS/s like a radio: no frame dropped, no sample
    lost, the air's frame rate, and the counters still the model's."""
    await setup(dut)
    air_clocks = frame_seconds(NUM_10MHZ) * 100e6
    nframes = 8
    # 64qam-2/3 at 13.5 dB: every codeword runs 20 iterations, two a frame
    for name, esno in (("64qam-3/4", 15.75), ("16qam-1/2", 6.0), ("64qam-2/3", 13.5)):
        _, got = await check_point(dut, name, esno, nframes, pace=REAL_TIME_PACE, mixed=False)
        # the frames' air time, then the last frame through the stages (a 20-iteration
        # codeword alone is 1.7 frame times)
        clocks = got["clocks_lo"] + (got["clocks_hi"] << 32)
        assert clocks < (nframes + 3) * air_clocks, f"{name}: {clocks} clocks for {nframes} frames"
        assert got["fifo_high"] < 512, f"{name}: the FIFO filled to {got['fifo_high']}"
