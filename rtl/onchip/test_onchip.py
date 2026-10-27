"""cocotb test of the whole radio on the chip without the Zynq (rtl/onchip
behind rtl/onchip_axi): the ARM's AXI4-Lite writes, a run of frames through
the transmitter, the emulated air with a carrier offset, the synchroniser
and the receiver; the counters must equal harness/onchip.OnChipLink's."""
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

import tbutil  # noqa: F401  (puts the repository on sys.path)
from harness.loopback import REAL_TIME_PACE
from harness.onchip import FAST_PACE, OC_COUNTERS, OC_CTRL, OC_NFRAMES, OC_STATUS, OnChipLink, cfo_inc_of
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
    await RisingEdge(dut.s_axi_aclk)
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


async def check_point(dut, name, esno, cfo_hz, nframes, pace=0):
    link = OnChipLink(NUM_10MHZ, MCS[name], cfo_inc=cfo_inc_of(cfo_hz), num_frames=3)
    w = link.board_words(esno, pace)
    words = w["rx"] + w["tx"] + w["sync"] + w["ref"] + w["regs"] + [(OC_NFRAMES, nframes), (OC_CTRL, 2), (OC_CTRL, 1)]
    dut._log.info(f"{name}: {len(words)} configuration words")
    for a, d in words:
        await axi_write(dut, a, d)
    polls = 0
    while await axi_read(dut, OC_STATUS) & 1:
        for _ in range(5000):
            await RisingEdge(dut.s_axi_aclk)
        polls += 1
        if polls % 40 == 0:
            oc = dut.oc
            dut._log.info(f"  {polls * 5000} clocks: source {int(oc.sst.value)} seg {int(oc.seg.value)}, "
                          f"matched {int(oc.m_r.value)}, decoded {await axi_read(dut, OC_COUNTERS['frames'])}, "
                          f"rx_feed {int(oc.rx_feed.value)}, settled {int(oc.settled.value)}; sync sq "
                          f"{int(oc.sync.sq.value)} pend {int(oc.sync.pend.value)} out {int(oc.sy_ov.value)} "
                          f"first {int(oc.sy_first.value)} o_cnt {int(oc.sync.o_cnt.value)} det_run "
                          f"{int(oc.sync.det_run.value)} n_sync {int(oc.sync.n_sync.value)}; mq {int(oc.mq_w.value)}"
                          f"/{int(oc.mq_r.value)} rx_can {int(oc.rx_can.value)} m_busy {int(oc.m_busy.value)}")
    got = {k: await axi_read(dut, a) for k, a in OC_COUNTERS.items()}
    want = link.run(nframes, esno)
    clocks = got["clocks_lo"] + (got["clocks_hi"] << 32)
    dut._log.info(f"{name} at {esno} dB, {cfo_hz / 1e3:+.0f} kHz, pace {pace}: chip {got}, model {want}; "
                  f"{nframes / (clocks / 100e6):.0f} frames/s at 100 MHz")
    assert got["sync_overruns"] == 0, f"{name}: the synchroniser fell behind"
    assert got["frames"] == want["synced"], f"{name}: {got['frames']} decoded, model {want['synced']}"
    assert got["frame_errors"] + got["missed"] == want["frame_errors"], f"{name}: frame errors"
    for k in ("missed", "spurious", "codeword_errors", "bit_errors", "iterations"):
        assert got[k] == want[k], f"{name}: {k} {got[k]} != {want[k]}"
    return got, want


@cocotb.test()
async def chain_matches_the_model(dut):
    """The whole chain at the waterfall, with carrier offsets."""
    await setup(dut)
    assert await axi_read(dut, 63) == 0x4E505859
    points = (("qpsk-1/2", 1.5, 20e3, 3),) if ICARUS else (("qpsk-1/2", 1.25, 20e3, 6), ("64qam-3/4", 15.75, -50e3, 6))
    for name, esno, cfo, n in points:
        await check_point(dut, name, esno, cfo, n, pace=FAST_PACE)


@cocotb.test(skip=ICARUS)
async def real_time(dut):
    """At the radio's 10 MS/s: nothing falls behind, and the counts hold."""
    await setup(dut)
    got, want = await check_point(dut, "16qam-1/2", 7.0, 90e3, 6, pace=REAL_TIME_PACE)
    assert got["missed"] == 0
