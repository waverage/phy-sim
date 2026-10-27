"""cocotb tests of rtl/chan_emu against harness/chanemu.py: every ADC
sample must match exactly."""
import cocotb
import numpy as np
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from tbutil import load, s, start
from harness import chanemu
from harness.fxmodem import FixedPoint


async def run(dut, x: np.ndarray, seed: int, gx: int, gn: int, gap: int = 0) -> np.ndarray:
    dut.seed.value = seed
    dut.gx.value = gx
    dut.gn.value = gn
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    while True:
        await ReadOnly()
        if int(dut.ready.value):
            break
        await RisingEdge(dut.clk)
    # drive and sample on the falling edge, between the rising ones
    await FallingEdge(dut.clk)
    out = []
    i = 0
    idle = 0
    while len(out) < len(x):
        if int(dut.out_valid.value):
            out.append((s(dut.out_re), s(dut.out_im)))
        if i < len(x) and idle == 0:
            dut.in_valid.value = 1
            dut.in_re.value = int(x[i, 0])
            dut.in_im.value = int(x[i, 1])
            i += 1
            idle = gap
        else:
            dut.in_valid.value = 0
            idle = max(idle - 1, 0)
        await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    await RisingEdge(dut.clk)
    return np.array(out, np.int64)


async def setup(dut):
    dut.start.value = 0
    dut.in_valid.value = 0
    dut.in_re.value = 0
    dut.in_im.value = 0
    dut.seed.value = 0
    dut.gx.value = 0
    dut.gn.value = 0
    await start(dut)


@cocotb.test()
async def tx_frames(dut):
    """A transmitted frame of the vectors at three noise levels and seeds."""
    await setup(dut)
    fp = FixedPoint()
    v = load("frame_tdl600_16qam12")
    x = v["tx"][0].astype(np.int64)
    p = float((x.astype(float) ** 2).sum(-1).mean()) / fp.tx_scale ** 2
    for esno, seed in ((-2.0, 1), (6.0, 0xDEADBEEF), (30.0, 12345)):
        gx, gn, _ = chanemu.gains(10 ** (-esno / 10), p, fp)
        got = await run(dut, x, seed, gx, gn)
        want = chanemu.apply(x, seed, gx, gn)
        assert np.array_equal(got, want), f"Es/N0 {esno}: {int((got != want).any(-1).sum())} samples differ"
        dut._log.info(f"Es/N0 {esno} dB, seed {seed}: {len(x)} samples match")


@cocotb.test()
async def extremes(dut):
    """Full-scale samples, zero signal (noise only, deep tails), gaps
    between samples."""
    await setup(dut)
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.integers(-2048, 2048, (n, 2))
    for gx, gn, seed, gap in ((1 << 31, 1 << 20, 7, 0), (0, (1 << 24) - 1, 99, 2), (123456789, 0, 5, 1)):
        got = await run(dut, x, seed, gx, gn, gap)
        want = chanemu.apply(x, seed, gx, gn)
        assert np.array_equal(got, want), f"gx {gx} gn {gn}: {int((got != want).any(-1).sum())} samples differ"
    nr, ni = chanemu.noise(99, 200000)
    var = (nr.astype(float) ** 2 + ni.astype(float) ** 2).mean() / 2 / 2.0 ** 40
    assert abs(var - 1) < 0.01, var
