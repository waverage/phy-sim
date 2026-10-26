"""cocotb: the NCO derotator against the sync golden vectors and the twin.

  sync_*.npz   the ADC buffer, the synchroniser's start and NCO increment
               -> frame (the aligned frame the receiver decodes);
  twin         the coarse derotation of the first pilot symbol (the FFT's
               input in fft256's sync test), and full-scale 16-bit samples
               at random increments.
"""
import numpy as np
import cocotb
from cocotb.triggers import RisingEdge

from tbutil import SYNC_FILES, load, s, start
from harness import twin


async def derotate(dut, x: np.ndarray, n0: int, inc: int) -> np.ndarray:
    """Samples x [L, 2] as samples n0.. of a buffer, rotated by -n inc."""
    dut.phase0.value = (-n0 * inc) % (1 << 32)
    dut.inc.value = inc % (1 << 32)
    out = []
    for i in range(len(x)):
        dut.start.value = int(i == 0)
        dut.in_valid.value = 1
        dut.in_re.value = int(x[i, 0])
        dut.in_im.value = int(x[i, 1])
        await RisingEdge(dut.clk)
        if dut.out_valid.value:
            out.append((s(dut.out_re), s(dut.out_im)))
    dut.in_valid.value = 0
    dut.start.value = 0
    while len(out) < len(x):
        await RisingEdge(dut.clk)
        if dut.out_valid.value:
            out.append((s(dut.out_re), s(dut.out_im)))
    return np.array(out)


async def init(dut):
    dut.start.value = 0
    dut.in_valid.value = 0
    dut.in_re.value = 0
    dut.in_im.value = 0
    dut.phase0.value = 0
    dut.inc.value = 0
    await start(dut)


@cocotb.test()
async def aligned_frames(dut):
    await init(dut)
    for name in SYNC_FILES:
        v = load(name)
        c = v["meta"]["sync_config"]
        L = v["frame"].shape[1]
        for f in range(v["rx"].shape[0]):
            r = v["rx"][f].astype(np.int64)
            n = int(v["sync_start"][f]) - c["backoff"] + np.arange(L)
            valid = (n >= 0) & (n < len(r))
            x = np.where(valid[:, None], r[np.clip(n, 0, len(r) - 1)], 0)       # outside the buffer: zeros
            got = await derotate(dut, x, int(n[0]), int(v["sync_inc"][f]))
            assert np.array_equal(got, v["frame"][f]), f"{name} frame {f}: {int((got != v['frame'][f]).any(-1).sum())} differ"
    dut._log.info("all aligned frames match")


@cocotb.test()
async def pilot_windows_and_extremes(dut):
    await init(dut)
    for name in SYNC_FILES:
        v = load(name)
        c = v["meta"]["sync_config"]
        N, slot, pre = c["fft_size"], c["fft_size"] + c["cp_len"], c["period"] * c["reps"]
        for f in range(v["rx"].shape[0]):
            r = v["rx"][f].astype(np.int64)
            inc1 = int(v["sync_inc_coarse"][f])
            n = np.clip(int(v["sync_peak"][f]) + pre + c["pilot_symbol"] * slot + c["cp_len"] + np.arange(N), 0, len(r) - 1)
            got = await derotate(dut, r[n], int(n[0]), inc1)
            assert np.array_equal(got, twin.rot16(r[n], twin.nco(n, inc1))), f"{name} {f} pilot window"
    rng = np.random.default_rng(9)
    for k in range(6):
        x = rng.integers(-32768, 32768, (500, 2))
        if k == 0:
            x[:] = [-32768, -32768]
        inc = int(rng.integers(-2 ** 31, 2 ** 31))
        n0 = int(rng.integers(-10_000, 10_000))
        got = await derotate(dut, x, n0, inc)
        want = twin.rot16(x.astype(np.int64), twin.nco(n0 + np.arange(len(x)), inc))
        assert np.array_equal(got, want), f"random case {k}"
