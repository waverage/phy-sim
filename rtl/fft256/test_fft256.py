"""cocotb: fft256 against every golden vector an FFT produced.

  blocks.npz      fft_in -> fft_out and ifft_out;
  frame_*.npz     the receiver's per-symbol FFTs: rx windows -> y (after the
                  twin's l_min rotation and the phase tracker's derotation,
                  which follow the FFT);
  sync_*.npz      the synchroniser's pilot window -> pilot_fft (the window is
                  the twin's NCO-derotated samples), and the pilot-matched
                  spectrum -> cir (the inverse FFT).
"""
import numpy as np
import cocotb
from cocotb.triggers import RisingEdge

from tbutil import FRAME_FILES, SYNC_FILES, bins, load, s, start
from harness import twin


async def fft(dut, x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """One transform of x [256, 2] through the block."""
    while not dut.in_ready.value:
        await RisingEdge(dut.clk)
    dut.inverse.value = int(inverse)
    for i in range(256):
        dut.in_valid.value = 1
        dut.in_re.value = int(x[i, 0])
        dut.in_im.value = int(x[i, 1])
        await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    out = np.zeros((256, 2), np.int64)
    got = 0
    while got < 256:
        await RisingEdge(dut.clk)
        if dut.out_valid.value:
            i = int(dut.out_idx.value)
            out[i] = (s(dut.out_re), s(dut.out_im))
            got += 1
    return out


async def init(dut):
    dut.in_valid.value = 0
    dut.inverse.value = 0
    dut.in_re.value = 0
    dut.in_im.value = 0
    await start(dut)


def diff(a, b, what):
    assert np.array_equal(a, b), f"{what}: {int((a != b).any(-1).sum())} samples differ"


@cocotb.test()
async def blocks_vectors(dut):
    await init(dut)
    v = load("blocks")
    for i in range(v["fft_in"].shape[0]):
        diff(await fft(dut, v["fft_in"][i]), v["fft_out"][i], f"fft {i}")
        diff(await fft(dut, v["fft_in"][i], True), v["ifft_out"][i], f"ifft {i}")


@cocotb.test()
async def receiver_symbols(dut):
    """Every OFDM symbol of every frame vector: rx -> FFT -> the y probe."""
    await init(dut)
    count = 0
    for name in FRAME_FILES:
        v = load(name)
        num, l_min = v["meta"]["numerology"], v["meta"]["l_min"]
        N, cp, S = num["fft_size"], num["cp_len"], num["num_ofdm_symbols"]
        b = bins(num)
        tw = twin.twiddles(N)
        for f in range(v["rx"].shape[0]):
            y = v["y"][f].reshape(S, len(b), 2)
            for sym in range(S):
                yb = await fft(dut, v["rx"][f, sym * (N + cp) + cp:][:N])
                if l_min != 0:                      # the receiver's timing rotation, after the FFT
                    j = (l_min * np.arange(N)) % N
                    w = tw[j % (N // 2)] * np.where(j < N // 2, 1, -1)[:, None]
                    re = yb[:, 0] * w[:, 0] - yb[:, 1] * w[:, 1]
                    im = yb[:, 0] * w[:, 1] + yb[:, 1] * w[:, 0]
                    yb = np.stack([twin.sat(twin.rshr(re, 16), 16), twin.sat(twin.rshr(im, 16), 16)], -1)
                yb = yb[b]
                phi = int(v["phi"][f, sym])
                if phi:                             # the phase tracker's derotation, in place on Y
                    yb = twin.rot16(yb, twin.cossin(-phi))
                diff(yb, y[sym], f"{name} frame {f} symbol {sym}")
                count += 1
    dut._log.info(f"{count} receiver symbols match")


@cocotb.test()
async def sync_pilot_and_cir(dut):
    """The synchroniser's two transforms: the pilot symbol's FFT and the
    impulse response's inverse FFT."""
    await init(dut)
    for name in SYNC_FILES:
        v = load(name)
        c, num = v["meta"]["sync_config"], v["meta"]["numerology"]
        N, cp = c["fft_size"], c["cp_len"]
        P, pre, slot = c["period"], c["period"] * c["reps"], N + cp
        ts_bins = bins(num)
        psign = twin.tables.pilot_signs(len(ts_bins), c["pilot_seed"]).astype(np.int64)
        for f in range(v["rx"].shape[0]):
            r = v["rx"][f].astype(np.int64)
            a1 = int(v["sync_ang_coarse"][f])
            inc1 = a1 * (1 << 32) // (twin.TURN * P)
            coarse = int(v["sync_peak"][f]) + pre
            n = np.clip(coarse + c["pilot_symbol"] * slot + cp + np.arange(N), 0, len(r) - 1)
            x = twin.rot16(r[n], twin.nco(n, inc1))
            y = await fft(dut, x)
            diff(y, v["pilot_fft"][f], f"{name} {f} pilot_fft")
            yb = y[ts_bins]
            sa, sb = psign[:, 0], psign[:, 1]
            z = np.zeros((N, 2), np.int64)
            z[ts_bins, 0] = twin.sat(twin.rshr(sa * yb[:, 0] + sb * yb[:, 1], 1), 16)
            z[ts_bins, 1] = twin.sat(twin.rshr(sa * yb[:, 1] - sb * yb[:, 0], 1), 16)
            diff(await fft(dut, z, True), v["cir"][f], f"{name} {f} cir")


@cocotb.test()
async def extremes(dut):
    """Full-scale inputs: saturation and the rounding of exact halves."""
    await init(dut)
    rng = np.random.default_rng(7)
    cases = [np.full((256, 2), 32767), np.full((256, 2), -32768),
             np.tile([[32767, -32768], [-32768, 32767]], (128, 1)),
             rng.integers(-32768, 32768, (256, 2)), np.eye(256, 2, dtype=np.int64) * 32767]
    for i, x in enumerate(cases):
        for inv in (False, True):
            diff(await fft(dut, x, inv), twin.fft(x, inv), f"case {i} inverse {inv}")
