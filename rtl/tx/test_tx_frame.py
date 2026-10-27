"""cocotb tests of rtl/tx/tx_frame: frames back to back, each the
preamble and twin.TwinModem.tx of the LDPC codewords (modem.ldpc_encode) of
its information bits, sample for sample."""
import os
import random

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

from tbutil import FRAME_FILES, load, start
from harness import modem as zm
from harness import sync, twin
from harness.loopback import ref_words
from harness.rtlcfg import ldpc_out_words, tx_cfg

ICARUS = os.environ.get("SIM", "verilator") == "icarus"


def twin_modem(v: dict):
    import tools.make_vectors as mv
    return mv.twin_of(v["meta"], v["pilot_mask"])


async def configure(dut, cfg):
    for a, d in cfg:
        dut.cfg_we.value = 1
        dut.cfg_addr.value = a
        dut.cfg_data.value = d
        await RisingEdge(dut.clk)
    dut.cfg_we.value = 0
    await RisingEdge(dut.clk)


def s12(v: int) -> int:
    return v - 4096 if v & 2048 else v


def wrong_bits(tw, samples: np.ndarray, info: np.ndarray) -> str:
    """For a failure: the transmitted frame back to coded bits (FFT, nearest
    level), and which codeword variables (column, word) came out wrong."""
    slot = tw.N + tw.cp
    got = []
    for s in tw.dsym:
        y = twin.fft(samples[s * slot + tw.cp:(s + 1) * slot])[tw.bin]
        for dim in (0, 1):
            pass
        got.append(y)
    y = np.concatenate(got)                                     # [data RE, 2]
    md = tw.B // 2
    lev = tw.tx_level
    idx = np.abs(y[:, :, None] - lev[None, None, :]).argmin(-1)  # level index per dim
    bits = np.zeros((len(y), tw.B), np.int64)
    for j in range(md):
        bits[:, 2 * j] = (idx[:, 0] >> (md - 1 - j)) & 1
        bits[:, 2 * j + 1] = (idx[:, 1] >> (md - 1 - j)) & 1
    txb = bits.reshape(-1)
    coded = np.zeros(tw.frame_bits, np.int64)
    coded[tw.perm] = txb
    want = np.concatenate([zm.ldpc_encode(info[c], tw.n) for c in range(tw.num_cw)])
    have = coded[:tw.num_cw * tw.n]
    wrong = np.argwhere(have != want).reshape(-1)
    if not len(wrong):
        return "the coded bits are all right (the mapping is wrong)"
    code = tw.code
    t = wrong % tw.n
    n_sys = code.k - 2 * code.z
    v = np.where(t < n_sys, 2 * code.z + t, code.k_ldpc + (t - n_sys))
    cols, cnt = np.unique(v // code.z, return_counts=True)
    q = code.z // 16
    r = v % code.z
    return (f"{len(wrong)} coded bits wrong, by column: {dict(zip(cols.tolist(), cnt.tolist()))}; "
            f"(column, word, lane): {list(zip((v // code.z).tolist(), (r % q).tolist(), (r // q).tolist()))[:20]}; "
            f"tx bit index {np.argwhere(np.isin(tw.perm, wrong)).reshape(-1)[:20].tolist()}")


async def check_file(dut, name: str, nframes: int, ready_p: float = 1.0, seed: int = 2):
    v = load(name)
    tw = twin_modem(v)
    pre = twin.TwinSync(sync.sync_config(tw.num)).preamble()
    await configure(dut, tx_cfg(tw, pre))
    info = v["info"]                                    # [frames, codewords, k]
    nv = info.shape[0]
    outw = ldpc_out_words(tw.code)
    words = [ref_words(tw.code, info[f]).reshape(-1) for f in range(nv)]   # c OUTW + w
    want = [np.concatenate([pre, tw.tx(np.stack([zm.ldpc_encode(info[f, c], tw.n)
                                                 for c in range(tw.num_cw)]))]).astype(np.int64)
            for f in range(nv)]
    assert all(len(w) == outw * tw.num_cw for w in words)
    flen = len(want[0])
    rng = random.Random(seed)
    frames, cur = [], None
    assert int(dut.idle.value), "the transmitter is not idle"
    f0 = int(dut.enc_frame.value)                       # the next frame's number
    dut.go.value = 1
    clk = 0
    while len(frames) < nframes or len(frames[-1]) < flen:
        await ReadOnly()
        addr, fr = int(dut.info_addr.value), int(dut.info_frame.value)
        if int(dut.out_valid.value) and int(dut.out_ready.value):
            if int(dut.out_first.value):
                cur = []
                frames.append(cur)
            cur.append((s12(int(dut.out_re.value)), s12(int(dut.out_im.value))))
            if len(frames) == nframes and len(cur) == flen - 1:
                pass
        await RisingEdge(dut.clk)
        clk += 1
        if len(frames) >= nframes - 1:
            dut.go.value = 0
        w = words[(fr - f0) % nv]
        dut.info_data.value = int(w[addr]) if addr < len(w) else 0
        dut.out_ready.value = int(rng.random() < ready_p)
        if clk >= 400000 * nframes:
            st = {k: int(getattr(dut, k).value) for k in ("est", "mst", "fst", "xst", "bank_full", "gfull", "ofull",
                                                          "ebank", "mbank", "gw", "gr", "ow", "orr", "eop", "ms", "mi")}
            raise AssertionError(f"the transmitter stalled: {len(frames)} frames, last {len(cur or [])} samples; {st}")
    for f, got in enumerate(frames[:nframes]):
        got = np.array(got, np.int64)
        w = want[f % nv]
        if not np.array_equal(got, w):
            bad = np.argwhere((got != w).any(-1)).reshape(-1)
            raise AssertionError(f"{name} frame {f}: {len(bad)} of {flen} samples differ, first at {bad[:4]}: "
                                 f"{got[bad[:3]].tolist()} vs {w[bad[:3]].tolist()}; "
                                 f"{wrong_bits(tw, got[len(pre):], info[f % nv])}")
    dut._log.info(f"{name}: {nframes} frames of {flen} samples (preamble and {tw.S} symbols) match the twin; "
                  f"{clk / nframes:.0f} clocks a frame with ready {ready_p}")
    dut.go.value = 0
    dut.out_ready.value = 1
    for _ in range(200000):                             # let the frames already encoded out
        await RisingEdge(dut.clk)
        if int(dut.idle.value):
            break
    assert int(dut.idle.value), "the transmitter did not drain"


async def setup(dut):
    dut.cfg_we.value = 0
    dut.go.value = 0
    dut.info_data.value = 0
    dut.out_ready.value = 1
    await start(dut)


@cocotb.test()
async def frames(dut):
    """Every frame vector file: frames back to back."""
    await setup(dut)
    for name in FRAME_FILES[:1] if ICARUS else FRAME_FILES:
        await check_file(dut, name, 2 if ICARUS else 3)


@cocotb.test(skip=ICARUS)
async def paced(dut):
    """The DAC's pace (ready one clock in ten) and a stalling one."""
    await setup(dut)
    await check_file(dut, FRAME_FILES[2], 3, ready_p=0.1)
