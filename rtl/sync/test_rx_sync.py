"""cocotb tests of rtl/sync/rx_sync against twin.TwinSync.stream: the sync
vectors' buffers (each a frame somewhere in noise, with the channel and the
oscillator offsets of M5) back to back as one stream; every synchronised
frame's fields and every aligned sample must match exactly."""
import os
import random
import types

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

from tbutil import SYNC_FILES, load, start
from harness import twin
from harness.rtlcfg import SYNC_ENABLE, sync_cfg

ICARUS = os.environ.get("SIM", "verilator") == "icarus"
REARM = 64
FIELDS = ("start", "peak", "tau", "ang_coarse", "ang_cp", "inc", "cir_peak", "cir_sum")


async def configure(dut, cfg):
    for a, d in cfg:
        dut.cfg_we.value = 1
        dut.cfg_addr.value = a
        dut.cfg_data.value = d
        await RisingEdge(dut.clk)
    dut.cfg_we.value = 0
    await RisingEdge(dut.clk)


def s32(v: int, bits: int) -> int:
    return v - (1 << bits) if v >> (bits - 1) else v


async def run_stream(dut, cfg, r: np.ndarray, pace: int, ready_p: float = 1.0, seed: int = 1, n_want: int = 0):
    """Feed r one sample every pace clocks; collect the frames and their info."""
    rng = random.Random(seed)
    flen = cfg.num_symbols * 288
    await configure(dut, sync_cfg(cfg, REARM) + [(SYNC_ENABLE, 1)])
    wn0 = int(dut.wn.value)                    # the search starts at the next sample
    infos, frames, cur = [], [], None
    i, clk, tail = 0, 0, 0
    while True:
        await ReadOnly()
        if int(dut.info_valid.value):
            infos.append({"start": int(dut.info_start.value), "peak": int(dut.info_peak.value),
                          "tau": s32(int(dut.info_tau.value), 9), "ang_coarse": s32(int(dut.info_ang_coarse.value), 18),
                          "ang_cp": s32(int(dut.info_ang_cp.value), 18), "inc": s32(int(dut.info_inc.value), 32),
                          "cir_peak": int(dut.info_cir_peak.value), "cir_sum": int(dut.info_cir_sum.value),
                          "base": int(dut.info_base.value)})
        ready = int(dut.out_ready.value)
        if int(dut.out_valid.value) and ready:
            v = int(dut.out_re.value), int(dut.out_im.value)
            v = (s32(v[0], 16), s32(v[1], 16))
            if int(dut.out_first.value):
                cur = []
                frames.append(cur)
            cur.append(v)
        # done when the stream is in and either everything is idle, or the
        # expected frames are out and nothing has come for a while (a
        # detection at the very end waits for samples that never come)
        busy = int(dut.sq.value) != 0 or int(dut.out_valid.value) or int(dut.pend.value)
        if i == len(r):
            done_frames = len(frames) >= n_want and len(frames[n_want - 1]) == flen if n_want else True
            tail = tail + 1 if (not busy or done_frames) and not int(dut.out_valid.value) else 0
            if tail > 2000:
                break
        await RisingEdge(dut.clk)
        clk += 1
        dut.out_ready.value = int(rng.random() < ready_p)
        feed = i < len(r) and clk % pace == 0
        dut.in_valid.value = int(feed)
        if feed:
            dut.in_re.value = int(r[i, 0])
            dut.in_im.value = int(r[i, 1])
            i += 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    await configure(dut, [(SYNC_ENABLE, 0)])
    return infos, [np.array(f, np.int64) for f in frames], wn0


async def setup(dut):
    dut.cfg_we.value = 0
    dut.in_valid.value = 0
    dut.in_re.value = 0
    dut.in_im.value = 0
    dut.out_ready.value = 1
    await start(dut)


async def check_file(dut, name: str, order, pace: int, ready_p: float = 1.0, **over):
    v = load(name)
    cfg = types.SimpleNamespace(**{**v["meta"]["sync_config"], **over})
    # one more buffer behind the last: the twin decides a frame once its whole span is there
    r = np.concatenate([v["rx"][k] for k in order + (order[0],)]).astype(np.int64)
    want = twin.TwinSync(cfg).stream(r, REARM)
    assert len(want) >= len(order), f"{name}: the twin found {len(want)} frames in {len(order)} buffers"
    # the first frame's base is the stream's start, as the vectors' buffer
    # (later frames count the NCO from their own base)
    if not over:
        assert np.array_equal(want[0]["frame"], v["frame"][order[0]]), f"{name}: the twin's stream differs from the vectors"
    n0, j0 = int(dut.n_sync.value), int(dut.n_reject.value)
    infos, frames, wn0 = await run_stream(dut, cfg, r, pace, ready_p, n_want=len(want))
    assert int(dut.n_overrun.value) == 0, f"{name}: buffer overrun"
    assert len(infos) >= len(want) and len(frames) >= len(want), f"{name}: {len(infos)} frames, twin {len(want)}"
    for f, (w, got, fr) in enumerate(zip(want, infos, frames)):
        for k in ("start", "peak", "base"):
            got[k] -= wn0
        for k in FIELDS + ("base",):
            assert got[k] == int(w[k]), f"{name} frame {f}: {k} {got[k]} != twin {int(w[k])}"
        if not np.array_equal(fr, w["frame"]):
            n = min(len(fr), len(w["frame"]))
            bad = np.argwhere((fr[:n] != w["frame"][:n]).any(-1)).reshape(-1)
            raise AssertionError(f"{name} frame {f}: {len(fr)} samples (twin {len(w['frame'])}), {len(bad)} differ, "
                                 f"first at {bad[:3]}: {fr[bad[:3]].tolist()} vs {w['frame'][bad[:3]].tolist()}; "
                                 f"rtl[:3] {fr[:3].tolist()} twin[:3] {w['frame'][:3].tolist()}")
    dut._log.info(f"{name}: {len(want)} frames in {len(r)} samples, pace {pace}, ready {ready_p} {over}: start, "
                  f"tau, angles, inc, CIR and every aligned sample match ({int(dut.n_sync.value) - n0} synchronised, "
                  f"{int(dut.n_reject.value) - j0} rejected)")
    return int(dut.n_reject.value) - j0


@cocotb.test()
async def streams(dut):
    """Each sync vector file's buffers back to back, frames found in order."""
    await setup(dut)
    for name in SYNC_FILES[:1] if ICARUS else SYNC_FILES:
        await check_file(dut, name, (0, 1) if ICARUS else (0, 1, 0, 1), pace=4)


@cocotb.test(skip=ICARUS)
async def false_alarms(dut):
    """A threshold low enough for noise to cross it: the CIR check rejects
    those, and the search goes on past them as the twin's does."""
    await setup(dut)
    rejected = await check_file(dut, SYNC_FILES[0], (0, 1), pace=4, threshold_q8=2)
    assert rejected > 0, "no false alarm: the rejection path went untested"


@cocotb.test(skip=ICARUS)
async def real_time_and_backpressure(dut):
    """10 clocks a sample like the radio, and a receiver that stalls."""
    await setup(dut)
    await check_file(dut, SYNC_FILES[1], (1, 0, 1), pace=10, ready_p=0.3)
