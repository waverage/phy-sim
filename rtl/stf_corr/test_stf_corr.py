"""cocotb: the preamble correlator and the coarse stage (stf_detect)
against the sync golden vectors and TwinSync.

  sync_*.npz   the ADC buffer -> trace_p, trace_e (every d), the detection,
               the peak, P and E at the peak, the coarse angle and NCO
               increment;
  twin         buffers of noise only (no detection), a preamble cut off by
               the end of the buffer, and a full-scale preamble.
"""
import types

import numpy as np
import cocotb
from cocotb.triggers import RisingEdge

from tbutil import SYNC_FILES, load, s, start
from harness import twin


async def run(dut, r: np.ndarray, thr: int) -> dict:
    """One buffer [L, 2] through the block, from reset."""
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_last.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    dut.threshold_q8.value = thr
    trace_p, trace_e = {}, {}
    out = {}

    def sample():
        if dut.trace_valid.value:
            d = int(dut.trace_d.value)
            trace_p[d] = (s(dut.trace_pr), s(dut.trace_pi))
            trace_e[d] = int(dut.trace_e.value)
        if dut.done.value:
            out.update(detected=int(dut.detected.value), peak=int(dut.peak.value), p_re=s(dut.peak_pr),
                       p_im=s(dut.peak_pi), e=int(dut.peak_e.value))
        if dut.ang_valid.value:
            out.update(ang_coarse=s(dut.ang_coarse), inc_coarse=s(dut.inc_coarse))

    # one sample every other clock at most (stf_corr's decision takes two;
    # rtl/sync feeds it at that rate)
    for i in range(len(r)):
        dut.in_valid.value = 1
        dut.in_re.value = int(r[i, 0])
        dut.in_im.value = int(r[i, 1])
        dut.in_last.value = int(i == len(r) - 1)
        await RisingEdge(dut.clk)
        sample()
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        sample()
    dut.in_valid.value = 0
    dut.in_last.value = 0
    for _ in range(30):
        await RisingEdge(dut.clk)
        sample()
    n = max(trace_p) + 1 if trace_p else 0
    assert sorted(trace_p) == list(range(n))
    out["trace_p"] = np.array([trace_p[d] for d in range(n)])
    out["trace_e"] = np.array([trace_e[d] for d in range(n)])
    return out


async def init(dut):
    dut.in_valid.value = 0
    dut.in_last.value = 0
    dut.in_re.value = 0
    dut.in_im.value = 0
    dut.threshold_q8.value = 0
    await start(dut)


def check(got: dict, want: dict, trace: dict | None, what: str):
    if trace is not None:
        assert np.array_equal(got["trace_p"], trace["p"]), f"{what}: trace_p"
        assert np.array_equal(got["trace_e"], trace["e"]), f"{what}: trace_e"
    assert got["detected"] == (want["detected"] & 1), f"{what}: detected {got['detected']} {want['detected']}"
    if got["detected"]:
        for k in ("peak", "p_re", "p_im", "e", "ang_coarse", "inc_coarse"):
            assert got[k] == want[k], f"{what}: {k} {got[k]} != {want[k]}"


@cocotb.test()
async def sync_vectors(dut):
    await init(dut)
    for name in SYNC_FILES:
        v = load(name)
        thr = v["meta"]["sync_config"]["threshold_q8"]
        for f in range(v["rx"].shape[0]):
            got = await run(dut, v["rx"][f], thr)
            want = {k: int(v[f"sync_{k}"][f]) for k in ("detected", "peak", "p_re", "p_im", "e", "ang_coarse",
                                                          "inc_coarse")}
            check(got, want, {"p": v["trace_p"][f], "e": v["trace_e"][f]}, f"{name} frame {f}")
            assert got["detected"] == 1
    dut._log.info("all sync vectors match")


@cocotb.test()
async def twin_cases(dut):
    await init(dut)
    v = load("sync_awgn_qpsk12")
    cfg = types.SimpleNamespace(**v["meta"]["sync_config"])
    ts = twin.TwinSync(cfg)
    rng = np.random.default_rng(3)
    pre = v["preamble"].astype(np.int64)
    cases = {
        "noise only": np.clip(np.round(rng.standard_normal((3000, 2)) * 300), -2048, 2047),
        "preamble cut by the end": np.concatenate([np.round(rng.standard_normal((900, 2)) * 30),
                                                   pre[:200]]),
        "full scale": np.clip(np.concatenate([np.zeros((100, 2)), pre * 6, pre * 6, np.zeros((300, 2))]),
                              -2048, 2047),
        "silence": np.zeros((700, 2)),
    }
    for what, r in cases.items():
        r = r.astype(np.int64)
        probe = {}
        want = ts.run_one(r, probe)
        got = await run(dut, r, cfg.threshold_q8)
        check(got, want, {"p": probe["p"], "e": probe["e"]}, what)
        assert got["detected"] == (what in ("full scale", "preamble cut by the end")), (what, got["detected"])
