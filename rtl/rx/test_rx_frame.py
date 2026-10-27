"""cocotb tests of rtl/rx/rx_frame against the frame vectors: every
intermediate result of the receiver (FFT output after the tracker, pilot
estimates, tracker phase, channel estimate and samples at the data REs,
deinterleaved LLRs) and the decoded bits must match exactly."""
import os

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

from tbutil import FRAME_FILES, SYNC_FILES, load, s, start
from harness.rtlcfg import rx_cfg, rx_scale_cfg

ICARUS = os.environ.get("SIM", "verilator") == "icarus"


def twin_modem(v: dict):
    import tools.make_vectors as mv
    tw = mv.twin_of(v["meta"], v["pilot_mask"])
    tw.set_filter({k: v[f"filt_{k}"] for k in ("bt", "start", "at", "r")})
    return tw


async def configure(dut, cfg):
    for a, d in cfg:
        dut.cfg_we.value = 1
        dut.cfg_addr.value = a
        dut.cfg_data.value = d
        await RisingEdge(dut.clk)
    dut.cfg_we.value = 0
    await RisingEdge(dut.clk)


def pair(x: int) -> tuple[int, int]:
    re, im = (x >> 16) & 0xFFFF, x & 0xFFFF
    return re - 65536 * (re >> 15), im - 65536 * (im >> 15)


async def run_frame(dut, samples: np.ndarray, tw) -> dict:
    """One frame through rx_frame; the probes as arrays."""
    S, F = tw.S, tw.F
    n_samp = S * (tw.N + tw.cp)
    res = {"y": np.zeros((S * F, 2), np.int64), "hf": np.zeros((len(tw.psym) * F, 2), np.int64),
           "phi": np.zeros(S, np.int64), "h": [], "dy": [], "llr_tx": np.zeros(len(tw.perm), np.int64),
           "bits": np.full((tw.num_cw, tw.k), 2, np.uint8)}
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    i = 0
    dut.in_valid.value = 1
    dut.in_re.value = int(samples[0, 0])
    dut.in_im.value = int(samples[0, 1])
    while True:
        await ReadOnly()
        accepted = int(dut.in_ready.value) and i < n_samp
        if int(dut.dbg_y_we.value):
            res["y"][int(dut.dbg_y_addr.value)] = pair(int(dut.dbg_y_data.value))
        if int(dut.dbg_hf_we.value):
            res["hf"][int(dut.dbg_hf_addr.value)] = pair(int(dut.dbg_hf_data.value))
        if int(dut.dbg_phi_valid.value):
            res["phi"][int(dut.dbg_phi_idx.value)] = s(dut.dbg_phi)
        if int(dut.dbg_h_valid.value):
            res["h"].append(pair(int(dut.dbg_h.value)))
            res["dy"].append(pair(int(dut.dbg_dy.value)))
        if int(dut.dbg_llr_valid.value):
            res["llr_tx"][int(dut.dbg_llr_idx.value)] = s(dut.dbg_llr)
        if int(dut.out_valid.value):
            res["bits"][int(dut.out_cw.value), int(dut.out_idx.value)] = int(dut.out_bit.value)
        if int(dut.done.value):
            res["iters"] = int(dut.iters.value)
            await RisingEdge(dut.clk)
            break
        await RisingEdge(dut.clk)
        if accepted:
            i += 1
            if i < n_samp:
                dut.in_re.value = int(samples[i, 0])
                dut.in_im.value = int(samples[i, 1])
            else:
                dut.in_valid.value = 0
    res["h"], res["dy"] = np.array(res["h"]), np.array(res["dy"])
    return res


async def setup(dut):
    dut.cfg_we.value = 0
    dut.start.value = 0
    dut.in_valid.value = 0
    dut.in_re.value = 0
    dut.in_im.value = 0
    await start(dut)


async def check_file(dut, name: str, frames, key_rx="rx", l_min=None):
    v = load(name)
    tw = twin_modem(v)
    lm = v["meta"]["l_min"] if l_min is None else l_min
    await configure(dut, rx_cfg(tw, lm))
    for fr in frames:
        scale = float(v["llr_scale"][fr])
        t = tw.rx(v[key_rx][fr], scale, lm)
        await configure(dut, rx_scale_cfg(scale))
        r = await run_frame(dut, v[key_rx][fr], tw)
        tag = f"{name}[{fr}]"
        assert np.array_equal(r["y"], t["y"]), f"{tag}: frame memory (y) differs at {np.argwhere(r['y'] != t['y'])[:4]}"
        assert np.array_equal(r["hf"], t["hf"]), f"{tag}: pilot estimates differ"
        assert np.array_equal(r["phi"], t["phi"]), f"{tag}: phase {r['phi']} vs {t['phi']}"
        assert np.array_equal(r["h"], t["h"]), f"{tag}: channel estimate differs"
        data_y = t["y"].reshape(tw.S, tw.F, 2)[tw.dsym].reshape(-1, 2)
        assert np.array_equal(r["dy"], data_y), f"{tag}: data samples differ"
        llr = np.empty_like(r["llr_tx"])
        llr[tw.perm] = r["llr_tx"]
        llr = llr[:tw.num_cw * tw.n].reshape(tw.num_cw, tw.n)
        assert np.array_equal(llr, t["llr"]), f"{tag}: {int((llr != t['llr']).sum())} LLRs differ"
        assert np.array_equal(r["bits"], t["bits"]), f"{tag}: decoded bits differ"
        assert r["iters"] == t["iters"], f"{tag}: {r['iters']} iterations, twin {t['iters']}"
        if "bits" in v and key_rx == "rx":
            assert np.array_equal(r["bits"], v["bits"][fr])
        dut._log.info(f"{tag}: y, hf, phi, h, LLRs, bits and {r['iters']} iterations match "
                      f"({int((r['bits'] != v['info'][fr]).sum())} bit errors against the source)")


@cocotb.test()
async def frame_vectors(dut):
    """Both frames of every frame vector file, the receiver end to end."""
    await setup(dut)
    for name in FRAME_FILES[:1] if ICARUS else FRAME_FILES:
        await check_file(dut, name, range(1) if ICARUS else range(2))


@cocotb.test(skip=ICARUS)
async def sync_frames(dut):
    """The synchroniser's aligned frames (M5 vectors, tracker on)."""
    await setup(dut)
    for name in SYNC_FILES:
        await check_file(dut, name, range(2), key_rx="frame", l_min=0)
