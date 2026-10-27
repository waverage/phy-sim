"""cocotb tests of rtl/rx/rx_frame against the frame vectors: every
intermediate result of the receiver (FFT output after the tracker, pilot
estimates, tracker phase, channel estimate and samples at the data REs,
deinterleaved LLRs) and the decoded bits must match exactly, one frame at
a time; then frames back to back through the three stages, where the
decoded bits and iterations must match and the clocks per frame are
measured."""
import os

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

from tbutil import FRAME_FILES, SYNC_FILES, load, s, start
from harness.rtlcfg import ldpc_out_bits, rx_cfg, rx_scale_cfg

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


class Output:
    """The decoded words of rx_frame by tag, and the iterations by codeword."""

    def __init__(self, tw):
        self.tw, self.map = tw, ldpc_out_bits(tw.code)
        self.bits, self.iters, self.done = {}, {}, {}

    def sample(self, dut):
        if int(dut.out_valid.value):
            tag, cw = int(dut.out_tag.value), int(dut.out_cw.value)
            b = self.bits.setdefault(tag, np.full((self.tw.num_cw, self.tw.k), 2, np.uint8))
            a, wd, mk = int(dut.out_addr.value), int(dut.out_word.value), int(dut.out_mask.value)
            for j in range(16):
                v = self.map[a, j]
                assert ((mk >> j) & 1) == (v >= 0), f"output word {a}: mask {mk:04x}"
                if v >= 0:
                    b[cw, v] = (wd >> j) & 1
        if int(dut.cw_done.value):
            tag = int(dut.out_tag.value)
            self.iters[tag] = self.iters.get(tag, 0) + int(dut.cw_iters.value)
            self.done[tag] = self.done.get(tag, 0) + 1


STAGES = ("fe", "chest", "track", "demap")


async def run_frame(dut, samples: np.ndarray, tw) -> dict:
    """One frame through rx_frame, alone; the probes as arrays."""
    S, F = tw.S, tw.F
    n_samp = S * (tw.N + tw.cp)
    res = {"y": np.zeros((S * F, 2), np.int64), "hf": np.zeros((len(tw.psym) * F, 2), np.int64),
           "phi": np.zeros(S, np.int64), "h": [], "dy": [], "llr_tx": np.zeros(len(tw.perm), np.int64)}
    out = Output(tw)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    i = 0
    dut.in_valid.value = 1
    dut.in_re.value = int(samples[0, 0])
    dut.in_im.value = int(samples[0, 1])
    clocks = dict.fromkeys(STAGES + ("dec", "total"), 0)
    while True:
        await ReadOnly()
        clocks["total"] += 1
        if int(dut.a_run.value):
            clocks["fe"] += 1
        ph = int(dut.ph.value)
        if ph:
            clocks[STAGES[ph]] += 1
        if int(dut.d_busy.value):
            clocks["dec"] += 1
        accepted = int(dut.in_ready.value) and i < n_samp
        if int(dut.dbg_y_we.value):
            res["y"][int(dut.dbg_y_addr.value)] = pair(int(dut.dbg_y_data.value))
        if int(dut.dbg_hf_we.value):
            a = int(dut.dbg_hf_addr.value)
            res["hf"][(a >> 8) * F + (a & 255)] = pair(int(dut.dbg_hf_data.value))
        if int(dut.dbg_phi_valid.value):
            res["phi"][int(dut.dbg_phi_idx.value)] = s(dut.dbg_phi)
        if int(dut.dbg_h_valid.value):
            res["h"].append(pair(int(dut.dbg_h.value)))
            res["dy"].append(pair(int(dut.dbg_dy.value)))
        if int(dut.dbg_llr_valid.value):
            res["llr_tx"][int(dut.dbg_llr_idx.value)] = s(dut.dbg_llr)
        out.sample(dut)
        if int(dut.idle.value) and clocks["total"] > 2:
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
    (tag,) = out.bits
    assert out.done[tag] == tw.num_cw
    res["h"], res["dy"] = np.array(res["h"]), np.array(res["dy"])
    res["bits"], res["iters"], res["clocks"] = out.bits[tag], out.iters[tag], clocks
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
                      f"({int((r['bits'] != v['info'][fr]).sum())} bit errors against the source); "
                      f"clocks {r['clocks']}")


async def stream(dut, name: str, nframes: int, key_rx="rx", l_min=None) -> float:
    """nframes frames of a vector file back to back (as fast as the stages
    take them, one LLR scale for all): the bits and iterations of every
    frame against the twin; returns the clocks per frame after the first."""
    v = load(name)
    tw = twin_modem(v)
    lm = v["meta"]["l_min"] if l_min is None else l_min
    await configure(dut, rx_cfg(tw, lm))
    scale = float(v["llr_scale"][0])
    await configure(dut, rx_scale_cfg(scale))
    nv = v[key_rx].shape[0]
    want = [tw.rx(v[key_rx][f % nv], scale, lm) for f in range(nv)]
    base = int(dut.tag_next.value)
    n_samp = tw.S * (tw.N + tw.cp)
    out = Output(tw)
    sent, i, clk, starts = 0, 0, 0, []
    feeding = False
    while True:
        await ReadOnly()
        clk += 1
        out.sample(dut)
        go = not feeding and sent < nframes and int(dut.can_start.value)
        accepted = feeding and int(dut.in_ready.value)
        if sent == nframes and not feeding and int(dut.idle.value):
            await RisingEdge(dut.clk)
            break
        await RisingEdge(dut.clk)
        dut.start.value = int(go)
        if go:
            starts.append(clk)
            feeding, i = True, 0
            x = v[key_rx][sent % nv]
            dut.in_valid.value = 1
            dut.in_re.value = int(x[0, 0])
            dut.in_im.value = int(x[0, 1])
        elif accepted:
            i += 1
            if i < n_samp:
                x = v[key_rx][sent % nv]
                dut.in_re.value = int(x[i, 0])
                dut.in_im.value = int(x[i, 1])
            else:
                dut.in_valid.value = 0
                feeding = False
                sent += 1
    dut.start.value = 0
    for f in range(nframes):
        tag = (base + f) & 255
        t = want[f % nv]
        assert out.done.get(tag) == tw.num_cw, f"{name} frame {f}: {out.done.get(tag)} codewords done"
        assert np.array_equal(out.bits[tag], t["bits"]), f"{name} frame {f}: decoded bits differ"
        assert out.iters[tag] == t["iters"], f"{name} frame {f}: {out.iters[tag]} iterations, twin {t['iters']}"
    gaps = np.diff(starts)
    per = float(gaps[-3:].mean())
    dut._log.info(f"{name}: {nframes} frames back to back match; {per:.0f} clocks a frame "
                  f"(starts {gaps.tolist()} apart, first start to idle {clk - starts[0]})")
    return per


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


@cocotb.test(skip=ICARUS)
async def back_to_back(dut):
    """Frames back to back: the three stages and both decoders at once."""
    await setup(dut)
    for name in FRAME_FILES:
        await stream(dut, name, 8)
    await stream(dut, SYNC_FILES[0], 8, key_rx="frame", l_min=0)
