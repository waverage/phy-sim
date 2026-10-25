"""M5: the preamble and the fixed-point synchroniser (modem/src/fxsync.zig)
against its NumPy twin (harness/twin.py, TwinSync) and the floating-point
reference (harness/sync.py), and the committed sync golden vectors."""
import json
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import modem as zm                              # noqa: E402
from harness import sync as S                                # noqa: E402
from harness import twin                                     # noqa: E402
from harness.fxmodem import FixedPoint, FxModem             # noqa: E402
from harness.synclink import SyncLink                       # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ, Numerology   # noqa: E402
from tools.make_vectors import SYNC_FRAMES, twin_of          # noqa: E402

if not zm.LIB.exists():
    pytest.skip("build the modem first: make modem", allow_module_level=True)

NUM = NUM_10MHZ
QPSK = MCS_TABLE[0]
TDLC = dict(channel="tdl", tdl_model="C", delay_spread_s=300e-9, doppler_hz=600.0)


@pytest.fixture(scope="module")
def fxs():
    return S.FxSync(NUM, num_threads=2)


def buffers(kw, esno_db, batch=6, seed=4):
    link = SyncLink(NUM, QPSK, device="cpu", receivers={"fixed": {"fixed": True}}, syncs=("sync",), **kw)
    np.random.seed(seed)
    torch.manual_seed(seed)
    info = np.random.randint(0, 2, (batch, link.plan.num_codewords, link.plan.k)).astype(np.uint8)
    no = 10 ** (-esno_db / 10)
    r, start, cfo = link.propagate(info, no)
    ys, yq, g = link.agc_adc(r, start)
    return link, info, ys, yq, g, start, cfo, no


def test_stf_seed_is_the_lowest_papr_pick():
    seed, papr = S.stf_seed_search(NUM)
    assert seed == S.STF_SEED and papr < 4.5


def test_preamble_zig_twin_float(fxs):
    pz = fxs.preamble()
    assert np.array_equal(pz, twin.TwinSync(fxs.cfg).preamble())
    pf = S.preamble(NUM) * FixedPoint().tx_scale
    assert np.abs(pz[:, 0] + 1j * pz[:, 1] - pf).max() < 2.0
    assert np.array_equal(pz[:32], pz[256:])                       # periodic: the last period repeats the first


@pytest.mark.parametrize("kw,esno", [(dict(channel="awgn"), 1.5), (TDLC, 9.0), (TDLC, 30.0)])
def test_sync_bit_exact_against_twin(fxs, kw, esno):
    link, _, _, yq, _, _, _, _ = buffers(kw, esno, batch=4)
    o = fxs.raw(yq)
    ot = twin.TwinSync(fxs.cfg).run(yq)
    for f in S.SYNC_OUT_FIELDS:
        assert np.array_equal(o[f], ot[f]), f
    y = fxs.align(yq, o["start"], o["inc"], link.frame_len)
    tw = twin.TwinSync(fxs.cfg)
    for i in range(len(yq)):
        assert np.array_equal(tw.align(yq[i], int(o["start"][i]), int(o["inc"][i]), link.frame_len), y[i])


def test_sync_finds_frames_at_full_cfo(fxs):
    """High SNR, CFO anywhere in +-116 kHz: every frame found, timing within
    a sample of the first path, CFO within the tracker's range by far."""
    link, _, _, yq, _, start, cfo, _ = buffers(dict(channel="awgn"), 25.0, batch=16)
    r = fxs(yq)
    assert r.detected.all()
    assert np.abs(r.start - start).max() <= 1
    assert np.abs((r.cfo - cfo) * NUM.bandwidth_hz).max() < 150
    assert np.abs(cfo * NUM.bandwidth_hz).max() > 50e3              # the draw did reach far offsets


def test_fixed_matches_float_sync(fxs):
    link, _, ys, yq, _, _, _, _ = buffers(TDLC, 12.0, batch=8)
    rq, rf = fxs(yq), link.sync(ys)
    assert np.array_equal(rq.detected, rf.detected)
    assert np.array_equal(rq.start, rf.start)
    assert np.abs((rq.cfo - rf.cfo) * NUM.bandwidth_hz).max() < 30


def test_no_false_alarm_on_noise(fxs):
    rng = np.random.default_rng(9)
    hi = 2 ** 11
    sigma = np.sqrt(FixedPoint().agc_power / 2)
    yq = np.clip(np.floor(rng.standard_normal((40, 6000, 2)) * sigma + 0.5), -hi, hi - 1).astype(np.int16)
    o = fxs.raw(yq)
    assert (o["detected"] & 1).sum() == 0


def test_sync_then_receiver_decodes():
    link, info, _, yq, g, _, _, no = buffers(dict(channel="awgn"), 12.0, batch=8)
    link.set_noise(no)
    o = link.fxsync.raw(yq)
    y = link.fxsync.align(yq, o["start"], o["inc"], link.frame_len)
    bits = link.rx["fixed"].rx(y, link.llr_scale(no, g), 0)
    assert np.array_equal(bits, info)


def _load(name):
    path = ROOT / "vectors" / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"{path} missing: run tools/make_vectors.py")
    return dict(np.load(path))


@pytest.mark.parametrize("name", list(SYNC_FRAMES))
def test_sync_vectors(name):
    v = _load(name)
    meta = json.loads(str(v["meta"]))
    num = Numerology(**{k: tuple(x) if isinstance(x, list) else x for k, x in meta["numerology"].items()})
    cfg = S.SyncConfig(**meta["sync_config"])
    sy = S.FxSync(num, cfg=cfg)
    assert np.array_equal(sy.preamble(), v["preamble"])
    o = sy.raw(v["rx"])
    for f in S.SYNC_OUT_FIELDS:
        assert np.array_equal(o[f], v[f"sync_{f}"]), f
    frame = sy.align(v["rx"], o["start"], o["inc"], v["frame"].shape[1])
    assert np.array_equal(frame, v["frame"])
    mcs = {m.name: m for m in MCS_TABLE}[meta["mcs"]]
    m = FxModem(num, mcs, k=meta["k"], n=meta["n"], num_codewords=meta["num_codewords"], pilot_mask=v["pilot_mask"],
                fixed=True, fp=FixedPoint(**meta["fixed_point"]), taps=meta["taps"], phase_track=True,
                track_gain=meta["design"]["track_gain"], num_threads=2)
    m.set_filter({k: v[f"filt_{k}"] for k in ("bt", "start", "at", "r")})
    bits, llr = m.rx(frame, v["llr_scale"], 0, return_llr=True)
    assert np.array_equal(llr, v["llr"]) and np.array_equal(bits, v["bits"])
    assert np.array_equal(bits, v["info"])                           # the vectors decode error-free
    ts = twin.TwinSync(cfg)
    pr = {}
    o0 = ts.run_one(v["rx"][0], pr)
    assert all(o0[f] == v[f"sync_{f}"][0] for f in S.SYNC_OUT_FIELDS)
    assert np.array_equal(pr["p"], v["trace_p"][0]) and np.array_equal(pr["e"], v["trace_e"][0])
    assert np.array_equal(pr["pilot_fft"], v["pilot_fft"][0]) and np.array_equal(pr["cir"], v["cir"][0])
    tm = twin_of(meta, v["pilot_mask"])
    tm.set_filter({k: v[f"filt_{k}"] for k in ("bt", "start", "at", "r")})
    assert np.array_equal(tm.rx(v["frame"][0], float(v["llr_scale"][0]), 0)["bits"], v["bits"][0])
