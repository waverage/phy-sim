"""M4 gate: the committed golden vectors (vectors/*.npz, tools/make_vectors.py)
replay bit exact through the Zig modem and the NumPy twin. A change to any
fixed-point block shows up here as a vector diff; regenerate the vectors
only when the change is intended (the RTL is verified against them)."""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import fxmodem as fm                            # noqa: E402
from harness import modem as zm                              # noqa: E402
from harness import twin                                     # noqa: E402
from harness.fxmodem import FixedPoint, FxModem             # noqa: E402
from harness.waveform import MCS_TABLE, Numerology          # noqa: E402
from tools.make_vectors import FRAMES, twin_of               # noqa: E402

VECTORS = ROOT / "vectors"
if not zm.LIB.exists():
    pytest.skip("build the modem first: make modem", allow_module_level=True)


def _load(name):
    path = VECTORS / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"{path} missing: run tools/make_vectors.py")
    return dict(np.load(path))


def test_block_vectors():
    v = _load("blocks")
    assert np.array_equal(fm.fft(v["fft_in"]), v["fft_out"])
    assert np.array_equal(fm.fft(v["fft_in"], inverse=True), v["ifft_out"])
    assert np.array_equal(twin.fft(v["fft_in"]), v["fft_out"])
    assert np.array_equal(fm.cordic_vector(v["cordic_xy"]), v["cordic_vector"])
    assert np.array_equal(fm.cossin(v["cossin_ang"]), v["cossin"])
    k, n = int(v["ldpc_k"]), int(v["ldpc_n"])
    bits, iters = fm.ldpc_decode(v["ldpc_llr"].astype(np.float32), k, "oms-fixed", offset=2)
    assert np.array_equal(bits, v["ldpc_bits"]) and np.array_equal(iters, v["ldpc_iters"])
    b, it = twin.ldpc_decode(twin.ldpc_code(k, n), v["ldpc_llr"][0])
    assert np.array_equal(b, v["ldpc_bits"][0]) and it == v["ldpc_iters"][0]


@pytest.mark.parametrize("name", list(FRAMES))
def test_frame_vectors(name):
    v = _load(name)
    meta = json.loads(str(v["meta"]))
    num = Numerology(**{k: tuple(x) if isinstance(x, list) else x for k, x in meta["numerology"].items()})
    mcs = {m.name: m for m in MCS_TABLE}[meta["mcs"]]
    fp = FixedPoint(**meta["fixed_point"])
    filt = {k: v[f"filt_{k}"] for k in ("bt", "start", "at", "r")}
    m = FxModem(num, mcs, k=meta["k"], n=meta["n"], num_codewords=meta["num_codewords"], pilot_mask=v["pilot_mask"],
                fixed=True, fp=fp, taps=meta["taps"], phase_track=meta["phase_track"],
                track_gain=meta["design"]["track_gain"], num_threads=2)
    m.set_filter(filt)
    assert np.array_equal(m.tx(v["info"]), v["tx"])
    bits, llr, pr = m.rx(v["rx"], v["llr_scale"], meta["l_min"], return_llr=True, probe=True)
    assert np.array_equal(pr["y"], v["y"]) and np.array_equal(pr["phi"], v["phi"]) and np.array_equal(pr["h"], v["h"])
    assert np.array_equal(llr, v["llr"]) and np.array_equal(bits, v["bits"])
    # the twin, first frame
    tw = twin_of(meta, v["pilot_mask"])
    tw.set_filter(filt)
    assert np.array_equal(tw.tx(v["coded"][0]), v["tx"][0])
    t = tw.rx(v["rx"][0], float(v["llr_scale"][0]), meta["l_min"])
    for key in ("y", "phi", "h", "llr", "bits"):
        assert np.array_equal(t[key], v[key][0]), key
