"""Golden vectors for the RTL (M7, M8): vectors/*.npz from the fixed-point
modem, checked against the NumPy twin before they are written.

Each frame file is self-contained: numerology, MCS and codeword plan, the
fixed-point widths, the estimator taps as designed (the RTL quantises
them as FxModem.setFilter does), the timing phase l_min and the per-frame
LLR scale, then the vectors: information and coded bits, the transmitted
samples, the received ADC samples, and the receiver's intermediate results
(FFT output after the tracker, tracker phase, channel estimate at the data
REs, deinterleaved LLRs, decoded bits). blocks.npz holds FFT, CORDIC and
LDPC decoder vectors on their own.
tests/test_m4_vectors.py replays all of them through the Zig modem and the
twin. Run:  .venv/bin/python -I tools/make_vectors.py
"""
import dataclasses
import functools
import json
import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import fxmodem as fm                            # noqa: E402
from harness import impairments as imp                      # noqa: E402
from harness import modem as zm                              # noqa: E402
from harness import twin                                     # noqa: E402
from harness.design import EstimatorDesign                  # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ           # noqa: E402

VECTORS = ROOT / "vectors"
MCS = {m.name: m for m in MCS_TABLE}
P3 = dataclasses.replace(NUM_10MHZ, name="n10-p3", pilot_ofdm_symbols=(1, 6, 11))
TDL = dict(channel="tdl", tdl_model="D", delay_spread_s=100e-9, doppler_hz=600.0, carrier_hz=5.8e9)
FRAMES = {
    # file: (numerology, OfdmLink kwargs, design, MCS, Es/N0)
    "frame_awgn_qpsk12": (NUM_10MHZ, dict(channel="awgn"), EstimatorDesign(), "qpsk-1/2", 2.0),
    "frame_tdl600_16qam12": (NUM_10MHZ, TDL, EstimatorDesign(), "16qam-1/2", 11.0),
    "frame_tdl600_cfo200_track_64qam34": (
        NUM_10MHZ, dict(TDL, phase_track="linear", channel_impairments=functools.partial(imp.cfo, offset_hz=200.0)),
        EstimatorDesign(), "64qam-3/4", 22.0),
    "frame_p3_tdl2000_16qam12": (P3, dict(TDL, doppler_hz=2000.0), EstimatorDesign(doppler_hz=1500.0, track_gain=0.0),
                                 "16qam-1/2", 12.0),
}
FRAMES_PER_FILE = 2


def twin_of(meta: dict, mask: np.ndarray) -> twin.TwinModem:
    from harness.waveform import Numerology
    num = Numerology(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta["numerology"].items()})
    fp = meta["fixed_point"]
    return twin.TwinModem(num, meta["bits_per_symbol"], k=meta["k"], n=meta["n"],
                          num_codewords=meta["num_codewords"], pilot_mask=mask, tx_scale=fp["tx_scale"],
                          dac_bits=fp["dac_bits"], llr_bits=fp["llr_bits"], app_bits=fp["app_bits"],
                          msg_bits=fp["msg_bits"], offset=fp["offset_lsb"], num_iter=fp["num_iter"],
                          phase_track=meta["phase_track"], track_gain_q15=fm.track_gain_q15(meta["design"]["track_gain"]))


def make_frame(name: str, seed: int = 1) -> dict:
    num, kw, design, mcs_name, esno = FRAMES[name]
    link = fm.FxOfdmLink(num, MCS[mcs_name], design=design, device="cpu", genie=False, **kw)
    p, fp = link.plan, link.fp
    torch.manual_seed(seed)
    no = 10 ** (-esno / 10)
    link.set_noise(no)
    b = link.ref.source([FRAMES_PER_FILE, p.num_codewords, p.k])
    info = b.to(torch.uint8).cpu().numpy()
    x = link.fixed.tx(info)
    y, _ = link.ref.propagate(link.transmit(info), torch.tensor(no))
    _, yq, g = fm.agc_adc(y.reshape(FRAMES_PER_FILE, -1).cpu().numpy(), fp)
    scale = link.llr_scale(no, g)
    bits, llr, pr = link.fixed.rx(yq, scale, link.ref.l_min, return_llr=True, probe=True)
    mask = link.ref.rg.pilot_pattern.mask[0, 0].cpu().numpy().astype(np.uint8)
    fields = {f.name: getattr(num, f.name) for f in dataclasses.fields(num)}
    meta = {"numerology": fields, "mcs": mcs_name, "bits_per_symbol": link.mcs.bits_per_symbol, "k": p.k, "n": p.n,
            "num_codewords": p.num_codewords, "fixed_point": fp.as_dict(), "design": design.as_dict(),
            "design_snr_db": design.design_snr_db(esno), "esno_db": esno, "l_min": link.ref.l_min,
            "phase_track": link.ref.phase_track == "linear", "taps": design.taps, "channel": repr(kw)}
    v = {"meta": np.array(json.dumps(meta)), "pilot_mask": mask, **{f"filt_{k}": a for k, a in link.fixed.filt.items()},
         "info": info, "coded": np.stack([zm.ldpc_encode(info[i], p.n) for i in range(FRAMES_PER_FILE)]),
         "tx": x, "rx": yq, "llr_scale": scale, "y": pr["y"].astype(np.int16), "phi": pr["phi"].astype(np.int32),
         "h": pr["h"].astype(np.int16), "llr": llr.astype(np.int8), "bits": bits}
    # the twin must agree before anything is written
    tw = twin_of(meta, mask)
    tw.set_filter({k: v[f"filt_{k}"] for k in ("bt", "start", "at", "r")})
    for i in range(FRAMES_PER_FILE):
        assert np.array_equal(tw.tx(v["coded"][i]), x[i])
        t = tw.rx(yq[i], float(scale[i]), link.ref.l_min)
        for key in ("y", "phi", "h", "llr", "bits"):
            assert np.array_equal(t[key], v[key][i]), (name, key)
    return v


def make_blocks() -> dict:
    rng = np.random.default_rng(5)
    x = rng.integers(-2048, 2048, (4, 256, 2))
    xy = rng.integers(-2 ** 30, 2 ** 30, (64, 2))
    ang = rng.integers(-65536, 65536, 64)
    k, n = 4776, 9552
    u = rng.integers(0, 2, (4, k)).astype(np.uint8)
    no = 10 ** (1.5 / 10)
    s = 1.0 - 2.0 * zm.ldpc_encode(u, n) + rng.standard_normal((4, n)) * math.sqrt(no / 2)
    llr = np.clip(np.floor(-4 * s / (2 * no) / 0.25 + 0.5), -31, 31)
    bits, iters = fm.ldpc_decode(llr, k, "oms-fixed", offset=2)
    v = {"fft_in": x.astype(np.int16), "fft_out": fm.fft(x).astype(np.int16),
         "ifft_out": fm.fft(x, inverse=True).astype(np.int16),
         "cordic_xy": xy, "cordic_vector": fm.cordic_vector(xy), "cossin_ang": ang, "cossin": fm.cossin(ang),
         "ldpc_k": np.array(k), "ldpc_n": np.array(n), "ldpc_llr": llr.astype(np.int8), "ldpc_bits": bits,
         "ldpc_iters": iters, "ldpc_info": u}
    assert np.array_equal(twin.fft(x), v["fft_out"]) and np.array_equal(twin.fft(x, True), v["ifft_out"])
    code = twin.ldpc_code(k, n)
    for i in range(4):
        b, it = twin.ldpc_decode(code, llr[i])
        assert np.array_equal(b, bits[i]) and it == iters[i]
    return v


def main():
    VECTORS.mkdir(exist_ok=True)
    np.savez_compressed(VECTORS / "blocks.npz", **make_blocks())
    print("blocks.npz")
    for name in FRAMES:
        np.savez_compressed(VECTORS / f"{name}.npz", **make_frame(name))
        print(f"{name}.npz")


if __name__ == "__main__":
    main()
