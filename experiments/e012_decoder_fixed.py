"""e012: which LDPC decoder and which widths for the fixed-point modem (M4).

The hardware decoder is a row-layered offset min-sum (modem/src/fxldpc.zig);
the M1-M3 curves used Sionna's flooding boxplus-phi (sum-product) with
exact LLRs. This measures what each step towards hardware costs, on AWGN
per resource element (as e001), every decoder decoding the same LLRs
(common random numbers):
- the max-log demapper instead of exact LLRs (both with flooding BP);
- layered offset min-sum in floating point, offsets 0.25 to 0.75 nat;
- the same in fixed point: channel LLR, message and a-posteriori widths,
  the LLR step (nats per LSB) and the offset in LSBs;
- 10 instead of 20 iterations (decoder throughput in hardware).

Every decoder stops early on a zero syndrome. The result per decoder is
the Es/N0 at 1 % codeword error rate, the loss against flooding BP with
exact LLRs, and the mean iterations near that point.
Run:  .venv/bin/python -I experiments/e012_decoder_fixed.py [--profile quick|reference]
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness import fxmodem as fm                            # noqa: E402
from harness import modem as zm                              # noqa: E402
from harness.sweep import profile_arg, reference_esno, snr_at_fer, write_result  # noqa: E402
from harness.twin import maxlog_llr                          # noqa: E402
from harness.waveform import MCS_TABLE, NUM_10MHZ, plan_codewords  # noqa: E402

NAME = "e012_decoder_fixed"
MCS_NAMES = ["qpsk-1/2", "16qam-1/2", "64qam-2/3", "64qam-5/6"]

# name: (demapper, algo, decoder kwargs, LLR step in nats per LSB)
DECODERS = {
    "bp exact": ("exact", "bp", {}, 1.0),
    "bp maxlog": ("maxlog", "bp", {}, 1.0),
    "oms float o0.25": ("maxlog", "oms-float", dict(offset=0.25), 1.0),
    "oms float o0.5": ("maxlog", "oms-float", dict(offset=0.5), 1.0),
    "oms float o0.75": ("maxlog", "oms-float", dict(offset=0.75), 1.0),
    "q5/5/7 d1 o1": ("maxlog", "oms-fixed", dict(llr_bits=5, msg_bits=5, app_bits=7, offset=1), 1.0),
    "q6/6/8 d0.75 o1": ("maxlog", "oms-fixed", dict(llr_bits=6, msg_bits=6, app_bits=8, offset=1), 0.75),
    "q6/6/8 d0.5 o1": ("maxlog", "oms-fixed", dict(llr_bits=6, msg_bits=6, app_bits=8, offset=1), 0.5),
    "q6/6/7 d0.5 o1": ("maxlog", "oms-fixed", dict(llr_bits=6, msg_bits=6, app_bits=7, offset=1), 0.5),
    "q6/6/8 d0.25 o2": ("maxlog", "oms-fixed", dict(llr_bits=6, msg_bits=6, app_bits=8, offset=2), 0.25),
    "q7/7/9 d0.25 o2": ("maxlog", "oms-fixed", dict(llr_bits=7, msg_bits=7, app_bits=9, offset=2), 0.25),
    "q6/6/8 d0.5 o1 it10": ("maxlog", "oms-fixed", dict(llr_bits=6, msg_bits=6, app_bits=8, offset=1,
                                                         num_iter=10), 0.5),
}
BASELINE = "bp exact"
PROFILES = {
    # SNR grid around the e001 1 % point, codewords per chunk, stop rule
    "quick": dict(lo=-0.75, hi=1.5, step=0.25, chunk=1000, min_errors=50, max_cw=5000),
    "reference": dict(lo=-0.75, hi=1.75, step=0.125, chunk=2000, min_errors=200, max_cw=40000),
}


def llrs(mcs, plan, esno_db: float, count: int, rng):
    """count codewords through QAM and AWGN: (info bits, exact LLRs, max-log LLRs)."""
    B = mcs.bits_per_symbol
    u = rng.integers(0, 2, (count, plan.k)).astype(np.uint8)
    c = zm.ldpc_encode(u, plan.n)
    x = zm.qam_map(c, B)
    no = 10 ** (-esno_db / 10)
    w = (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) * np.sqrt(no / 2)
    y = (x + w).astype(np.complex64)
    exact = zm.qam_demap(y, np.full(y.size, no, np.float32), B).reshape(count, plan.n)
    maxlog = maxlog_llr(y, no, B).astype(np.float32).reshape(count, plan.n)
    return u, exact, maxlog


def quantise(llr: np.ndarray, step: float, bits: int) -> np.ndarray:
    hi = 2 ** (bits - 1) - 1
    return np.clip(np.floor(llr / step + 0.5), -hi, hi)


def run_mcs(mcs, prof, log=print) -> dict:
    plan = plan_codewords(NUM_10MHZ, mcs)
    rng = np.random.default_rng(12)
    ref = reference_esno(mcs.name, "esno_db_at_fer1")
    grid = np.arange(ref + prof["lo"], ref + prof["hi"] + 1e-9, prof["step"])
    points = []
    for esno in grid:
        tot = {d: {"errors": 0, "iters": 0} for d in DECODERS}
        cw = 0
        while cw < prof["max_cw"] and min(t["errors"] for t in tot.values()) < prof["min_errors"]:
            u, exact, maxlog = llrs(mcs, plan, float(esno), prof["chunk"], rng)
            for name, (dem, algo, kw, step) in DECODERS.items():
                l = exact if dem == "exact" else maxlog
                if algo == "oms-fixed":
                    l = quantise(l, step, kw["llr_bits"])
                elif step != 1.0:
                    l = l / step
                b, it = fm.ldpc_decode(l, plan.k, algo, **kw)
                tot[name]["errors"] += int((b != u).any(-1).sum())
                tot[name]["iters"] += int(it.sum())
            cw += prof["chunk"]
        pt = {"esno_db": float(esno), "codewords": cw,
              **{d: {"bler": t["errors"] / cw, "errors": t["errors"], "iters": t["iters"] / cw}
                 for d, t in tot.items()}}
        points.append(pt)
        log(f"  {esno:6.3f} dB  {cw:6d} cw  " + "  ".join(f"{pt[d]['bler']:.4f}" for d in DECODERS))
        if all(pt[d]["bler"] < 1e-3 for d in DECODERS):
            break
    out = {"k": plan.k, "n": plan.n, "points": points, "decoders": {}}
    for d in DECODERS:
        pts = [{"esno_db": p["esno_db"], "fer": p[d]["bler"]} for p in points]
        s1 = snr_at_fer(pts, 0.01)
        # iterations at the grid point nearest the 1 % crossing
        near = min(points, key=lambda p: abs(p["esno_db"] - (s1 if s1 is not None else ref)))
        out["decoders"][d] = {"esno_db_at_bler1": s1, "esno_db_at_bler10": snr_at_fer(pts, 0.1),
                              "iters_near_bler1": near[d]["iters"]}
    base = out["decoders"][BASELINE]["esno_db_at_bler1"]
    for d, r in out["decoders"].items():
        r["loss_db"] = (r["esno_db_at_bler1"] - base
                        if r["esno_db_at_bler1"] is not None and base is not None else None)
    return out


def main():
    profile = profile_arg()
    prof = PROFILES[profile]
    t0 = time.time()
    results = {}
    print("decoders: " + " | ".join(DECODERS))
    for mcs in [m for m in MCS_TABLE if m.name in MCS_NAMES]:
        print(mcs.name)
        results[mcs.name] = run_mcs(mcs, prof)
    lines = [f"\nloss in dB at 1 % BLER vs '{BASELINE}' (iterations near 1 %)",
             f"{'decoder':22s} " + " ".join(f"{m:>16s}" for m in results)]
    for d in DECODERS:
        cells = []
        for m, r in results.items():
            x = r["decoders"][d]
            cells.append("      -        " if x["loss_db"] is None else f"{x['loss_db']:+6.2f} ({x['iters_near_bler1']:4.1f})")
        lines.append(f"{d:22s} " + " ".join(f"{c:>16s}" for c in cells))
    text = "\n".join(lines)
    print(text)
    write_result(NAME + (".quick" if profile == "quick" else ""), {
        "profile": profile, "baseline": BASELINE,
        "decoders": {d: {"demapper": dem, "algo": a, "kwargs": kw, "llr_step_nat": s}
                     for d, (dem, a, kw, s) in DECODERS.items()},
        "results": results, "summary": text, "seconds": round(time.time() - t0),
    })


if __name__ == "__main__":
    main()
