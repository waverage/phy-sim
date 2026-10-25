"""Monte Carlo sweep runner with the stop-at-N-frame-errors rule and a JSON
result format every experiment shares."""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import subprocess
import time

import numpy as np
import torch

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=RESULTS_DIR.parent).stdout.strip() or "none"
    except Exception:
        return "none"


def run_point(link, esno_db: float, *, batch: int, min_frame_errors: int, max_frames: int) -> dict:
    tot = {"frames": 0, "frame_errors": 0, "codewords": 0, "codeword_errors": 0, "bits": 0, "bit_errors": 0}
    t0 = time.time()
    while tot["frames"] < max_frames and tot["frame_errors"] < min_frame_errors:
        r = link.run(min(batch, max_frames - tot["frames"]), esno_db)
        for k, v in r.items():          # links may report extra counts
            tot[k] = tot.get(k, 0) + v
    tot.update(
        esno_db=esno_db,
        fer=tot["frame_errors"] / tot["frames"],
        bler=tot["codeword_errors"] / tot["codewords"],
        ber=tot["bit_errors"] / tot["bits"],
        seconds=round(time.time() - t0, 2),
    )
    return tot


# Run profiles. "quick" answers "is A better than B by more than ~0.5 dB" in
# seconds per MCS; "reference" is for curves that get committed and gated.
PROFILES = {
    "quick":     dict(step_db=1.0, batch=250, min_frame_errors=30,  max_frames=1_000),
    "reference": dict(step_db=0.5, batch=250, min_frame_errors=100, max_frames=5_000),
}


def snr_grid(start_db: float, span_db: float, profile: str = "reference"):
    return np.arange(start_db, start_db + span_db, PROFILES[profile]["step_db"])


def run_sweep(link, esno_dbs, *, batch=200, min_frame_errors=100, max_frames=10_000,
              seed=1, stop_below_fer=1e-3, log=print) -> list[dict]:
    """Sweep SNR upward; stop early once the FER is resolved below stop_below_fer
    (nothing more to learn higher up, the time is better spent elsewhere).
    Pass **sweep_kwargs(profile) for a standard profile."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    points = []
    for s in esno_dbs:
        pt = run_point(link, float(s), batch=batch, min_frame_errors=min_frame_errors, max_frames=max_frames)
        points.append(pt)
        if log:
            log(f"  Es/N0 {s:6.2f} dB  FER {pt['fer']:.4f}  BER {pt['ber']:.2e}  "
                f"frames {pt['frames']:6d}  {pt['seconds']:6.1f} s")
        if pt["fer"] < stop_below_fer and pt["frame_errors"] == 0:
            break
    return points


def snr_at_fer(points: list[dict], target: float) -> float | None:
    """Es/N0 where the FER curve crosses `target`, log-linear interpolation."""
    xs = [p["esno_db"] for p in points]
    ys = [max(p["fer"], 1e-12) for p in points]
    for i in range(1, len(xs)):
        if ys[i - 1] >= target > ys[i]:
            l0, l1, lt = np.log10(ys[i - 1]), np.log10(ys[i]), np.log10(target)
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (l0 - lt) / (l0 - l1))
    return None


def write_result(name: str, payload: dict) -> pathlib.Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    import sionna
    payload = {
        "experiment": name,
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_rev(),
        "sionna": sionna.__version__,
        "torch": torch.__version__,
        **payload,
    }
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1))
    return out


def sweep_kwargs(profile: str = "reference", seed: int = 1) -> dict:
    p = PROFILES[profile]
    return dict(batch=p["batch"], min_frame_errors=p["min_frame_errors"], max_frames=p["max_frames"], seed=seed)


def default_device() -> str:
    """The first CUDA GPU if there is one; Sionna accepts only cpu and cuda:N."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def run_curve(link, start_db: float, span_db: float, profile: str = "reference", seed: int = 1,
              log=print) -> dict:
    """Sweep one link from start_db upward and summarise it the way every
    experiment's JSON does."""
    pts = run_sweep(link, snr_grid(start_db, span_db, profile), **sweep_kwargs(profile, seed), log=log)
    return {"points": pts, "esno_db_at_fer10": snr_at_fer(pts, 0.10), "esno_db_at_fer1": snr_at_fer(pts, 0.01)}


def reference_esno(mcs_name: str, key: str = "esno_db_at_fer10") -> float:
    """A point of the committed e001 AWGN baseline, to start sweeps near the knee."""
    ref = json.loads((RESULTS_DIR / "e001_awgn_baseline.json").read_text())
    return ref["curves"][mcs_name][key]


def profile_arg(default: str = "quick") -> str:
    """`--profile quick|reference` from the command line."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default=default)
    return ap.parse_args().profile
