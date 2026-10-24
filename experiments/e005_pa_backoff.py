"""e005: PA back-off against EVM, spectral mask and link margin.

Hypothesis: with a Rapp p = 2 PA, 6 to 10 dB input back-off is the useful
range: below it the in-band EVM and the spectral regrowth break 64-QAM
and the mask, above it the output back-off wastes more power than the
distortion it removes.

Part A (no noise): OFDM frames through the PA at 4x oversampling; EVM over
the effective subcarriers, ACLR into the adjacent 10 MHz, and the margin to
a spectral mask. The mask is the 802.11 OFDM mask scaled to a 10 MHz channel
(0 dBr to 4.5 MHz, -20 at 5.5, -28 at 10, -40 at 15 MHz), a placeholder until
the regulatory target (ETSI EN 300 328 or FCC 15.247) is chosen. There is no
TX windowing or filtering, so the linear row shows the bare OFDM sidelobes.

Part B: FER through the PA on an AWGN channel with LMMSE estimation. Es/N0
is referred to the PA's mean output power; adding the output back-off refers
it to the saturated power instead, and the IBO that minimises that sum is the
best operating point.
Run:  .venv/bin/python -I experiments/e005_pa_backoff.py [--profile quick|reference]
"""
import functools
import math
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from harness import impairments as imp                      # noqa: E402
from harness.chain import OfdmLink                          # noqa: E402
from harness.sweep import (RESULTS_DIR, default_device, profile_arg, reference_esno,  # noqa: E402
                           run_curve, write_result)
from harness.waveform import MCS_TABLE, NUM_10MHZ           # noqa: E402

NAME = "e005_pa_backoff"
RAPP_P = 2.0
OVERSAMPLE = 4
IBOS_DB = (None, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0)          # None: linear PA
MCS_NAMES = ("16qam-3/4", "64qam-5/6")
MASK = ((0.0, 0.0), (4.5, 0.0), (5.5, -20.0), (10.0, -28.0), (15.0, -40.0), (1e3, -40.0))  # (MHz, dBr)
RBW_HZ = 100e3
SPAN_DB = 14.0


def label(ibo):
    return "linear" if ibo is None else f"ibo{ibo:g}"


def pa(ibo):
    """Rapp PA at IBO, run oversampled, with no normalisation (gain 1)."""
    return imp.oversampled(functools.partial(imp.rapp_pa, backoff_db=ibo, p=RAPP_P), OVERSAMPLE)


def pa_unit_power(ibo):
    """PA followed by a gain that restores the input mean power, so the link's
    Es/N0 is referred to the PA output power."""
    f = pa(ibo)

    def g(x, fs):
        y = f(x, fs)
        return y * torch.sqrt(imp.mean_power(x) / imp.mean_power(y))
    return g


def spectral(dev, frames=200):
    """Part A: EVM, ACLR, mask margin and the PSD for every IBO."""
    num = NUM_10MHZ
    link = OfdmLink(num, MCS_TABLE[-1], device=dev)
    torch.manual_seed(1)
    bits = torch.randint(0, 2, (frames, num.num_data_res * 6), device=dev).float()
    x_rg = link.rg_mapper(link.mapper(bits).reshape(frames, 1, 1, -1))
    x_t = link.modulator(x_rg)
    ref = link.remove_nulled(x_rg)
    x_up = imp.upsample(x_t, OVERSAMPLE)
    fs_up = num.bandwidth_hz * OVERSAMPLE
    n = x_up.shape[-1]
    f = torch.fft.fftshift(torch.fft.fftfreq(n, d=1 / fs_up)).cpu().numpy()
    smooth = max(1, int(round(RBW_HZ / (fs_up / n))))
    mask = np.interp(np.abs(f) / 1e6, *zip(*MASK))
    out_of_band = np.abs(f) >= 4.5e6
    rows, psds = {}, {}
    for ibo in IBOS_DB:
        y_up = x_up if ibo is None else imp.rapp_pa(x_up, fs_up, backoff_db=ibo, p=RAPP_P)
        p_in, p_out = float(imp.mean_power(x_up)), float(imp.mean_power(y_up))
        # PSD: averaged periodogram, smoothed to the RBW, in dB relative to the in-band peak
        w = torch.hann_window(n, device=dev, dtype=torch.float32)
        psd = torch.fft.fftshift(torch.fft.fft(y_up * w).abs().square().mean(dim=(0, 1, 2))).cpu().numpy()
        psd = np.convolve(psd, np.ones(smooth) / smooth, mode="same")
        psd_dbr = 10 * np.log10(psd / psd[np.abs(f) < 3.5e6].max())
        inband = psd[np.abs(f) < 5e6].sum()
        aclr = 10 * np.log10(max(psd[(f >= 5e6) & (f < 15e6)].sum(), psd[(f <= -5e6) & (f > -15e6)].sum()) / inband)
        y_rg = link.remove_nulled(link.demodulator(imp.downsample(y_up, OVERSAMPLE)))
        a = (ref.conj() * y_rg).sum() / ref.abs().square().sum()
        evm = 10 * math.log10(float((y_rg - a * ref).abs().square().mean() / (a * ref).abs().square().mean()))
        obo = None if ibo is None else ibo + 10 * math.log10(p_in / p_out)
        rows[label(ibo)] = {"ibo_db": ibo, "obo_db": obo, "evm_db": evm, "aclr_db": float(aclr),
                            "mask_margin_db": float((mask - psd_dbr)[out_of_band].min())}
        psds[label(ibo)] = psd_dbr
    return rows, f, psds, mask


def main():
    profile = profile_arg()
    dev = default_device()
    t0 = time.time()
    rows, f, psds, mask = spectral(dev)
    print(f"{'PA':8s} {'OBO dB':>7s} {'EVM dB':>7s} {'ACLR dB':>8s} {'mask margin':>12s}")
    for k, r in rows.items():
        obo = f"{r['obo_db']:7.2f}" if r["obo_db"] is not None else f"{'-':>7s}"
        print(f"{k:8s} {obo} {r['evm_db']:7.1f} {r['aclr_db']:8.1f} {r['mask_margin_db']:12.1f}")

    curves = {}
    for mcs in [m for m in MCS_TABLE if m.name in MCS_NAMES]:
        curves[mcs.name] = {}
        for ibo in IBOS_DB:
            print(f"{mcs.name} {label(ibo)}", flush=True)
            link = OfdmLink(NUM_10MHZ, mcs, device=dev, estimator="lmmse",
                            tx_impairments=None if ibo is None else pa_unit_power(ibo))
            c = run_curve(link, reference_esno(mcs.name) - 1.5, SPAN_DB, profile,
                          log=lambda s: print(s, flush=True))
            obo = rows[label(ibo)]["obo_db"] or 0.0
            c["esno_db_at_psat_fer1"] = c["esno_db_at_fer1"] + obo if c["esno_db_at_fer1"] is not None else None
            curves[mcs.name][label(ibo)] = c

    name = NAME if profile == "reference" else f"{NAME}.quick"
    out = write_result(name, {"profile": profile, "device": dev, "rapp_p": RAPP_P, "oversample": OVERSAMPLE,
                              "mask": MASK, "seconds": round(time.time() - t0),
                              "spectral": rows, "curves": curves})
    print(f"\nwrote {out}")
    print(f"\nEs/N0 for 1 % FER referred to PA output / to saturation (dB)")
    print(f"{'PA':8s} " + " ".join(f"{m:>16s}" for m in curves))
    for k in rows:
        cells = []
        for m in curves:
            c = curves[m][k]
            cells.append(f"{c['esno_db_at_fer1']:6.2f} / {c['esno_db_at_psat_fer1']:6.2f}"
                         if c["esno_db_at_fer1"] is not None else f"{'n/a':>16s}")
        print(f"{k:8s} " + " ".join(f"{c:>16s}" for c in cells))
    plot(rows, f, psds, mask, curves, name)


def plot(rows, f, psds, mask, curves, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for k, psd in psds.items():
        axs[0].plot(f / 1e6, psd, lw=0.8, label=k)
    axs[0].plot(f / 1e6, mask, "k--", lw=1, label="mask (placeholder)")
    axs[0].set(xlabel="MHz", ylabel="dBr", ylim=(-70, 5), xlim=(-20, 20), title=f"PSD, Rapp p={RAPP_P:g}")
    axs[0].grid(True, lw=0.3)
    axs[0].legend(fontsize=7)
    ibo_rows = [r for r in rows.values() if r["ibo_db"] is not None]
    x = [r["ibo_db"] for r in ibo_rows]
    axs[1].plot(x, [r["evm_db"] for r in ibo_rows], marker="o", label="EVM")
    axs[1].plot(x, [r["aclr_db"] for r in ibo_rows], marker="s", label="ACLR")
    axs[1].set(xlabel="input back-off (dB)", ylabel="dB", title="EVM and ACLR")
    axs[1].grid(True, lw=0.3)
    axs[1].legend(fontsize=8)
    for m, cs in curves.items():
        pts = [(rows[k]["ibo_db"], c["esno_db_at_psat_fer1"]) for k, c in cs.items()
               if rows[k]["ibo_db"] is not None and c["esno_db_at_psat_fer1"] is not None]
        axs[2].plot(*zip(*pts), marker="o", label=m)
    axs[2].set(xlabel="input back-off (dB)", ylabel="Es/N0 at Psat for 1 % FER (dB)",
               title="Link cost incl. back-off (lower is better)")
    axs[2].grid(True, lw=0.3)
    axs[2].legend(fontsize=8)
    fig.suptitle("e005 PA back-off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=130)


if __name__ == "__main__":
    main()
