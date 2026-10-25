"""The product channel estimator (M4): separable, fixed, designed offline.

M3's receiver used GenieLmmseEstimator, a joint 2D filter built from the
true channel statistics per noise level: 2,388 x 398 complex taps, a new
set for every channel. The product receiver cannot know the statistics and
the FPGA cannot afford the matrix. It filters in two separable steps with
taps designed once for a worst-case channel, one set per SNR range:

1. Frequency, per pilot symbol: each subcarrier's estimate is a `taps`-tap
   Wiener filter over the nearest pilot subcarriers (window clamped at the
   band edges), for an exponential power delay profile of rms delay_spread_s
   truncated at max_delay_s (the cyclic prefix).
2. Time, per data symbol: real Wiener weights over the pilot symbols for a
   Jakes spectrum at doppler_hz, given the noise left after step 1.

Both steps are constrained to unit gain on a constant channel: a plain
Wiener filter designed for more Doppler than the channel has scales the
estimate and costs 64-QAM 1 to 3 dB. The defaults (300 ns, 1000 Hz, 32
taps; 1500 Hz with three pilot symbols) came from a scan of delay spread,
Doppler and taps against the genie estimator on AWGN, TDL-D at 600 and
2000 Hz and TDL-C at 300 ns: within 0.25 dB of the genie on all but p3
at 2000 Hz with 64-QAM (0.5 dB). A design Doppler matched to the channel
is worse than that under TDL-D: its line of sight keeps the channel more
correlated in time than the Rayleigh Jakes model assumes.

Interior subcarriers share one tap set; only the edges and the DC gap
differ, so the coefficient ROM is small. The design also gives r per RE,
1 / (1 + noise gain of the estimate), the demapper's weight for the
estimation error (GenieLmmseEstimator's err_var, noise part only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.special import j0

from .waveform import Numerology


@dataclass(frozen=True)
class EstimatorDesign:
    delay_spread_s: float = 300e-9
    max_delay_s: float = 3.2e-6
    doppler_hz: float = 1000.0     # two pilot symbols; 1500 for the three-pilot format
    taps: int = 32
    snr_step_db: float = 3.0       # design SNRs on this grid; the nearest is used
    # weights constrained to sum to 1 (unbiased for a constant channel). A
    # plain Wiener filter designed for more Doppler or delay spread than the
    # channel has scales the estimate (1.17 x at mid-frame for a 1000 Hz
    # design on a static channel), which costs 1 to 3 dB on 64-QAM.
    unbiased: bool = True
    # Phase tracker: the fraction of the fitted per-frame phase ramp that
    # is removed. The tracker cannot tell residual CFO from the line-of-sight
    # path's Doppler, and removing all of the latter shifts the diffuse
    # paths' Doppler spectrum by as much, beyond what two pilot symbols can
    # interpolate. Half the ramp matched or beat the genie estimator on
    # TDL-D 600 Hz with 100 and 300 Hz residual CFO (full removal: 2.5x the
    # FER). With three pilot symbols at 2000 Hz any removal costs (FER 0.26
    # at half, 0.02 at none): use 0 there.
    track_gain: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)

    def design_snr_db(self, esno_db: float) -> float:
        return self.snr_step_db * round(esno_db / self.snr_step_db)


def subcarrier_offsets(num: Numerology) -> np.ndarray:
    """Offset from the carrier, in subcarriers, of each effective subcarrier
    (Sionna's centred order, guards and DC removed)."""
    n = num.fft_size
    idx = [i for i in range(num.num_guard[0], n - num.num_guard[1]) if not (num.dc_null and i == n // 2)]
    return np.array(idx) - n // 2


def freq_correlation(dk: np.ndarray, spacing_hz: float, d: EstimatorDesign) -> np.ndarray:
    """E[h(k) h*(k - dk)] for the design power delay profile."""
    tau = np.arange(0.0, d.max_delay_s + 1e-12, 5e-9)
    p = np.exp(-tau / d.delay_spread_s)
    p /= p.sum()
    return (p * np.exp(-2j * np.pi * np.asarray(dk)[..., None] * spacing_hz * tau)).sum(-1)


def time_correlation(ds: np.ndarray, symbol_s: float, d: EstimatorDesign) -> np.ndarray:
    return j0(2 * np.pi * d.doppler_hz * symbol_s * np.asarray(ds, dtype=float))


def _wiener(a: np.ndarray, r: np.ndarray, unbiased: bool) -> np.ndarray:
    """Row weights w minimising E|h - w x|^2 for x = h + noise with
    E[x x^H] = a and E[h x^H] = r; unbiased: subject to sum(w) = 1."""
    w = np.linalg.solve(a.T, r)
    if not unbiased:
        return w
    g = np.linalg.solve(a.T, np.ones(len(r)))
    return w + g * (1 - w.sum()) / g.sum()


def design_filters(num: Numerology, d: EstimatorDesign, esno_db: float) -> dict:
    """Taps for the design SNR nearest esno_db: bt [F, taps] complex (with
    the LS's 1/sqrt(2) folded in, see fxmodem.zig), start [F], at [S, Npsym],
    r [S, F]."""
    F, S, L = num.num_effective_subcarriers, num.num_ofdm_symbols, d.taps
    psym = list(num.pilot_ofdm_symbols)
    sigma2 = 10 ** (-d.design_snr_db(esno_db) / 10)
    k = subcarrier_offsets(num)
    bt = np.zeros((F, L), np.complex128)
    start = np.clip(np.arange(F) - L // 2, 0, F - L)
    for f in range(F):
        win = k[start[f]:start[f] + L]
        r_ww = freq_correlation(win[:, None] - win[None, :], num.subcarrier_spacing_hz, d)
        r_fw = freq_correlation(k[f] - win, num.subcarrier_spacing_hz, d)
        bt[f] = _wiener(r_ww + sigma2 * np.eye(L), r_fw, d.unbiased)
    gain_f = (np.abs(bt) ** 2).sum(-1)                                 # noise gain per subcarrier
    # time: the frequency filter leaves its average noise gain per pilot symbol
    ps = np.array(psym, dtype=float)
    r_tt = time_correlation(ps[:, None] - ps[None, :], num.symbol_duration_s, d)
    noise_t = sigma2 * gain_f.mean()
    at = np.zeros((S, len(psym)))
    for s in range(S):
        r_st = time_correlation(s - ps, num.symbol_duration_s, d)
        at[s] = _wiener(r_tt + noise_t * np.eye(len(psym)), r_st, d.unbiased)
    for t, s in enumerate(psym):
        at[s] = np.eye(len(psym))[t]                                   # pilot symbols: unused
    gain = (at ** 2).sum(-1)[:, None] * gain_f[None, :]
    return {"bt": (bt / math.sqrt(2)).astype(np.complex64), "start": start.astype(np.uint32),
            "at": at.astype(np.float32), "r": (1 / (1 + gain)).astype(np.float32)}
