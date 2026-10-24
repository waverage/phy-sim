"""RF and converter impairments as plain torch functions on complex baseband.

Every impairment has the signature fn(x, fs) -> y, where x is a complex
tensor whose last dimension is time (one frame per leading index) and fs is
the sample rate in Hz. Bind the parameters with functools.partial and hand
the result to OfdmLink's impairment hooks. Random draws use torch's global
generator, so a sweep's torch.manual_seed makes them reproducible.

Power references are taken from the signal itself (the mean over the whole
tensor), so the functions do not need to know the link's scaling.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

Impairment = Callable[[torch.Tensor, float], torch.Tensor]


def _n(x: torch.Tensor) -> torch.Tensor:
    """Sample index along the time axis, as a real tensor on x's device."""
    return torch.arange(x.shape[-1], device=x.device, dtype=x.real.dtype)


def mean_power(x: torch.Tensor) -> torch.Tensor:
    return x.abs().square().mean()


def cfo(x: torch.Tensor, fs: float, offset_hz: float) -> torch.Tensor:
    """Carrier frequency offset: a constant frequency shift of offset_hz.
    20 ppm at 2.4 GHz is 48 kHz, more than one subcarrier at 10 MHz/256, so
    the raw offset is for testing a synchroniser; residual CFO is what the
    link experiments sweep."""
    return x * torch.exp(1j * 2 * math.pi * offset_hz / fs * _n(x))


def sco(x: torch.Tensor, fs: float, ppm: float, taps: int = 16) -> torch.Tensor:
    """Sampling clock offset: resample x at t_n = n (1 + ppm 1e-6) with a
    Hann-windowed sinc interpolator of `taps` taps. Samples that would need
    data beyond the ends read zeros."""
    if ppm == 0:
        return x
    t = _n(x) * (1 + ppm * 1e-6)
    i0 = torch.floor(t)
    frac = t - i0
    half = taps // 2
    m = torch.arange(-half + 1, half + 1, device=x.device, dtype=t.dtype)  # tap offsets
    idx = i0[:, None] + m[None, :]                                          # [N, taps]
    d = frac[:, None] - m[None, :]
    w = torch.sinc(d) * (0.5 + 0.5 * torch.cos(math.pi * d / (half + 1)))
    valid = (idx >= 0) & (idx < x.shape[-1])
    w = torch.where(valid, w, torch.zeros_like(w))
    g = x[..., idx.clamp(0, x.shape[-1] - 1).long()]                        # [..., N, taps]
    return (g * w.to(x.dtype)).sum(-1)


def phase_noise(x: torch.Tensor, fs: float, dbc_hz: float = -94.0, offset_hz: float = 100e3) -> torch.Tensor:
    """Free-running oscillator phase noise as a Wiener process. Its phase
    variance grows as s2 t, which gives the Lorentzian SSB spectrum
    L(f) ~ s2 / (4 pi^2 f^2) far from the carrier; s2 is set so L passes
    through dbc_hz at offset_hz. The default is a placeholder for the AD9363
    LO at 2.4 GHz; check it against the datasheet mask before trusting
    absolute numbers. Model TX and RX together by raising dbc_hz by 3 dB."""
    step_var = 4 * math.pi ** 2 * offset_hz ** 2 * 10 ** (dbc_hz / 10) / fs
    inc = torch.randn(x.shape, device=x.device, dtype=x.real.dtype) * math.sqrt(step_var)
    return x * torch.exp(1j * torch.cumsum(inc, dim=-1))


def rapp_pa(x: torch.Tensor, fs: float, backoff_db: float = 8.0, p: float = 2.0) -> torch.Tensor:
    """Rapp solid-state PA (AM/AM only), small-signal gain 1. backoff_db is
    the input back-off: saturation power over the mean input power."""
    a_sat = torch.sqrt(mean_power(x) * 10 ** (backoff_db / 10))
    r = x.abs() / a_sat
    return x / (1 + r ** (2 * p)) ** (1 / (2 * p))


def iq_imbalance(x: torch.Tensor, fs: float, gain_db: float = 0.5, phase_deg: float = 2.0) -> torch.Tensor:
    """Receiver IQ imbalance, y = mu x + nu conj(x), gain and phase error on
    the Q branch. Image rejection is |mu / nu|^2 (about 29 dB at 0.5 dB, 2 deg)."""
    g = 10 ** (gain_db / 20)
    phi = math.radians(phase_deg)
    mu = (1 + g * complex(math.cos(phi), -math.sin(phi))) / 2
    nu = (1 - g * complex(math.cos(phi), math.sin(phi))) / 2
    return mu * x + nu * x.conj()


def adc(x: torch.Tensor, fs: float, bits: int = 12, headroom_db: float = 12.0, agc_error_db: float = 0.0) -> torch.Tensor:
    """AGC plus a `bits`-bit ADC on I and Q. The AGC measures each frame's
    RMS and sets full scale headroom_db above it; agc_error_db is a gain error
    (positive: signal too strong, less headroom, more clipping). The output is
    scaled back, so only clipping and quantisation remain."""
    rms = torch.sqrt(x.abs().square().mean(dim=-1, keepdim=True)).clamp_min(1e-12)
    full = rms * 10 ** ((headroom_db - agc_error_db) / 20) / math.sqrt(2)   # per-rail full scale
    step = 2 * full / 2 ** bits

    def q(v):
        return ((torch.floor(v / step) + 0.5) * step).clamp(-full + step / 2, full - step / 2)

    return torch.complex(q(x.real), q(x.imag))


def wifi_burst(x: torch.Tensor, fs: float, sir_db: float = 10.0, band_hz: tuple[float, float] = (0.0, 5e6),
               burst_s: float = 200e-6, prob: float = 1.0) -> torch.Tensor:
    """A Wi-Fi packet overlapping our channel, modelled as Gaussian noise (an
    OFDM signal is close to Gaussian) confined to band_hz (the part of the
    20 MHz Wi-Fi channel that falls inside our sampled band) and to a burst of
    burst_s at a random start, present in each frame with probability prob.
    sir_db is the mean signal power over the interferer power while on."""
    n = x.shape[-1]
    f = torch.fft.fftfreq(n, d=1 / fs).to(x.device)
    mask = ((f >= band_hz[0]) & (f < band_hz[1])).to(x.dtype)
    w = torch.randn(x.shape, device=x.device, dtype=x.dtype)          # unit-power complex noise
    w = torch.fft.ifft(torch.fft.fft(w) * mask)
    w = w / torch.sqrt(mean_power(w))
    lead = x.shape[:-1]
    length = min(n, int(round(burst_s * fs)))
    start = torch.randint(-length + 1, n, lead + (1,), device=x.device)
    t = torch.arange(n, device=x.device)
    on = (t >= start) & (t < start + length)
    on &= (torch.rand(lead + (1,), device=x.device) < prob)
    return x + w * on * torch.sqrt(mean_power(x) / 10 ** (sir_db / 10))


def oversampled(fn: Impairment, factor: int = 4) -> Impairment:
    """Run `fn` at `factor` times the sample rate: band-limited upsampling by
    FFT zero-padding, fn, then an ideal low-pass back to the original rate.
    Needed for nonlinearities, whose spectral regrowth would otherwise alias
    back into the band."""
    def g(x: torch.Tensor, fs: float) -> torch.Tensor:
        return downsample(fn(upsample(x, factor), fs * factor), factor)
    return g


def upsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    n = x.shape[-1]
    xf = torch.fft.fft(x, norm="ortho")
    pad = torch.zeros(x.shape[:-1] + (n * (factor - 1),), dtype=x.dtype, device=x.device)
    xf = torch.cat([xf[..., : (n + 1) // 2], pad, xf[..., (n + 1) // 2:]], dim=-1)
    return torch.fft.ifft(xf, norm="ortho") * math.sqrt(factor)


def downsample(y: torch.Tensor, factor: int) -> torch.Tensor:
    n = y.shape[-1] // factor
    yf = torch.fft.fft(y, norm="ortho")
    yf = torch.cat([yf[..., : (n + 1) // 2], yf[..., y.shape[-1] - n // 2:]], dim=-1)
    return torch.fft.ifft(yf, norm="ortho") / math.sqrt(factor)


def chain(*fns: Impairment) -> Impairment:
    """Compose impairments left to right."""
    def g(x: torch.Tensor, fs: float) -> torch.Tensor:
        for fn in fns:
            x = fn(x, fs)
        return x
    return g
