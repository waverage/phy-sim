"""A TDL channel that is continuous in time across frames (M6).

Sionna's TDL draws new sinusoid phases and angles on every call, so each
batch row is an independent channel that starts at t = 0. That is right
for a FER curve and wrong for video: how long a fade lasts, and so how
many consecutive frames it takes, is what the link layer has to survive.

ContinuousTdl is the same sum-of-sinusoids model (Sionna's TDL: the same
power delay profile, line of sight, 20 sinusoids per path, angles and
phases drawn the same way) with the random draws kept per link, so it can
be evaluated at any absolute time. Each of `num_links` links is one
independent channel; consecutive frames of a link see one fading process.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from sionna.phy.channel import time_lag_discrete_time_channel
from sionna.phy.channel.tr38901 import TDL

SPEED_OF_LIGHT = 299_792_458.0


class ContinuousTdl:
    def __init__(self, num_links: int, *, model: str = "D", delay_spread_s: float = 100e-9,
                 doppler_hz: float = 600.0, carrier_hz: float = 5.8e9, fs: float = 10e6,
                 seed: int = 0, device: str = "cpu"):
        speed = doppler_hz * SPEED_OF_LIGHT / carrier_hz
        tdl = TDL(model, delay_spread=delay_spread_s, carrier_frequency=carrier_hz,
                  min_speed=speed, max_speed=speed, device=device)
        self.num_links, self.fs, self.device = num_links, fs, device
        self.doppler_hz, self.model = doppler_hz, model
        self.los = bool(tdl.los)
        self.tau = tdl.delays.to(torch.float64).to(device)                     # [P]
        self.l_min, self.l_max = time_lag_discrete_time_channel(fs, max(float(self.tau.max()), 1 / fs))
        self.l_tot = self.l_max - self.l_min + 1
        p = self.tau.numel()
        s = int(tdl._num_sinusoids)
        g = torch.Generator(device="cpu").manual_seed(seed)
        rand = lambda *shape: torch.rand(*shape, generator=g, dtype=torch.float64).to(device)
        alpha = (2 * math.pi / s * torch.arange(1, s + 1, dtype=torch.float64, device=device)
                 + rand(num_links, p, s) * (2 * math.pi / s) - math.pi / s)
        self.w = 2 * math.pi * doppler_hz * torch.cos(alpha)                    # [links, P, S] rad/s
        self.phi = rand(num_links, p, s) * 2 * math.pi - math.pi
        self.amp = torch.sqrt(torch.real(tdl._mean_powers).to(torch.float64).to(device)) / math.sqrt(s)
        if self.los:
            self.los_amp = math.sqrt(float(tdl.mean_power_los))
            self.w_los = 2 * math.pi * doppler_hz * math.cos(float(tdl._los_angle_of_arrival))
            self.phi_los = rand(num_links) * 2 * math.pi - math.pi
            self.doppler_centroid_hz = float(tdl.mean_power_los) * math.cos(float(tdl._los_angle_of_arrival)) * doppler_hz
        else:
            self.doppler_centroid_hz = 0.0
        # gains at fs / decim, linear interpolation (as OfdmLink.tdl_decim)
        self.decim = 64 if doppler_hz == 0 else int(min(64, max(1, fs / (200 * doppler_hz))))

    def _rows(self, link) -> torch.Tensor:
        if link is None:
            return torch.arange(self.num_links, device=self.device)
        return torch.as_tensor(np.asarray(link), device=self.device).long()

    def gains(self, t: torch.Tensor, link=None) -> torch.Tensor:
        """Path gains a [rows, P, T] (complex128) at times t [rows, T] in
        seconds; row r is link link[r] (default: row r is link r)."""
        li = self._rows(link)
        arg = self.w[li][..., None] * t[:, None, None, :] + self.phi[li][..., None]   # [R, P, S, T]
        a = torch.polar(torch.ones_like(arg), arg).sum(-2) * self.amp[None, :, None]
        if self.los:
            spec = torch.polar(torch.full_like(t, self.los_amp), self.w_los * t + self.phi_los[li][:, None])
            a[:, 0] += spec
        return a

    def time_channel(self, n0: np.ndarray, n: int, link=None) -> torch.Tensor:
        """Discrete-time taps h [rows, n + l_tot - 1, l_tot] (complex64) for
        an input of n samples starting at absolute sample n0[row] of link
        link[row]; the same convention as OfdmLink.time_channel
        (ApplyTimeChannel's input)."""
        m = n + self.l_tot - 1
        d = self.decim
        k = m // d + 2
        n0 = torch.as_tensor(np.asarray(n0, np.float64), device=self.device)
        t = (n0[:, None] + d * torch.arange(k, device=self.device, dtype=torch.float64)[None, :]) / self.fs
        a_c = self.gains(t, link)                                                     # [R, P, k]
        ti = torch.arange(m, device=self.device, dtype=torch.float64) / d
        i0 = ti.floor().long()
        w = (ti - i0)
        a = a_c[..., i0] * (1 - w) + a_c[..., i0 + 1] * w                              # [L, P, m]
        lags = torch.arange(self.l_min, self.l_max + 1, device=self.device, dtype=torch.float64)
        s = torch.sinc(lags[None, :] - self.tau[:, None] * self.fs).to(a.dtype)       # [P, l]
        return torch.einsum("lpt,pk->ltk", a, s).to(torch.complex64)

    def frequency_response(self, t: np.ndarray, freqs_hz: np.ndarray, link=None) -> np.ndarray:
        """H [rows, T, F] at times t [rows, T] (s) and baseband frequencies."""
        a = self.gains(torch.as_tensor(np.asarray(t, np.float64), device=self.device), link)   # [R, P, T]
        f = torch.as_tensor(np.asarray(freqs_hz, np.float64), device=self.device)
        e = torch.exp(-2j * math.pi * f[None, :] * self.tau[:, None])                     # [P, F]
        return torch.einsum("lpt,pf->ltf", a, e).cpu().numpy()
