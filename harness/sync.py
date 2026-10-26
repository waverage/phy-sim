"""Synchronisation (M5): the preamble, the synchroniser, and a link that
puts each frame at an unknown time and carrier offset in a sample buffer.

Preamble (one symbol slot, 288 samples, ahead of the 14-symbol frame): the
short training field, nine repetitions of a 32-sample sequence. It is an
OFDM symbol on every 8th subcarrier (24 tones within the occupied band,
QPSK from the 38.211 Gold sequence, seed chosen for low PAPR), at the same
mean power as a data symbol unless boosted.

Synchroniser, per frame:
  1. detect: the lag-32 autocorrelation P over a 256-sample window against
     the energy E; the frame is there when 4 |P|^2 > threshold E^2 (the
     metric is 1 for a noiseless periodic signal);
  2. coarse timing at the metric's peak, coarse CFO from the angle of P
     there (range +-fs/64 = +-156 kHz);
  3. fine timing: after the coarse correction, the channel impulse
     response from the first pilot symbol (FFT, times the conjugate pilot,
     IFFT); the frame starts at its first path (the earliest tap within a
     CP before the peak above first_path times the peak);
  4. medium CFO: the cyclic prefixes of the frame against their copies
     one FFT later (range +-fs/512 = +-19.5 kHz);
  5. out: the frame derotated by the coarse plus medium CFO, its FFT
     windows `backoff` samples before the first path. The receiver's
     per-frame phase tracker (range +-1.9 kHz) takes the rest.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch

from . import tables
from .design import subcarrier_offsets
from .modem import _ptr, lib
from .waveform import Numerology

STF_PERIOD = 32
STF_REPS = 9
STF_SEED = 0x4E533C        # c_init of the STF tones: stf_seed_search's pick, PAPR 4.3 dB


def stf_offsets(num: Numerology, period: int = STF_PERIOD) -> np.ndarray:
    """Subcarrier offsets of the STF tones: multiples of fft_size / period
    inside the occupied band, DC excluded."""
    step = num.fft_size // period
    k = subcarrier_offsets(num)
    return np.array([v for v in k if v % step == 0 and v != 0])


def stf_tones(num: Numerology, seed: int = STF_SEED) -> np.ndarray:
    """QPSK value per STF tone (unit magnitude)."""
    s = tables.pilot_signs(len(stf_offsets(num)), seed).astype(np.float64)
    return (s[:, 0] + 1j * s[:, 1]) / math.sqrt(2)


def stf_symbol(num: Numerology, seed: int = STF_SEED, boost_db: float = 0.0) -> np.ndarray:
    """One fft_size-sample STF symbol (period STF_PERIOD), unitary IFFT, at the
    mean power of a data symbol (num_effective_subcarriers unit REs) times
    the boost."""
    k = stf_offsets(num)
    x = np.zeros(num.fft_size, np.complex128)
    amp = math.sqrt(num.num_effective_subcarriers / len(k) * 10 ** (boost_db / 10))
    x[k % num.fft_size] = amp * stf_tones(num, seed)
    return np.fft.ifft(x) * math.sqrt(num.fft_size)


def preamble(num: Numerology, seed: int = STF_SEED, boost_db: float = 0.0) -> np.ndarray:
    """STF_REPS periods: one slot of fft_size + cp_len samples."""
    s = stf_symbol(num, seed, boost_db)
    n = STF_PERIOD * STF_REPS
    assert n == num.fft_size + num.cp_len
    return s[np.arange(n) % STF_PERIOD]


def papr_db(x: np.ndarray, oversample: int = 4) -> float:
    n = len(x)
    xf = np.fft.fft(x)
    pad = np.concatenate([xf[: n // 2], np.zeros(n * (oversample - 1)), xf[n // 2:]])
    xo = np.fft.ifft(pad)
    p = np.abs(xo) ** 2
    return float(10 * np.log10(p.max() / p.mean()))


def stf_seed_search(num: Numerology, base: int = 0x4E5300, count: int = 256) -> tuple[int, float]:
    """The c_init in [base, base + count) whose STF has the lowest PAPR."""
    best = min(range(base, base + count), key=lambda s: papr_db(stf_symbol(num, s)[:STF_PERIOD]))
    return best, papr_db(stf_symbol(num, best)[:STF_PERIOD])


# ---- the synchroniser (floating point reference) ---------------------------------------------

@dataclass(frozen=True)
class SyncParams:
    """Each threshold is a small integer or a power of two, as the
    fixed-point synchroniser (modem/src/fxsync.zig) uses it."""
    window: int = 256              # autocorrelation window, samples
    threshold_q8: int = 26         # detection: 4 |P|^2 > threshold_q8 / 256 E^2 (0.10)
    first_path_shift: int = 3      # CIR first path: power at least the peak's / 2^shift ...
    noise_floor: int = 10          # ... and this times the mean power of the taps away from the peak
    backoff: int = 4               # FFT window starts this many samples before the first path
    cp_cfo: bool = True            # medium CFO from the cyclic prefixes
    cir_min_ratio: int = 8         # reject a detection whose CIR peak is below this times the mean

    @property
    def threshold(self) -> float:
        return self.threshold_q8 / 256

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SyncResult:
    detected: np.ndarray           # [batch] bool
    start: np.ndarray              # [batch] frame start (first path of the frame's first symbol)
    cfo: np.ndarray                # [batch] cycles per sample, coarse + medium
    coarse_start: np.ndarray
    coarse_cfo: np.ndarray
    metric: np.ndarray             # [batch] detection metric at the coarse peak
    cir_ratio: np.ndarray          # [batch] CIR peak over mean


def _running_sum(v: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([np.zeros(v.shape[:-1] + (1,), v.dtype), np.cumsum(v, -1)], -1)
    return c[..., w:] - c[..., :-w]


class FloatSync:
    """The synchroniser on one buffer per frame, float64."""

    def __init__(self, num: Numerology, p: SyncParams = SyncParams(), pilot_symbol: int | None = None,
                 pilots: np.ndarray | None = None):
        self.num, self.p = num, p
        self.slot = num.fft_size + num.cp_len
        self.pre = STF_PERIOD * STF_REPS
        self.psym = num.pilot_ofdm_symbols[0] if pilot_symbol is None else pilot_symbol
        F = num.num_effective_subcarriers
        if pilots is None:
            pilots = tables.pilots(F * len(num.pilot_ofdm_symbols))
        self.pilot = np.asarray(pilots[:F], np.complex128)       # the first pilot symbol, effective subcarriers
        self.bins = subcarrier_offsets(num) % num.fft_size

    def detect(self, r: np.ndarray):
        """Coarse stage on r [batch, len]: (detected, peak index, metric, P at peak)."""
        P_, W = STF_PERIOD, self.p.window
        c = r[:, P_:] * r[:, :-P_].conj()
        e = np.abs(r[:, P_:]) ** 2 + np.abs(r[:, :-P_]) ** 2
        Ps = _running_sum(c, W)
        Es = _running_sum(e, W)
        m = 4 * np.abs(Ps) ** 2 / np.maximum(Es, 1e-30) ** 2
        over = m > self.p.threshold
        det = over.any(-1)
        d0 = np.where(det, over.argmax(-1), 0)
        # peak of |P| within one preamble after the first crossing (the
        # energy is flat there, so no division is needed)
        idx = d0[:, None] + np.arange(self.pre + 1)[None, :]
        idx = np.minimum(idx, m.shape[-1] - 1)
        pw = np.take_along_axis(np.abs(Ps) ** 2, idx, -1)
        d = d0 + pw.argmax(-1)
        b = np.arange(r.shape[0])
        return det, d, m[b, d], Ps[b, d]

    def cir(self, r: np.ndarray, start: np.ndarray) -> np.ndarray:
        """Channel impulse response [batch, fft_size] from the first pilot
        symbol of a frame assumed to start at `start`; tap t is delay t (mod N)
        relative to that start."""
        N, cp = self.num.fft_size, self.num.cp_len
        ws = start + self.psym * self.slot + cp
        idx = ws[:, None] + np.arange(N)[None, :]
        y = np.fft.fft(np.take_along_axis(r, np.clip(idx, 0, r.shape[-1] - 1), -1), axis=-1) / math.sqrt(N)
        z = np.zeros_like(y)
        z[:, self.bins] = y[:, self.bins] * self.pilot.conj()
        return np.fft.ifft(z, axis=-1) * N / len(self.bins)

    def __call__(self, r: np.ndarray) -> SyncResult:
        r = np.asarray(r, np.complex128)
        batch, n = r.shape
        N, cp, p = self.num.fft_size, self.num.cp_len, self.p
        det, d, metric, P = self.detect(r)
        f1 = np.angle(P) / (2 * math.pi * STF_PERIOD)
        nn = np.arange(n)
        r1 = r * np.exp(-2j * math.pi * f1[:, None] * nn[None, :])
        coarse = d + self.pre
        h = self.cir(r1, coarse)
        pw = np.abs(h) ** 2
        pk = pw.argmax(-1)
        ratio = pw.max(-1) / np.maximum(pw.mean(-1), 1e-30)
        cand = (pk[:, None] - np.arange(cp - 1, -1, -1)[None, :]) % N         # cp taps ending at the peak, earliest first
        near = (pk[:, None] + np.arange(-cp, cp + 1)[None, :]) % N
        floor = (pw.sum(-1) - np.take_along_axis(pw, near, -1).sum(-1)) / (N - near.shape[-1])
        thr = np.maximum(pw.max(-1) / 2 ** p.first_path_shift, p.noise_floor * floor)
        ok = np.take_along_axis(pw, cand, -1) >= thr[:, None]
        ok[:, -1] = True                                                       # the peak itself
        first = np.take_along_axis(cand, ok.argmax(-1)[:, None], -1)[:, 0]
        tau = np.where(first >= N // 2, first - N, first)
        start = coarse + tau
        f = f1.copy()
        if p.cp_cfo:
            s = np.arange(self.num.num_ofdm_symbols)
            i0 = start[:, None, None] + s[None, :, None] * self.slot + np.arange(cp)[None, None, :]
            i0 = np.clip(i0.reshape(batch, -1), 0, n - N - 1)
            a = np.take_along_axis(r1, i0, -1)
            b = np.take_along_axis(r1, i0 + N, -1)
            f += np.angle((b * a.conj()).sum(-1)) / (2 * math.pi * N)
        det = det & (ratio >= p.cir_min_ratio)
        return SyncResult(det, start, f, coarse, f1, metric, ratio)

    def align(self, r: np.ndarray, res: SyncResult, length: int) -> np.ndarray:
        """The frame derotated by res.cfo, from backoff samples before its start:
        [batch, length] complex64 (zeros past the buffer)."""
        r = np.asarray(r, np.complex128)
        n = r.shape[-1]
        idx = res.start[:, None] - self.p.backoff + np.arange(length)[None, :]
        valid = (idx >= 0) & (idx < n)
        y = np.take_along_axis(r, np.clip(idx, 0, n - 1), -1) * valid
        y = y * np.exp(-2j * math.pi * res.cfo[:, None] * idx)
        return y.astype(np.complex64)


# ---- the fixed-point synchroniser (modem/src/fxsync.zig) --------------------------------------

class SyncConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "fft_size", "cp_len", "guard_lo", "guard_hi", "dc_null", "num_symbols", "pilot_symbol", "period", "reps",
        "window", "metric_shift", "threshold_q8", "first_path_shift", "noise_floor", "cir_min_ratio", "backoff",
        "cp_cfo", "tx_scale", "dac_bits", "stf_seed", "pilot_seed", "num_threads")]


class SyncOut(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_uint32) for n in ("detected",)]
                + [(n, ctypes.c_int32) for n in ("start", "peak", "tau", "ang_coarse", "ang_cp", "inc", "inc_coarse")]
                + [(n, ctypes.c_int64) for n in ("p_re", "p_im", "e", "cir_peak", "cir_sum")])


SYNC_OUT_FIELDS = [f for f, _ in SyncOut._fields_]
METRIC_SHIFT = 8


def sync_config(num: Numerology, p: SyncParams = SyncParams(), tx_scale: int = 512, dac_bits: int = 12,
                num_threads: int = 0, stf_seed: int = STF_SEED, pilot_seed: int = tables.PILOT_SEED) -> SyncConfig:
    return SyncConfig(
        fft_size=num.fft_size, cp_len=num.cp_len, guard_lo=num.num_guard[0], guard_hi=num.num_guard[1],
        dc_null=int(num.dc_null), num_symbols=num.num_ofdm_symbols, pilot_symbol=num.pilot_ofdm_symbols[0],
        period=STF_PERIOD, reps=STF_REPS, window=p.window, metric_shift=METRIC_SHIFT, threshold_q8=p.threshold_q8,
        first_path_shift=p.first_path_shift, noise_floor=p.noise_floor, cir_min_ratio=p.cir_min_ratio,
        backoff=p.backoff, cp_cfo=int(p.cp_cfo), tx_scale=tx_scale, dac_bits=dac_bits, stf_seed=stf_seed,
        pilot_seed=pilot_seed, num_threads=num_threads)


def _bind_sync():
    L = lib()
    if not hasattr(L, "_sync_bound"):
        i16p = ctypes.POINTER(ctypes.c_int16)
        i32p = ctypes.POINTER(ctypes.c_int32)
        L.nphy_sync_create.restype = ctypes.c_void_p
        L.nphy_sync_create.argtypes = [ctypes.POINTER(SyncConfig)]
        L.nphy_sync_destroy.argtypes = [ctypes.c_void_p]
        L.nphy_sync_preamble_len.restype = ctypes.c_uint32
        L.nphy_sync_preamble_len.argtypes = [ctypes.c_void_p]
        L.nphy_sync_preamble.restype = ctypes.c_int32
        L.nphy_sync_preamble.argtypes = [ctypes.c_void_p, i16p]
        L.nphy_sync_run.argtypes = [ctypes.c_void_p, i16p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(SyncOut)]
        L.nphy_sync_align.argtypes = [ctypes.c_void_p, i16p, ctypes.c_uint32, ctypes.c_uint32, i32p, i32p,
                                      ctypes.c_uint32, i16p]
        L._sync_bound = True
    return L


def inc_of(cfo: np.ndarray) -> np.ndarray:
    """NCO increment (2^32 per turn) of a CFO in cycles per sample."""
    return np.round(np.asarray(cfo, np.float64) * 2 ** 32).astype(np.int32)


class FxSync:
    """The fixed-point synchroniser on int16 [batch, len, 2] buffers of
    12-bit ADC samples."""

    def __init__(self, num: Numerology, p: SyncParams = SyncParams(), tx_scale: int = 512, dac_bits: int = 12,
                 num_threads: int = 0, cfg: SyncConfig | None = None):
        """cfg, when given (a stored configuration), replaces p and the widths."""
        self.num, self.p = num, p
        self.cfg = cfg if cfg is not None else sync_config(num, p, tx_scale, dac_bits, num_threads)
        self._lib = _bind_sync()
        self._h = self._lib.nphy_sync_create(ctypes.byref(self.cfg))
        if not self._h:
            raise ValueError("nphy_sync_create rejected the configuration")
        self.pre_len = self._lib.nphy_sync_preamble_len(self._h)

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.nphy_sync_destroy(self._h)
            self._h = None

    def preamble(self) -> np.ndarray:
        out = np.empty((self.pre_len, 2), np.int16)
        if self._lib.nphy_sync_preamble(self._h, _ptr(out, ctypes.POINTER(ctypes.c_int16))) != 0:
            raise RuntimeError("nphy_sync_preamble failed")
        return out

    def raw(self, buf: np.ndarray) -> np.ndarray:
        """Per-buffer SyncOut as a structured array."""
        buf = np.ascontiguousarray(buf, dtype=np.int16)
        if buf.size and np.abs(buf.astype(np.int32)).max() > 2048:
            # the correlator's widths (metric_shift 8, 64-bit products) assume 12-bit samples
            raise ValueError("the synchroniser takes 12-bit ADC samples")
        batch, n = buf.shape[0], buf.shape[1]
        out = (SyncOut * batch)()
        self._lib.nphy_sync_run(self._h, _ptr(buf, ctypes.POINTER(ctypes.c_int16)), n, batch, out)
        raw = np.ctypeslib.as_array(out)          # the C records, zero copy
        rec = np.empty(batch, dtype=[(f, np.int64) for f in SYNC_OUT_FIELDS])
        for f in SYNC_OUT_FIELDS:
            rec[f] = raw[f]
        return rec

    def __call__(self, buf: np.ndarray) -> SyncResult:
        return result_of(self.raw(buf), self.num.fft_size)

    def align(self, buf: np.ndarray, start: np.ndarray, inc: np.ndarray, length: int) -> np.ndarray:
        buf = np.ascontiguousarray(buf, dtype=np.int16)
        batch, n = buf.shape[0], buf.shape[1]
        st = np.ascontiguousarray(start, dtype=np.int32)
        ic = np.ascontiguousarray(inc, dtype=np.int32)
        out = np.empty((batch, length, 2), np.int16)
        i16p = ctypes.POINTER(ctypes.c_int16)
        i32p = ctypes.POINTER(ctypes.c_int32)
        self._lib.nphy_sync_align(self._h, _ptr(buf, i16p), n, batch, _ptr(st, i32p), _ptr(ic, i32p), length,
                                  _ptr(out, i16p))
        return out


def result_of(o: np.ndarray, fft_size: int) -> SyncResult:
    """A SyncResult from SyncOut records (FxSync.raw or TwinSync)."""
    e = np.maximum(o["e"].astype(np.float64), 1e-30)
    metric = 4 * (o["p_re"].astype(np.float64) ** 2 + o["p_im"].astype(np.float64) ** 2) / e ** 2
    ratio = o["cir_peak"] * float(fft_size) / np.maximum(o["cir_sum"], 1)
    return SyncResult(detected=o["detected"] == 3, start=o["start"].astype(np.int64), cfo=o["inc"] / 2 ** 32,
                      coarse_start=o["peak"].astype(np.int64) + STF_PERIOD * STF_REPS,
                      coarse_cfo=o["inc_coarse"] / 2 ** 32, metric=metric, cir_ratio=ratio)


# ---- the buffered link ------------------------------------------------------------------------

def frame_cfo(x: torch.Tensor, cfo: torch.Tensor) -> torch.Tensor:
    """A carrier offset per frame: x [batch, ..., time], cfo [batch] cycles/sample."""
    n = torch.arange(x.shape[-1], device=x.device, dtype=torch.float64)
    ph = 2 * math.pi * cfo.to(torch.float64).reshape(-1, *([1] * (x.dim() - 1))) * n
    return x * torch.exp(1j * ph).to(x.dtype)


def buffer_time_channel(link, batch: int, num_samples: int) -> torch.Tensor:
    """OfdmLink.time_channel for an arbitrary input length."""
    n = num_samples + link.l_tot - 1
    d = link.tdl_decim
    a_c, tau = link.tdl(batch, n // d + 2, link.fs / d)
    t = torch.arange(n, device=a_c.device) / d
    i0 = t.floor().long()
    w = (t - i0).to(a_c.dtype)
    a = a_c[..., i0] * (1 - w) + a_c[..., i0 + 1] * w
    lags = torch.arange(link.l_min, link.l_max + 1, device=tau.device, dtype=tau.dtype)
    s = torch.sinc(lags - tau[..., None] * link.fs)
    s = s[:, :, None, :, None].to(a.dtype)
    return torch.einsum("...pt,...pl->...tl", a, s)
