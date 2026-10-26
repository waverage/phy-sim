"""The M4 fixed-point modem (modem/src/fx*.zig) from Python.

Every block exists twice: in Zig, which is the product model and the
golden model for the RTL, and as a NumPy twin in harness/twin.py. The two
are bit exact in fixed point. The Zig code also runs each block as
floating-point arithmetic of the same algorithm (no rounding, no
saturation), which is what the fixed-point loss is measured against.
"""
from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from . import tables
from .chain import OfdmLink, count_errors
from .design import EstimatorDesign, design_filters
from .modem import ZigModem, _f32p, _ptr, _u8p, lib
from .twin import qam_norm
from .waveform import MCS, Numerology

_u32p = ctypes.POINTER(ctypes.c_uint32)
_f64p = ctypes.POINTER(ctypes.c_double)
_i8p = ctypes.POINTER(ctypes.c_int8)
_i64p = ctypes.POINTER(ctypes.c_int64)
_i32p = ctypes.POINTER(ctypes.c_int32)


class DecoderParams(ctypes.Structure):
    """fxldpc.Params: widths in bits, offset in LLR LSBs."""
    _fields_ = [("llr_bits", ctypes.c_uint32), ("app_bits", ctypes.c_uint32), ("msg_bits", ctypes.c_uint32),
                ("offset", ctypes.c_float), ("num_iter", ctypes.c_uint32), ("early_stop", ctypes.c_uint32)]


ALGOS = {"bp": 0, "oms-float": 1, "oms-fixed": 2}


class FxConfig(ctypes.Structure):
    """fxmodem.FxConfig."""
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "fft_size", "cp_len", "guard_lo", "guard_hi", "dc_null", "num_symbols", "bits_per_symbol",
        "num_codewords", "n", "k", "phase_track", "num_threads", "tx_scale", "dac_bits", "freq_taps",
        "pilot_seed", "perm_seed", "track_gain")] + [("dec", DecoderParams), ("pilot_mask", _u8p)]


class Probe(ctypes.Structure):
    _fields_ = [("y", _f32p), ("h", _f32p), ("phi", _f32p)]


def _bind():
    L = lib()
    if not hasattr(L, "_fx_bound"):
        L.nphy_fx_ldpc_decode.restype = ctypes.c_int64
        L.nphy_fx_ldpc_decode.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _f32p, ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.POINTER(DecoderParams), ctypes.c_uint32, _u8p, _u32p]
        L.nphy_fx_create.restype = ctypes.c_void_p
        L.nphy_fx_create.argtypes = [ctypes.POINTER(FxConfig), ctypes.c_uint32]
        L.nphy_fx_destroy.argtypes = [ctypes.c_void_p]
        L.nphy_fx_num_samples.restype = ctypes.c_uint32
        L.nphy_fx_num_samples.argtypes = [ctypes.c_void_p]
        L.nphy_fx_set_filter.restype = ctypes.c_int32
        L.nphy_fx_set_filter.argtypes = [ctypes.c_void_p, _f32p, _u32p, _f32p, _f32p]
        L.nphy_fx_tx.argtypes = [ctypes.c_void_p, _u8p, ctypes.c_uint32, ctypes.c_void_p]
        L.nphy_fx_rx.restype = ctypes.c_int32
        L.nphy_fx_rx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32, _f64p,
                                 ctypes.c_uint32, _u8p, _f32p, ctypes.POINTER(Probe)]
        L.nphy_fx_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64 * 2), ctypes.c_uint32]
        L.nphy_fx_fft.restype = ctypes.c_int32
        L.nphy_fx_fft.argtypes = [ctypes.c_uint32, _i32p, ctypes.c_uint32, ctypes.c_uint32]
        L.nphy_fx_cordic_vector.argtypes = [_i64p, ctypes.c_uint32, _i64p]
        L.nphy_fx_cossin.argtypes = [_i64p, ctypes.c_uint32, _i32p]
        L.nphy_fx_tables.restype = ctypes.c_int32
        L.nphy_fx_tables.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _i8p, ctypes.c_uint32, ctypes.c_uint32, _u32p]
        L._fx_bound = True
    return L


# ---- single blocks ---------------------------------------------------------------------

def fft(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Fixed-point FFT of int I/Q pairs [..., n, 2] (16-bit values)."""
    x = np.ascontiguousarray(x, dtype=np.int32).copy()
    n = x.shape[-2]
    if _bind().nphy_fx_fft(n, _ptr(x, _i32p), x.size // (2 * n), int(inverse)) != 0:
        raise ValueError(n)
    return x


def cordic_vector(xy: np.ndarray) -> np.ndarray:
    xy = np.ascontiguousarray(xy, dtype=np.int64)
    out = np.empty_like(xy)
    _bind().nphy_fx_cordic_vector(_ptr(xy, _i64p), xy.shape[0], _ptr(out, _i64p))
    return out


def cossin(ang: np.ndarray) -> np.ndarray:
    ang = np.ascontiguousarray(ang, dtype=np.int64)
    out = np.empty((ang.size, 2), np.int32)
    _bind().nphy_fx_cossin(_ptr(ang, _i64p), ang.size, _ptr(out, _i32p))
    return out


def zig_tables(num_pilots: int, frame_bits: int, pilot_seed: int = tables.PILOT_SEED,
               perm_seed: int = tables.PERM_SEED) -> tuple[np.ndarray, np.ndarray]:
    signs = np.empty((num_pilots, 2), np.int8)
    perm = np.empty(frame_bits, np.uint32)
    _bind().nphy_fx_tables(num_pilots, pilot_seed, _ptr(signs, _i8p), frame_bits, perm_seed, _ptr(perm, _u32p))
    return signs, perm


# ---- frame modem -------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedPoint:
    """The M4 widths and scalings (decoder choice from e012)."""
    llr_bits: int = 6
    msg_bits: int = 6
    app_bits: int = 8
    llr_step_nat: float = 0.25        # nats per LLR LSB
    offset_lsb: int = 2               # min-sum offset, 0.5 nat
    num_iter: int = 20
    tx_scale: int = 512               # IFFT input amplitude per unit: TX rms about 320 LSB per rail
    dac_bits: int = 12
    adc_bits: int = 12
    agc_headroom_db: float = 12.0     # ADC full scale over the per-rail rms (e008)
    # LLR scale cap: at most llr_cap / (a P), P the AGC's target power, so a
    # typical LLR (4 a^2 SNR nats) maps to about llr_cap * 4 LSB. Without it
    # every LLR saturates at high SNR and the decoder loses the difference
    # between strong and faded REs (TDL-C, 64-QAM: an error floor). Too low a
    # cap (M4's 8) floors the rate-3/4 codes on TDL-C instead: an RE 20 dB
    # below the frame's mean gain is still reliable at high SNR, but its LLR
    # comes out at a few LSB and the min-sum offset erases it (qpsk-3/4 at
    # 40 dB: 2.4 % FER at 8, none at 16; 64qam-5/6: 18 % to 7 %; TDL-D and
    # the AWGN waterfalls unchanged). M6 raised it to 16.
    llr_cap: float = 16.0

    @property
    def agc_power(self) -> float:
        """Target power per complex sample in ADC units squared."""
        return 2 * (2 ** (self.adc_bits - 1) * 10 ** (-self.agc_headroom_db / 20)) ** 2

    def as_dict(self) -> dict:
        return asdict(self)


def track_gain_q15(g: float) -> int:
    return int(round(g * 32768))


class FxModem:
    """One configured M4 modem, fixed point (fixed=True: int16 I/Q samples)
    or the same algorithm in floating point (complex64 samples)."""

    def __init__(self, num: Numerology, mcs: MCS, *, k: int, n: int, num_codewords: int, pilot_mask: np.ndarray,
                 fixed: bool, fp: FixedPoint = FixedPoint(), taps: int = 32, phase_track: bool = False,
                 track_gain: float = 0.5, early_stop: bool = True, num_threads: int = 0,
                 pilot_seed: int = tables.PILOT_SEED, perm_seed: int = tables.PERM_SEED):
        self.num, self.mcs, self.k, self.n, self.num_codewords = num, mcs, k, n, num_codewords
        self.fixed, self.fp = fixed, fp
        self._mask = np.ascontiguousarray(pilot_mask, dtype=np.uint8).reshape(-1)
        dec = DecoderParams(fp.llr_bits, fp.app_bits, fp.msg_bits, float(fp.offset_lsb), fp.num_iter, int(early_stop))
        cfg = FxConfig(
            fft_size=num.fft_size, cp_len=num.cp_len, guard_lo=num.num_guard[0], guard_hi=num.num_guard[1],
            dc_null=int(num.dc_null), num_symbols=num.num_ofdm_symbols, bits_per_symbol=mcs.bits_per_symbol,
            num_codewords=num_codewords, n=n, k=k, phase_track=int(phase_track), num_threads=num_threads,
            tx_scale=fp.tx_scale, dac_bits=fp.dac_bits, freq_taps=taps, pilot_seed=pilot_seed, perm_seed=perm_seed,
            track_gain=track_gain_q15(track_gain), dec=dec, pilot_mask=_ptr(self._mask, _u8p))
        self._lib = _bind()
        self._h = self._lib.nphy_fx_create(ctypes.byref(cfg), int(fixed))
        if not self._h:
            raise ValueError("nphy_fx_create rejected the configuration")
        self.num_samples = self._lib.nphy_fx_num_samples(self._h)

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.nphy_fx_destroy(self._h)
            self._h = None

    def set_filter(self, filt: dict):
        """filt: design.design_filters output."""
        f = {"bt": np.ascontiguousarray(filt["bt"], dtype=np.complex64),
             "start": np.ascontiguousarray(filt["start"], dtype=np.uint32),
             "at": np.ascontiguousarray(filt["at"], dtype=np.float32),
             "r": np.ascontiguousarray(filt["r"], dtype=np.float32)}
        rc = self._lib.nphy_fx_set_filter(self._h, _ptr(f["bt"].view(np.float32), _f32p), _ptr(f["start"], _u32p),
                                          _ptr(f["at"], _f32p), _ptr(f["r"], _f32p))
        if rc != 0:
            raise ValueError("nphy_fx_set_filter rejected the filter")
        self.filt = f

    def tx(self, bits: np.ndarray) -> np.ndarray:
        """bits [batch, num_codewords, k] -> int16 [batch, num_samples, 2]
        (fixed) or complex64 [batch, num_samples] (float)."""
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        batch = bits.shape[0]
        out = (np.empty((batch, self.num_samples, 2), np.int16) if self.fixed
               else np.empty((batch, self.num_samples), np.complex64))
        self._lib.nphy_fx_tx(self._h, _ptr(bits, _u8p), batch, out.ctypes.data_as(ctypes.c_void_p))
        return out

    def rx(self, y: np.ndarray, llr_scale, l_min: int = 0, return_llr: bool = False, probe: bool = False):
        """y: int16 [batch, len, 2] (fixed) or complex64 [batch, len];
        llr_scale [batch]: a / (N0 * llr_step), N0 per RE in squared input
        units. Returns bits [batch, num_codewords, k], plus the LLRs
        [batch, num_codewords, n] and/or a dict of intermediate results."""
        y = np.ascontiguousarray(y, dtype=np.int16 if self.fixed else np.complex64)
        batch, length = y.shape[0], y.shape[1]
        scale = np.ascontiguousarray(np.broadcast_to(llr_scale, (batch,)), dtype=np.float64)
        out = np.empty((batch, self.num_codewords, self.k), np.uint8)
        llr = np.empty((batch, self.num_codewords, self.n), np.float32) if return_llr else None
        S, F = self.num.num_ofdm_symbols, self.num.num_effective_subcarriers
        pr = None
        p = Probe()
        if probe:
            pr = {"y": np.empty((batch, S * F, 2), np.float32),
                  "h": np.empty((batch, self.num.num_data_res, 2), np.float32),
                  "phi": np.empty((batch, S), np.float32)}
            p = Probe(*(_ptr(pr[k], _f32p) for k in ("y", "h", "phi")))
        rc = self._lib.nphy_fx_rx(self._h, y.ctypes.data_as(ctypes.c_void_p), length, l_min, _ptr(scale, _f64p),
                                  batch, _ptr(out, _u8p), None if llr is None else _ptr(llr, _f32p), ctypes.byref(p))
        if rc != 0:
            raise RuntimeError("nphy_fx_rx failed (no filter set, or y too short)")
        res = (out,) + ((llr,) if return_llr else ()) + ((pr,) if probe else ())
        return res[0] if len(res) == 1 else res

    def decoder_stats(self, reset: bool = True) -> tuple[int, int]:
        out = (ctypes.c_uint64 * 2)()
        self._lib.nphy_fx_stats(self._h, ctypes.byref(out), int(reset))
        return int(out[0]), int(out[1])


def agc_adc(y: np.ndarray, fp: FixedPoint) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal per-frame AGC, then the ADC. y [batch, len] complex. Returns
    (y in ADC units, complex64; the int16 ADC samples [batch, len, 2]; the
    gain per frame)."""
    rms = np.sqrt((np.abs(y) ** 2).mean(-1, keepdims=True)).clip(1e-12)
    g = math.sqrt(fp.agc_power) / rms
    ys = (y * g).astype(np.complex64)
    hi = 2 ** (fp.adc_bits - 1)
    q = np.stack([ys.real, ys.imag], -1).astype(np.float64)
    yq = np.clip(np.floor(q + 0.5), -hi, hi - 1).astype(np.int16)
    return ys, yq, g[:, 0]


def frame_tables(num: Numerology, mcs: MCS, mask: np.ndarray):
    """The generated pilots and interleaver of a numerology and MCS."""
    num_pilots = int(np.asarray(mask).sum())
    return tables.pilots(num_pilots), tables.interleaver(num.num_data_res * mcs.bits_per_symbol)


class FxOfdmLink:
    """The M4 link: the fixed-point transmitter, Sionna's channel,
    impairments and noise (OfdmLink.propagate), then three receivers on the
    same samples:
      genie  the M3 f32 receiver with GenieLmmseEstimator (the e011 curves);
      float  the product algorithm in floating point after an ideal AGC;
      fixed  the product receiver in fixed point after AGC and ADC.
    run() reports the fixed receiver's counts, float_* and genie_* counts,
    disagree_codewords (fixed vs float) and each receiver's wall time.
    The waveform uses the generated tables (harness/tables.py)."""

    def __init__(self, num: Numerology, mcs: MCS, *, design: EstimatorDesign = EstimatorDesign(),
                 fp: FixedPoint = FixedPoint(), early_stop: bool = True, num_threads: int = 0,
                 genie: bool = True, **link_kw):
        link_kw = dict(link_kw, estimator="lmmse", compile=False)
        self.ref = OfdmLink(num, mcs, **link_kw)
        self.num, self.mcs, self.plan, self.device = num, mcs, self.ref.plan, self.ref.device
        self.design, self.fp = design, fp
        if self.ref.phase_track not in (None, "linear"):
            raise ValueError("the product receiver tracks phase with phase_track='linear' only")
        mask = self.ref.rg.pilot_pattern.mask[0, 0].cpu().numpy()
        pilots, perm = frame_tables(num, mcs, mask)
        p = self.plan
        self.genie = (ZigModem(num, mcs, k=p.k, n=p.n, num_codewords=p.num_codewords, pilot_mask=mask, pilots=pilots,
                               perm=perm, num_iter=fp.num_iter, early_stop=early_stop,
                               phase_track=self.ref.phase_track, num_threads=num_threads) if genie else None)
        kw = dict(k=p.k, n=p.n, num_codewords=p.num_codewords, pilot_mask=mask, fp=fp, taps=design.taps,
                  phase_track=self.ref.phase_track == "linear", track_gain=design.track_gain,
                  early_stop=early_stop, num_threads=num_threads)
        self.fixed = FxModem(num, mcs, fixed=True, **kw)
        self.float = FxModem(num, mcs, fixed=False, **kw)
        self.a = qam_norm(mcs.bits_per_symbol)
        self._no = self._design_snr = None

    def set_noise(self, no: float):
        if self._no == no:
            return
        if self.genie is not None:
            w, err_var = self.ref.estimator.filters(no)
            wb = self.ref.block_estimator.filters(no)[0].cpu().numpy() if self.ref.phase_track else None
            self.genie.set_filter(w.cpu().numpy(), err_var.cpu().numpy(), wb)
        snr = self.design.design_snr_db(-10 * math.log10(no))
        if snr != self._design_snr:
            filt = design_filters(self.num, self.design, snr)
            self.fixed.set_filter(filt)
            self.float.set_filter(filt)
            self._design_snr = snr
        self._no = no

    def llr_scale(self, no: float, gain: np.ndarray) -> np.ndarray:
        """a / (N0 LSB) per frame, N0 in ADC units (the AGC gain squared times
        the channel's), capped (FixedPoint.llr_cap); in hardware the control
        processor computes it from the noise estimate."""
        fp = self.fp
        scale = self.a / (no * gain.astype(np.float64) ** 2 * fp.llr_step_nat)
        return np.minimum(scale, fp.llr_cap / (self.a * fp.agc_power))

    def transmit(self, b_np: np.ndarray) -> torch.Tensor:
        """Fixed-point TX, scaled to unit energy per RE: [batch, 1, 1, samples]."""
        x_q = self.fixed.tx(b_np)
        x = (x_q[..., 0].astype(np.float32) + 1j * x_q[..., 1].astype(np.float32)) / self.fp.tx_scale
        return torch.as_tensor(x.astype(np.complex64), device=self.device).reshape(b_np.shape[0], 1, 1, -1)

    def run(self, batch: int, esno_db: float) -> dict:
        ref, p = self.ref, self.plan
        no = 10 ** (-esno_db / 10)
        self.set_noise(no)
        b = ref.source([batch, p.num_codewords, p.k])
        bb = b.to(torch.uint8).cpu().numpy()
        y, _ = ref.propagate(self.transmit(bb), torch.tensor(no, device=self.device))
        y_np = y.reshape(batch, -1).cpu().numpy()
        ys, yq, g = agc_adc(y_np, self.fp)
        scale = self.llr_scale(no, g)

        def errors(b_hat, prefix=""):
            r = count_errors(b, torch.as_tensor(b_hat, device=b.device).to(b.dtype))
            return r if not prefix else {f"{prefix}{k}": v for k, v in r.items()
                                         if k not in ("frames", "codewords", "bits")}

        t0 = time.perf_counter()
        b_fix = self.fixed.rx(yq, scale, ref.l_min)
        t1 = time.perf_counter()
        b_flt = self.float.rx(ys, scale, ref.l_min)
        t2 = time.perf_counter()
        res = {"fixed_rx_s": t1 - t0, "float_rx_s": t2 - t1, **errors(b_fix), **errors(b_flt, "float_"),
               "disagree_codewords": int(((b_fix != bb).any(-1) != (b_flt != bb).any(-1)).sum())}
        if self.genie is not None:
            t0 = time.perf_counter()
            b_gen = self.genie.rx(y_np, no, ref.l_min)
            res["genie_rx_s"] = time.perf_counter() - t0
            res.update(errors(b_gen, "genie_"))
        return res


def ldpc_decode(llr: np.ndarray, k: int, algo: str = "oms-fixed", *, llr_bits: int = 6, app_bits: int = 8,
                msg_bits: int = 6, offset: float = 1.0, num_iter: int = 20, early_stop: bool = True,
                num_threads: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """llr [..., n] logits log p(1) / p(0): nats for "bp", LSB units for the
    min-sum decoders (integers for "oms-fixed"). Returns (bits [..., k],
    iterations per codeword [...])."""
    llr = np.ascontiguousarray(llr, dtype=np.float32)
    n = llr.shape[-1]
    flat = llr.reshape(-1, n)
    out = np.empty((flat.shape[0], k), np.uint8)
    iters = np.empty(flat.shape[0], np.uint32)
    p = DecoderParams(llr_bits, app_bits, msg_bits, offset, num_iter, int(early_stop))
    rc = _bind().nphy_fx_ldpc_decode(k, n, _ptr(flat, _f32p), flat.shape[0], ALGOS[algo], ctypes.byref(p),
                                     num_threads, _ptr(out, _u8p), _ptr(iters, _u32p))
    if rc < 0:
        raise ValueError(f"unsupported code k={k} n={n} or algo {algo}")
    return out.reshape(*llr.shape[:-1], k), iters.reshape(llr.shape[:-1])
