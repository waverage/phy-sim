"""The Zig modem (modem/, L2) from Python, through its C ABI.

ZigModem wraps one configured modem (TX and RX for a numerology and MCS).
ZigOfdmLink runs it inside the Sionna simulation: Zig transmitter, then the
same channel, impairments and noise as OfdmLink, then the Zig receiver, and
optionally Sionna's receiver on the very same samples, so the two receivers
are compared on common random numbers.

The modem takes everything that defines the waveform from Python: the
numerology and codeword plan (harness/waveform.py), the pilot pattern and
the frame interleaver (OfdmLink), and per noise level the 2D estimator
filter (GenieLmmseEstimator.filters).
"""
from __future__ import annotations

import ctypes
import pathlib
import sys
import time

import numpy as np
import torch

from .chain import OfdmLink, count_errors
from .waveform import MCS, Numerology

LIB = (pathlib.Path(__file__).resolve().parents[1] / "modem" / "zig-out" / "lib"
       / ("libnestuaphy.dylib" if sys.platform == "darwin" else "libnestuaphy.so"))

_u8p = ctypes.POINTER(ctypes.c_uint8)
_f32p = ctypes.POINTER(ctypes.c_float)
_u32p = ctypes.POINTER(ctypes.c_uint32)


class Config(ctypes.Structure):
    """modem.Config in modem/src/modem.zig."""
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "fft_size", "cp_len", "guard_lo", "guard_hi", "dc_null", "num_symbols", "bits_per_symbol",
        "num_codewords", "n", "k", "num_iter", "early_stop", "phase_track", "num_threads")] + [
        ("pilot_mask", _u8p), ("pilots", _f32p), ("perm", _u32p)]


_lib = None


def lib() -> ctypes.CDLL:
    """The shared library, built by `make modem`."""
    global _lib
    if _lib is None:
        if not LIB.exists():
            raise FileNotFoundError(f"{LIB} missing, build it with `make modem`")
        _lib = ctypes.CDLL(str(LIB))
        _lib.nphy_create.restype = ctypes.c_void_p
        _lib.nphy_create.argtypes = [ctypes.POINTER(Config)]
        _lib.nphy_destroy.argtypes = [ctypes.c_void_p]
        _lib.nphy_num_samples.restype = ctypes.c_uint32
        _lib.nphy_num_samples.argtypes = [ctypes.c_void_p]
        _lib.nphy_set_filter.restype = ctypes.c_int32
        _lib.nphy_set_filter.argtypes = [ctypes.c_void_p, _f32p, _f32p, _f32p]
        _lib.nphy_tx.argtypes = [ctypes.c_void_p, _u8p, ctypes.c_uint32, _f32p]
        _lib.nphy_rx.restype = ctypes.c_int32
        _lib.nphy_rx.argtypes = [ctypes.c_void_p, _f32p, ctypes.c_uint32, ctypes.c_int32, ctypes.c_float,
                                 ctypes.c_uint32, _u8p, _f32p]
        _lib.nphy_decoder_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64 * 2), ctypes.c_uint32]
        _lib.nphy_ldpc_info.restype = ctypes.c_int32
        _lib.nphy_ldpc_info.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32 * 6)]
        _lib.nphy_ldpc_encode.restype = ctypes.c_int32
        _lib.nphy_ldpc_encode.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _u8p, ctypes.c_uint32, _u8p]
        _lib.nphy_ldpc_decode.restype = ctypes.c_int64
        _lib.nphy_ldpc_decode.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _f32p, ctypes.c_uint32,
                                          ctypes.c_uint32, ctypes.c_uint32, _u8p]
        _lib.nphy_qam_map.restype = ctypes.c_int32
        _lib.nphy_qam_map.argtypes = [ctypes.c_uint32, _u8p, ctypes.c_uint32, _f32p]
        _lib.nphy_qam_demap.restype = ctypes.c_int32
        _lib.nphy_qam_demap.argtypes = [ctypes.c_uint32, _f32p, _f32p, ctypes.c_uint32, _f32p]
    return _lib


def _ptr(a: np.ndarray, t):
    assert a.flags.c_contiguous
    return a.ctypes.data_as(t)


def _f32(a) -> np.ndarray:
    """complex64 or float32 array as a contiguous float32 view."""
    a = np.ascontiguousarray(a)
    return a.view(np.float32) if np.iscomplexobj(a) else a.astype(np.float32, copy=False)


# ---- single blocks -------------------------------------------------------------

def ldpc_info(k: int, n: int) -> dict:
    out = (ctypes.c_uint32 * 6)()
    if lib().nphy_ldpc_info(k, n, ctypes.byref(out)) != 0:
        raise ValueError(f"unsupported code k={k} n={n}")
    return dict(zip(("bg", "z", "k_filler", "n_ldpc", "dec_rows", "dec_cols"), list(out)))


def ldpc_encode(bits: np.ndarray, n: int) -> np.ndarray:
    bits = np.ascontiguousarray(bits, dtype=np.uint8)
    k = bits.shape[-1]
    flat = bits.reshape(-1, k)
    out = np.empty((flat.shape[0], n), np.uint8)
    if lib().nphy_ldpc_encode(k, n, _ptr(flat, _u8p), flat.shape[0], _ptr(out, _u8p)) != 0:
        raise ValueError(f"unsupported code k={k} n={n}")
    return out.reshape(*bits.shape[:-1], n)


def ldpc_decode(llr: np.ndarray, k: int, num_iter: int = 20, early_stop: bool = False) -> tuple[np.ndarray, int]:
    """llr [..., n] (log p(1) / p(0)) -> (bits [..., k], total iterations)."""
    llr = np.ascontiguousarray(llr, dtype=np.float32)
    n = llr.shape[-1]
    flat = llr.reshape(-1, n)
    out = np.empty((flat.shape[0], k), np.uint8)
    it = lib().nphy_ldpc_decode(k, n, _ptr(flat, _f32p), flat.shape[0], num_iter, int(early_stop), _ptr(out, _u8p))
    if it < 0:
        raise ValueError(f"unsupported code k={k} n={n}")
    return out.reshape(*llr.shape[:-1], k), int(it)


def qam_map(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    bits = np.ascontiguousarray(bits, dtype=np.uint8).reshape(-1)
    count = bits.size // bits_per_symbol
    out = np.empty(count, np.complex64)
    if lib().nphy_qam_map(bits_per_symbol, _ptr(bits, _u8p), count, _ptr(out.view(np.float32), _f32p)) != 0:
        raise ValueError(bits_per_symbol)
    return out


def qam_demap(y: np.ndarray, no: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    y = np.ascontiguousarray(y, dtype=np.complex64).reshape(-1)
    no = np.ascontiguousarray(np.broadcast_to(no, y.shape), dtype=np.float32)
    out = np.empty(y.size * bits_per_symbol, np.float32)
    if lib().nphy_qam_demap(bits_per_symbol, _ptr(y.view(np.float32), _f32p), _ptr(no, _f32p), y.size,
                            _ptr(out, _f32p)) != 0:
        raise ValueError(bits_per_symbol)
    return out.reshape(-1, bits_per_symbol)


# ---- frame modem ------------------------------------------------------------------

class ZigModem:
    def __init__(self, num: Numerology, mcs: MCS, *, k: int, n: int, num_codewords: int,
                 pilot_mask: np.ndarray, pilots: np.ndarray, perm: np.ndarray, num_iter: int = 20,
                 early_stop: bool = True, phase_track: str | None = None, num_threads: int = 0):
        if phase_track not in (None, "linear"):
            raise ValueError("the Zig receiver tracks phase with phase_track='linear' only")
        self.num, self.mcs, self.k, self.n, self.num_codewords = num, mcs, k, n, num_codewords
        self.phase_track = phase_track
        # the config points into these; keep them alive with the modem
        self._mask = np.ascontiguousarray(pilot_mask, dtype=np.uint8).reshape(-1)
        self._pilots = np.ascontiguousarray(pilots, dtype=np.complex64).reshape(-1)
        self._perm = np.ascontiguousarray(perm, dtype=np.uint32).reshape(-1)
        cfg = Config(
            fft_size=num.fft_size, cp_len=num.cp_len, guard_lo=num.num_guard[0], guard_hi=num.num_guard[1],
            dc_null=int(num.dc_null), num_symbols=num.num_ofdm_symbols, bits_per_symbol=mcs.bits_per_symbol,
            num_codewords=num_codewords, n=n, k=k, num_iter=num_iter, early_stop=int(early_stop),
            phase_track=int(phase_track == "linear"), num_threads=num_threads,
            pilot_mask=_ptr(self._mask, _u8p), pilots=_ptr(self._pilots.view(np.float32), _f32p),
            perm=_ptr(self._perm, _u32p))
        self._lib = lib()
        self._h = self._lib.nphy_create(ctypes.byref(cfg))
        if not self._h:
            raise ValueError("nphy_create rejected the configuration")
        self.num_samples = self._lib.nphy_num_samples(self._h)

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.nphy_destroy(self._h)
            self._h = None

    def set_filter(self, w: np.ndarray, err_var: np.ndarray, w_block: np.ndarray | None = None):
        """w [S*F, Np] complex, err_var [S*F] (or [S, F]), w_block [S*F, Npb]."""
        w = np.ascontiguousarray(w, dtype=np.complex64)
        ev = np.ascontiguousarray(err_var, dtype=np.float32).reshape(-1)
        wb = None if w_block is None else np.ascontiguousarray(w_block, dtype=np.complex64)
        wb_p = None if wb is None else _ptr(wb.view(np.float32), _f32p)
        if self._lib.nphy_set_filter(self._h, _ptr(w.view(np.float32), _f32p), _ptr(ev, _f32p), wb_p) != 0:
            raise ValueError("nphy_set_filter failed (phase tracking needs w_block)")

    def tx(self, bits: np.ndarray) -> np.ndarray:
        """bits [batch, num_codewords, k] -> samples [batch, num_samples] complex64."""
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        batch = bits.shape[0]
        out = np.empty((batch, self.num_samples), np.complex64)
        self._lib.nphy_tx(self._h, _ptr(bits, _u8p), batch, _ptr(out.view(np.float32), _f32p))
        return out

    def rx(self, y: np.ndarray, no: float, l_min: int = 0, return_llr: bool = False):
        """y [batch, len] complex64 -> bits [batch, num_codewords, k]
        (and LLRs [batch, num_codewords, n])."""
        y = np.ascontiguousarray(y, dtype=np.complex64)
        batch, length = y.shape
        out = np.empty((batch, self.num_codewords, self.k), np.uint8)
        llr = np.empty((batch, self.num_codewords, self.n), np.float32) if return_llr else None
        rc = self._lib.nphy_rx(self._h, _ptr(y.view(np.float32), _f32p), length, l_min, float(no), batch,
                               _ptr(out, _u8p), None if llr is None else _ptr(llr, _f32p))
        if rc != 0:
            raise RuntimeError("nphy_rx failed (no filter set, or y too short)")
        return (out, llr) if return_llr else out

    def decoder_stats(self, reset: bool = True) -> tuple[int, int]:
        """(iterations, codewords) decoded since the last reset."""
        out = (ctypes.c_uint64 * 2)()
        self._lib.nphy_decoder_stats(self._h, ctypes.byref(out), int(reset))
        return int(out[0]), int(out[1])


def zig_modem_for(link: OfdmLink, **kw) -> ZigModem:
    """A ZigModem with the waveform of an OfdmLink: its numerology, MCS,
    codeword plan, pilot pattern and frame interleaver."""
    pp = link.rg.pilot_pattern
    return ZigModem(link.num, link.mcs, k=link.plan.k, n=link.plan.n, num_codewords=link.plan.num_codewords,
                    pilot_mask=pp.mask[0, 0].cpu().numpy(), pilots=pp.pilots[0, 0].cpu().numpy(),
                    perm=link.perm.cpu().numpy(), **kw)


class ZigOfdmLink:
    """OfdmLink with the Zig transmitter and receiver around Sionna's
    channel. Takes OfdmLink's keyword arguments (estimator is always
    "lmmse": the Zig receiver applies GenieLmmseEstimator's filter).

    compare=True also runs Sionna's receiver on the same received samples;
    run() then returns the Zig counts plus "sionna_*" counts and
    "disagree_codewords", the codewords exactly one of the two got wrong."""

    def __init__(self, num: Numerology, mcs: MCS, *, compare: bool = False, early_stop: bool = True,
                 num_threads: int = 0, **link_kw):
        link_kw = dict(link_kw, estimator="lmmse")
        self.ref = OfdmLink(num, mcs, **link_kw)
        self.compare = compare
        self.num, self.mcs, self.plan, self.device = num, mcs, self.ref.plan, self.ref.device
        self.zig = zig_modem_for(self.ref, num_iter=link_kw.get("num_iter", 20), early_stop=early_stop,
                                 phase_track=self.ref.phase_track, num_threads=num_threads)
        self._no = None

    def set_noise(self, no: float):
        if self._no == no:
            return
        w, err_var = self.ref.estimator.filters(no)
        wb = self.ref.block_estimator.filters(no)[0].cpu().numpy() if self.ref.phase_track else None
        self.zig.set_filter(w.cpu().numpy(), err_var.cpu().numpy(), wb)
        self._no = no

    def run(self, batch: int, esno_db: float) -> dict:
        """Also reports zig_rx_s (and sionna_rx_s): receiver wall time."""
        ref, p = self.ref, self.plan
        no = torch.tensor(10 ** (-esno_db / 10), device=self.device)
        self.set_noise(float(no))
        b = ref.source([batch, p.num_codewords, p.k])
        x_t = torch.as_tensor(self.zig.tx(b.to(torch.uint8).cpu().numpy()), device=self.device)
        y, h_t = ref.propagate(x_t.reshape(batch, 1, 1, -1), no)
        y_np = y.reshape(batch, -1).cpu().numpy()
        t0 = time.perf_counter()
        b_zig = self.zig.rx(y_np, float(no), ref.l_min)
        t_zig = time.perf_counter() - t0
        b_zig = torch.as_tensor(b_zig, device=self.device).to(b.dtype)
        res = count_errors(b, b_zig)
        res["zig_rx_s"] = t_zig
        if self.compare:
            t0 = time.perf_counter()
            b_ref = ref.receive(y, h_t, no)
            res["disagree_codewords"] = int(((b_zig != b).any(-1) != (b_ref != b).any(-1)).sum())
            res["sionna_rx_s"] = time.perf_counter() - t0      # the count above synchronises
            r = count_errors(b, b_ref)
            res.update({f"sionna_{k}": v for k, v in r.items() if k not in ("frames", "codewords", "bits")})
        return res
