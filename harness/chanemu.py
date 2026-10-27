"""The AWGN channel emulator of the board loopback (M8): rtl/chan_emu and
its NumPy twin, bit exact.

The board has no radio: the PL takes the fixed-point TX samples, scales
them, adds Gaussian noise and quantises to the 12-bit ADC, then the RTL
receiver decodes. The noise is counter-based, so a frame's noise is a
function of (seed, sample index) and the twin reproduces the board's
error counts exactly:

  base = h(seed), u1 = h(base ^ 2i), u2 = h(base ^ (2i + 1))   h = lowbias32
  r = R[min(clz u1, 23), 8 bits after the leading one]          sqrt(-2 ln u1), Q11
  (c, s) = cossin(u2 >> 16)                                     the M4 CORDIC, Q15
  n = r (c + j s) >> 6                                          Q20 per rail sigma
  y = sat12(rshr(gx x + gn n, 28))

R holds sqrt(E[-2 ln u]) over each of its 24 x 256 bins of u, so the
noise power is exact; the radius takes 6144 values up to 5.8 sigma. gx and
gn set the AGC gain and the noise level: gx = g 2^28 / tx_scale,
gn = g sqrt(N0 / 2) 2^8, with g the gain an ideal AGC would apply on
average (FixedPoint.agc_power over signal plus noise power). The L2
reference (FxOfdmLink) differs only in its noise source and in measuring
the AGC gain per frame; e018 measures what that costs.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import twin

M32 = 0xFFFFFFFF
R_ROWS, R_BITS = 24, 8
SHIFT = 28


def lowbias32(x):
    """Chris Wellons' lowbias32 integer hash, on uint32 arrays."""
    x = np.asarray(x, dtype=np.uint64) & M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    x ^= x >> 16
    return x


@lru_cache(maxsize=1)
def radius_table() -> np.ndarray:
    """R[lz * 256 + m]: round(2^11 sqrt(E[-2 ln u])) over u in the bin."""
    out = np.empty(R_ROWS << R_BITS, np.int64)
    for lz in range(R_ROWS):
        for m in range(1 << R_BITS):
            lo = (256 + m) / 256 * 2.0 ** -(lz + 1)
            hi = (257 + m) / 256 * 2.0 ** -(lz + 1)
            # E[-2 ln u] for u uniform in [lo, hi)
            e = -2 * ((hi * math.log(hi) - hi) - (lo * math.log(lo) - lo)) / (hi - lo)
            out[(lz << R_BITS) | m] = math.floor(math.sqrt(e) * 2048 + 0.5)
    return out


def clz32(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.uint64)
    lz = np.full(u.shape, 32, np.int64)
    for b in range(31, -1, -1):
        lz = np.where((lz == 32) & ((u >> b) & 1 == 1), 31 - b, lz)
    return lz


def noise(seed: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """The emulator's noise of one frame: [length] int64 re, im in Q20 of
    the per-rail standard deviation."""
    base = int(lowbias32(seed))
    i = np.arange(length, dtype=np.uint64)
    u1 = lowbias32(base ^ (2 * i))
    u2 = lowbias32(base ^ (2 * i + 1))
    lz = np.minimum(clz32(u1), R_ROWS - 1)
    m = (((u1 << lz.astype(np.uint64)) & M32) >> 23) & 0xFF
    r = radius_table()[(lz << R_BITS) | m.astype(np.int64)]
    c, s = twin.cossin_vec((u2 >> 16).astype(np.int64))
    return (r * c) >> 6, (r * s) >> 6


def apply(x: np.ndarray, seed: int, gx: int, gn: int, adc_bits: int = 12) -> np.ndarray:
    """TX samples [length, 2] -> the emulator's ADC samples [length, 2] int16."""
    x = np.asarray(x, np.int64)
    nr, ni = noise(seed, x.shape[0])
    hi = 1 << (adc_bits - 1)
    out = np.stack([gx * x[:, 0] + gn * nr, gx * x[:, 1] + gn * ni], -1)
    return np.clip(twin.rshr(out, SHIFT), -hi, hi - 1).astype(np.int16)


def gains(no: float, signal_power: float, fp) -> tuple[int, int, float]:
    """(gx, gn, g) for noise N0 per complex sample (unit signal energy per
    RE) and the TX signal's mean |x / tx_scale|^2, fp a FixedPoint."""
    g = math.sqrt(fp.agc_power / (signal_power + no))
    gx = math.floor(g * 2.0 ** SHIFT / fp.tx_scale + 0.5)
    gn = math.floor(g * math.sqrt(no / 2) * 2.0 ** (SHIFT + 6 - 26) + 0.5)
    return gx, gn, g


def seed_of(run_seed: int, frame: int) -> int:
    """The noise seed of frame `frame` of a run (rtl/loopback)."""
    return (run_seed + frame) & M32
