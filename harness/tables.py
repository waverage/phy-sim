"""Generated waveform tables (M4): the pilot sequence and the frame bit
interleaver, defined by small generators instead of Python RNG output, so
the Zig modem (modem/src/tables.zig) and the RTL rebuild them from a seed.

Pilots: QPSK from the TS 38.211 5.2.1 length-31 Gold sequence, two bits
per pilot in pilot order (symbol-major, as the pilot mask): pilot p is
((1 - 2 c(2p)) + j (1 - 2 c(2p + 1))) / sqrt(2).

Interleaver: Fisher-Yates driven by xorshift32 (13, 17, 5). Statistically
the same as M2's random permutation; in hardware it is a ROM of
frame_bits addresses (14,328 x 14 bits for 64-QAM), filled from the
generator.
"""
from __future__ import annotations

import numpy as np

PILOT_SEED = 0x4E5354      # c_init, "NST"
PERM_SEED = 0x4E535450     # "NSTP"
_NC = 1600


def gold(c_init: int, length: int) -> np.ndarray:
    """c(0..length-1) of TS 38.211 5.2.1."""
    n = length + _NC
    x1 = np.zeros(n + 31, np.uint8)
    x2 = np.zeros(n + 31, np.uint8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for i in range(n):
        x1[i + 31] = x1[i + 3] ^ x1[i]
        x2[i + 31] = x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]
    return (x1[_NC:_NC + length] ^ x2[_NC:_NC + length]).astype(np.uint8)


def pilot_signs(num_pilots: int, seed: int = PILOT_SEED) -> np.ndarray:
    """[num_pilots, 2] int8 (a, b) in {-1, +1}: pilot = (a + j b) / sqrt(2)."""
    c = gold(seed, 2 * num_pilots).reshape(num_pilots, 2).astype(np.int8)
    return 1 - 2 * c


def pilots(num_pilots: int, seed: int = PILOT_SEED) -> np.ndarray:
    s = pilot_signs(num_pilots, seed).astype(np.float64)
    return ((s[:, 0] + 1j * s[:, 1]) / np.sqrt(2)).astype(np.complex64)


def interleaver(frame_bits: int, seed: int = PERM_SEED) -> np.ndarray:
    """Permutation: transmitted bit i is coded bit perm[i]."""
    perm = np.arange(frame_bits, dtype=np.uint32)
    s = seed & 0xFFFFFFFF
    for i in range(frame_bits - 1, 0, -1):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        j = (s * (i + 1)) >> 32
        perm[i], perm[j] = perm[j], perm[i]
    return perm
