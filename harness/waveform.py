"""Waveform definition: numerology (OFDM frame geometry) and the MCS table.

Everything downstream (Sionna chain, Zig modem, RTL) derives its sizes from
these two tables so a hypothesis about the waveform is a change here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Numerology:
    name: str
    bandwidth_hz: float           # sample rate at baseband, FFT spans it
    fft_size: int
    cp_len: int                   # cyclic prefix in samples
    num_guard: tuple[int, int]    # unused subcarriers at the band edges (low, high)
    dc_null: bool
    num_ofdm_symbols: int         # OFDM symbols per frame
    pilot_ofdm_symbols: tuple[int, ...]   # Kronecker (block) pilot symbols

    @property
    def subcarrier_spacing_hz(self) -> float:
        return self.bandwidth_hz / self.fft_size

    @property
    def symbol_duration_s(self) -> float:
        return (self.fft_size + self.cp_len) / self.bandwidth_hz

    @property
    def frame_duration_s(self) -> float:
        return self.num_ofdm_symbols * self.symbol_duration_s

    @property
    def num_effective_subcarriers(self) -> int:
        return self.fft_size - sum(self.num_guard) - (1 if self.dc_null else 0)

    @property
    def num_data_res(self) -> int:
        """Data resource elements per frame (pilot symbols carry no data)."""
        return self.num_effective_subcarriers * (self.num_ofdm_symbols - len(self.pilot_ofdm_symbols))

    @property
    def occupied_bandwidth_hz(self) -> float:
        return (self.num_effective_subcarriers + (1 if self.dc_null else 0)) * self.subcarrier_spacing_hz

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(subcarrier_spacing_hz=self.subcarrier_spacing_hz,
                 symbol_duration_us=self.symbol_duration_s * 1e6,
                 frame_duration_us=self.frame_duration_s * 1e6,
                 num_effective_subcarriers=self.num_effective_subcarriers,
                 num_data_res=self.num_data_res)
        return d


@dataclass(frozen=True)
class MCS:
    name: str
    bits_per_symbol: int          # 2 = QPSK, 4 = 16-QAM, 6 = 64-QAM
    code_rate: float
    constellation: str = "qam"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CodewordPlan:
    """How one frame's data REs are split into LDPC codewords."""
    num_codewords: int
    n: int                        # coded bits per codeword
    k: int                        # information bits per codeword
    padding_bits: int             # frame bits not used by any codeword

    @property
    def info_bits_per_frame(self) -> int:
        return self.num_codewords * self.k


# 10 MHz baseline, roughly 802.11-shaped so the curves can be sanity-checked
# against published 802.11n LDPC results: 256-point FFT, 12.5 % cyclic prefix,
# 200 used subcarriers (199 effective with the DC null), 14-symbol frame with
# block pilots on two symbols. Comb pilots are a later hypothesis.
NUM_10MHZ = Numerology(
    name="n10",
    bandwidth_hz=10e6,
    fft_size=256,
    cp_len=32,
    num_guard=(28, 28),
    dc_null=True,
    num_ofdm_symbols=14,
    pilot_ofdm_symbols=(2, 11),
)

# Single-stream MCS table, 802.11n MCS 1-7 shaped. BPSK omitted for now.
MCS_TABLE: tuple[MCS, ...] = (
    MCS("qpsk-1/2", 2, 1 / 2),
    MCS("qpsk-3/4", 2, 3 / 4),
    MCS("16qam-1/2", 4, 1 / 2),
    MCS("16qam-3/4", 4, 3 / 4),
    MCS("64qam-2/3", 6, 2 / 3),
    MCS("64qam-3/4", 6, 3 / 4),
    MCS("64qam-5/6", 6, 5 / 6),
)

LDPC5G_MAX_K = 8448


def plan_codewords(num: Numerology, mcs: MCS, max_k: int = LDPC5G_MAX_K) -> CodewordPlan:
    """Split the frame's coded bits into the fewest equal 5G LDPC codewords
    with k <= max_k. Leftover bits (from making n divisible) are padding."""
    frame_bits = num.num_data_res * mcs.bits_per_symbol
    num_cw = max(1, math.ceil(frame_bits * mcs.code_rate / max_k))
    n = frame_bits // num_cw
    n -= n % 2                      # keep n even, harmless and avoids odd corner cases
    k = int(round(n * mcs.code_rate))
    return CodewordPlan(num_cw, n, k, frame_bits - num_cw * n)


def phy_rate_bps(num: Numerology, mcs: MCS) -> float:
    plan = plan_codewords(num, mcs)
    return plan.info_bits_per_frame / num.frame_duration_s


def esno_to_ebno_db(esno_db: float, mcs: MCS) -> float:
    return esno_db - 10 * math.log10(mcs.bits_per_symbol * mcs.code_rate)
