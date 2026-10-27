"""Configuration words of the RTL blocks (rtl/), from the twins' structures.

The receiver's tables and registers are written by the control processor
through a (cfg_addr, cfg_data) port; these functions give the (address,
data) pairs, for the cocotb tests and for the ARM program on the board
(tools/gen_arm.py). Each docstring names the RTL module it configures.
"""
from __future__ import annotations

from . import twin

LDPC_P = 16          # rtl/ldpc_dec lanes
LDPC_AW = 10         # APP address bits
LDPC_EDGE_BASE = 512


def ldpc_cfg(code: twin.LdpcCode, *, num_iter: int = 20, offset: int = 2, early_stop: bool = True,
             base: int = 0) -> list[tuple[int, int]]:
    """rtl/ldpc_dec/ldpc_dec.v: registers 0..12 and the edge table."""
    z = code.z
    if z % LDPC_P:
        raise ValueError(f"lifting {z} is not a multiple of {LDPC_P}")
    q = z // LDPC_P
    edges = [(r_i, e, c, s) for r_i, ents in enumerate(code.rows) for e, (c, s) in enumerate(ents)]
    if code.dec_cols * q > 1 << LDPC_AW or len(edges) * q > 1 << 12 or len(edges) > 1 << 9:
        raise ValueError("code too large for the decoder's memories")
    if max(len(r) for r in code.rows) > 31 or not 1 <= num_iter <= 31:
        raise ValueError("row degree or iteration count out of range")
    regs = [z, q, code.k, code.n, code.k - 2 * z, code.k_b * q, code.dec_cols * q, len(edges), num_iter, offset,
            int(early_stop), code.k_ldpc, 2 * q]
    out = [(base + i, v) for i, v in enumerate(regs)]
    for i, (r_i, e, c, s) in enumerate(edges):
        last = int(e == len(code.rows[r_i]) - 1)
        word = (last << (LDPC_AW + 9)) | ((c * q) << 9) | ((s % q) << 4) | (s // q)
        out.append((base + LDPC_EDGE_BASE + i, word))
    return out


RX_LDPC, RX_AT, RX_START, RX_LV, RX_SIGNS, RX_R, RX_TAPS, RX_PERM = (0x0400, 0x0800, 0x0900, 0x0A00, 0x0C00, 0x1000,
                                                                     0x2000, 0x4000)
RX_MANT, RX_SHIFT = 19, 20


def _u(v: int, bits: int) -> int:
    return int(v) & ((1 << bits) - 1)


def rx_cfg(tw: twin.TwinModem, l_min: int = 0) -> list[tuple[int, int]]:
    """rtl/rx/rx_frame.v: registers and tables of one numerology, MCS and
    estimator design (tw.set_filter done). The LLR scale is per frame:
    rx_scale_cfg."""
    S, F, np_ = tw.S, tw.F, len(tw.psym)
    L = tw.bq.shape[1]
    if not 2 <= np_ <= 3 or L > 32 or S > 14 or F > 255 or len(tw.perm) > 1 << 14:
        raise ValueError("frame outside rx_frame's limits")
    psym = list(tw.psym) + [15] * (3 - np_)
    regs = {0: S, 1: F, 2: np_, 3: psym[0], 4: psym[1], 5: psym[2], 6: sum(tw.psym), 7: L,
            8: tw.num.num_guard[0], 9: tw.num.num_guard[1], 10: int(tw.num.dc_null), 11: _u(l_min, 8), 12: tw.B,
            13: tw.num_cw, 14: tw.n, 16: tw.a_q, 17: int(tw.phase_track), 18: tw.track_gain, 21: tw.cp}
    out = list(regs.items())
    out += ldpc_cfg(tw.code, num_iter=tw.dec["num_iter"], offset=tw.dec["offset"], early_stop=tw.dec["early_stop"],
                    base=RX_LDPC)
    out += [(RX_AT + 4 * s + t, _u(tw.aq[s, t], 16)) for s in range(S) for t in range(np_)]
    out += [(RX_START + f, int(tw.start[f])) for f in range(F)]
    out += [(RX_LV + i, _u(v, 4)) for i, v in enumerate(tw.levels)]
    out += [(RX_SIGNS + t * F + f, (int(tw.signs[t, f, 0] < 0) << 1) | int(tw.signs[t, f, 1] < 0))
            for t in range(np_) for f in range(F)]
    out += [(RX_R + d, int(v)) for d, v in enumerate(tw.rq)]
    out += [(RX_TAPS + 32 * f + j, (_u(tw.bq[f, j, 0], 16) << 16) | _u(tw.bq[f, j, 1], 16))
            for f in range(F) for j in range(L)]
    out += [(RX_PERM + i, int(p)) for i, p in enumerate(tw.perm)]
    return out


def rx_scale_cfg(scale: float) -> list[tuple[int, int]]:
    """The per-frame LLR scale of rx_frame: mantissa and shift
    (twin.llr_mantissa)."""
    mant, shift = twin.llr_mantissa(scale)
    return [(RX_MANT, mant), (RX_SHIFT, shift)]


def ldpc_cycles(code: twin.LdpcCode, iterations: int, early_stop: bool = True) -> int:
    """Approximate clocks of one codeword in rtl/ldpc_dec (load, iterations,
    syndrome checks, output)."""
    q = code.z // LDPC_P
    edges = sum(len(r) for r in code.rows)
    per_iter = edges * q + sum(len(r) + 5 for r in code.rows)
    synd = (iterations - 1 if early_stop else 0) * per_iter
    return code.dec_cols * q + code.n + iterations * per_iter + synd + code.k
