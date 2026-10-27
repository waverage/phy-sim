"""Configuration words of the RTL blocks (rtl/), from the twins' structures.

The receiver's tables and registers are written by the control processor
through a (cfg_addr, cfg_data) port; these functions give the (address,
data) pairs, for the cocotb tests and for the ARM program on the board
(tools/gen_arm.py). Each docstring names the RTL module it configures.
"""
from __future__ import annotations

import numpy as np

from . import tables, twin

LDPC_P = 16          # rtl/ldpc_dec lanes
LDPC_AW = 10         # APP address bits
LDPC_EDGE_BASE = 512
LDPC_LLR_AW = 10     # rtl/rx/rx_frame LLR bank words


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
    regs = [z, q, code.k, code.k_ldpc, ldpc_vend(code), 2 * z, code.dec_cols * q, ldpc_out_words(code), len(edges),
            num_iter, offset, int(early_stop), 2 * q]
    out = [(base + i, v) for i, v in enumerate(regs)]
    for i, (r_i, e, c, s) in enumerate(edges):
        last = int(e == len(code.rows[r_i]) - 1)
        word = (last << (LDPC_AW + 9)) | ((c * q) << 9) | ((s % q) << 4) | (s // q)
        out.append((base + LDPC_EDGE_BASE + i, word))
    return out


def ldpc_vend(code: twin.LdpcCode) -> int:
    """One past the last transmitted variable: k_ldpc + the parity LLRs."""
    return code.k_ldpc + code.n - (code.k - 2 * code.z)


def ldpc_llr_words(code: twin.LdpcCode) -> int:
    """Words of one codeword in rtl/ldpc_dec's LLR memory (columns 2 ..)."""
    return (code.dec_cols - 2) * (code.z // LDPC_P)


def ldpc_out_words(code: twin.LdpcCode) -> int:
    """Words of rtl/ldpc_dec's output: the columns holding the k bits."""
    return -(-code.k // code.z) * (code.z // LDPC_P)


def ldpc_llr_slot(code: twin.LdpcCode) -> tuple[np.ndarray, np.ndarray]:
    """Where rtl/ldpc_dec's LLR memory holds LLR t of a codeword: (word,
    lane), word (c - 2) Q + w and lane j for the variable v = c z + j Q + w
    that LLR t feeds (twin.ldpc_decode's mapping)."""
    z, q = code.z, code.z // LDPC_P
    t = np.arange(code.n)
    n_sys = code.k - 2 * z
    v = np.where(t < n_sys, 2 * z + t, code.k_ldpc + (t - n_sys))
    c, r = v // z, v % z
    return (c - 2) * q + r % q, r // q


def ldpc_out_bits(code: twin.LdpcCode) -> np.ndarray:
    """The bit index of each output word and lane of rtl/ldpc_dec [words,
    16], -1 for the lanes at or past k."""
    z, q = code.z, code.z // LDPC_P
    a = np.arange(ldpc_out_words(code))[:, None]
    v = (a // q) * z + np.arange(LDPC_P)[None, :] * q + a % q
    return np.where(v < code.k, v, -1)


def ldpc_cycles(code: twin.LdpcCode, iterations: int, early_stop: bool = True, num_iter: int = 20) -> int:
    """Approximate clocks of one codeword in rtl/ldpc_dec (load, iterations,
    the part of the next one the early stop overlaps, output)."""
    q = code.z // LDPC_P
    edges = sum(len(r) for r in code.rows)
    per_iter = edges * q + sum(len(r) + 7 for r in code.rows)
    spec = edges * q if early_stop and iterations < num_iter else 0
    return code.dec_cols * q + iterations * per_iter + spec + ldpc_out_words(code)


ACC = -1             # the encoder's accumulator vector, in an encoder program


def ldpc_enc_program(code: twin.LdpcCode) -> list[tuple[int, list[tuple[int, int]]]]:
    """ldpc.Code.encode (modem/src/ldpc.zig) as a program of vector
    operations for rtl/tx: each (dst, [(src, s), ...]) sets the z-vector
    dst to the XOR of the src vectors read shifted by s (x[(i + s) mod z]);
    vectors are base columns or ACC. p0 is the four core rows' sum over the
    information columns, placed by the core shift; each other core column
    comes from a core row where it is the only unknown; each extension row
    gives its own parity column. Placing acc at shift s is reading it at
    z - s."""
    z, kb = code.z, code.k_b
    rows = code.rows
    last_bit = 2 * z + code.n - 1 + (code.k_ldpc - code.k)
    enc_rows = max(4, last_bit // z + 1 - kb)
    if enc_rows > len(rows):
        raise ValueError("the decoder's rows do not reach the transmitted parity")
    place = lambda s: (ACC, (z - s) % z)                                       # noqa: E731
    prog = []
    # p0: the core shift is the one shift of column k_b that the core rows do not cancel
    left = []
    for r in rows[:4]:
        for c, s in r:
            if c == kb:
                left.remove(s) if s in left else left.append(s)
    if len(left) != 1:
        raise ValueError("no core shift")
    prog.append((ACC, [(c, s) for r in rows[:4] for c, s in r if c < kb]))
    prog.append((kb, [place(left[0])]))
    known = {kb}
    while len(known) < 4:
        progress = False
        for r in rows[:4]:
            unknown = [(c, s) for c, s in r if kb <= c < kb + 4 and c not in known]
            if len(unknown) != 1:
                continue
            col, s_t = unknown[0]
            prog.append((ACC, [(c, s) for c, s in r if c != col]))
            prog.append((col, [place(s_t)]))
            known.add(col)
            progress = True
        if not progress:
            raise ValueError("the core does not solve")
    for ri in range(4, enc_rows):
        r = rows[ri]
        (s_t,) = [s for c, s in r if c == kb + ri]
        prog.append((ACC, [(c, s) for c, s in r if c != kb + ri]))
        prog.append((kb + ri, [place(s_t)]))
    return prog


def ldpc_encode_program(code: twin.LdpcCode, info: np.ndarray) -> np.ndarray:
    """The n code bits of k information bits by ldpc_enc_program, on plain
    vectors (the model of the RTL encoder)."""
    z = code.z
    cols = np.zeros((code.dec_cols, z), np.uint8)
    cols.reshape(-1)[:code.k] = info
    acc = np.zeros(z, np.uint8)
    for dst, srcs in ldpc_enc_program(code):
        v = np.zeros(z, np.uint8)
        for c, s in srcs:
            v ^= np.roll(acc if c == ACC else cols[c], -s)
        if dst == ACC:
            acc = v
        else:
            cols[dst] = v
    flat = cols.reshape(-1)
    t = np.arange(code.n)
    n_sys = code.k - 2 * z
    return flat[np.where(t < n_sys, 2 * z + t, code.k_ldpc + (t - n_sys))]


TX_LEVELS, TX_OPS, TX_SRCS, TX_SIGNS, TX_PRE, TX_PERM = 0x40, 0x400, 0x800, 0xC00, 0x1000, 0x4000


def tx_cfg(tw: twin.TwinModem, preamble: np.ndarray) -> list[tuple[int, int]]:
    """rtl/tx/tx_frame.v: registers and tables of one numerology and MCS,
    and the preamble (TwinSync.preamble)."""
    code, np_ = tw.code, len(tw.psym)
    z, q = code.z, code.z // LDPC_P
    psym = list(tw.psym) + [15] * (3 - np_)
    prog = ldpc_enc_program(code)

    def vec(c: int) -> int:                    # {scratch, word base}
        if c == ACC:
            return (1 << 11) | 2 * q
        return (1 << 11) | c * q if c < 2 else (c - 2) * q
    ops, srcs = [], []
    for dst, ss in prog:
        ops.append((vec(dst) << 9) | len(srcs))
        for i, (c, s) in enumerate(ss):
            srcs.append((int(i == len(ss) - 1) << 21) | ((s % q) << 16) | ((s // q) << 12) | vec(c))
    if len(ops) > 64 or len(srcs) > 512:
        raise ValueError("encoder program too large")
    regs = [tw.S, tw.F, np_, *psym, tw.B, tw.num_cw, ldpc_llr_words(code), q, ldpc_out_words(code), code.k, z,
            code.k_b * q, tw.tx_pilot, tw.cp, tw.num.num_guard[0], tw.num.num_guard[1], int(tw.num.dc_null), len(ops)]
    out = list(enumerate(regs))
    out += [(TX_LEVELS + i, _u(v, 12)) for i, v in enumerate(tw.tx_level)]
    out += [(TX_OPS + o, v) for o, v in enumerate(ops)]
    out += [(TX_SRCS + e, v) for e, v in enumerate(srcs)]
    out += [(TX_SIGNS + p, (int(s[0] < 0) << 1) | int(s[1] < 0)) for p, s in enumerate(tw.signs.reshape(-1, 2))]
    out += [(TX_PRE + i, (_u(x[0], 12) << 12) | _u(x[1], 12)) for i, x in enumerate(preamble)]
    out += [(TX_PERM + i, int(p)) for i, p in enumerate(rx_perm(tw))]
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
    cww = ldpc_llr_words(tw.code)
    if tw.num_cw * cww > 1 << LDPC_LLR_AW:
        raise ValueError("codewords too large for the LLR bank")
    regs = {0: S, 1: F, 2: np_, 3: psym[0], 4: psym[1], 5: psym[2], 6: sum(tw.psym), 7: L,
            8: tw.num.num_guard[0], 9: tw.num.num_guard[1], 10: int(tw.num.dc_null), 11: _u(l_min, 8), 12: tw.B,
            13: tw.num_cw, 14: cww, 16: tw.a_q, 17: int(tw.phase_track), 18: tw.track_gain, 21: tw.cp}
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
    out += [(RX_PERM + i, int(p)) for i, p in enumerate(rx_perm(tw))]
    return out


def rx_perm(tw: twin.TwinModem) -> np.ndarray:
    """rx_frame's deinterleaver: where LLR i of the frame (data RE d, bit b,
    i = d B + b) goes in an LLR bank, {discard, word, lane}: codeword c's
    LLR t at its decoder slot (ldpc_llr_slot) plus c times the codeword's
    words; the padding past the codewords is discarded."""
    word, lane = ldpc_llr_slot(tw.code)
    cww = ldpc_llr_words(tw.code)
    p = np.asarray(tw.perm, np.int64)
    used = p < tw.num_cw * tw.n
    c, t = np.divmod(np.where(used, p, 0), tw.n)
    return np.where(used, ((word[t] + c * cww) << 4) | lane[t], 1 << 14)


SYNC_ENABLE = 12
SYNC_SIGNS = 256


def sync_cfg(cfg, rearm: int = 64) -> list[tuple[int, int]]:
    """rtl/sync/rx_sync.v: registers and the first pilot symbol's signs of a
    synchroniser configuration (sync.SyncConfig or any object with its
    fields), without the enable (SYNC_ENABLE, 1, written last)."""
    c = cfg
    fixed = dict(fft_size=256, cp_len=32, period=32, reps=9, window=256, metric_shift=8)
    if any(getattr(c, k) != v for k, v in fixed.items()):
        raise ValueError(f"rx_sync is built for {fixed}")
    F = c.fft_size - c.guard_lo - c.guard_hi - int(bool(c.dc_null))
    regs = [c.threshold_q8, c.first_path_shift, c.noise_floor, c.cir_min_ratio, c.backoff, rearm, c.num_symbols,
            c.pilot_symbol, int(bool(c.cp_cfo)), c.guard_lo, c.guard_hi, int(bool(c.dc_null))]
    signs = tables.pilot_signs(F, c.pilot_seed)
    return list(enumerate(regs)) + [(SYNC_SIGNS + f, (int(s[0] < 0) << 1) | int(s[1] < 0)) for f, s in enumerate(signs)]


def rx_scale_cfg(scale: float) -> list[tuple[int, int]]:
    """The per-frame LLR scale of rx_frame: mantissa and shift
    (twin.llr_mantissa)."""
    mant, shift = twin.llr_mantissa(scale)
    return [(RX_MANT, mant), (RX_SHIFT, shift)]
