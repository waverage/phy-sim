"""NumPy twins of the M4 fixed-point blocks (modem/src/fx*.zig).

Each function computes, on numpy int64, exactly what its Zig block
computes: the same operations, the same rounding (fx.rshr: half up,
fx.rshrEven: half to even, in the FFT only) and the same saturation. They
are slow and written to be read; tests/test_m4_twins.py holds them
bit-exact against the Zig blocks, and tools/make_vectors.py writes their
outputs to vectors/ as the golden vectors for the RTL.
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field

import numpy as np

from . import tables

LDPC_DIR = pathlib.Path(__file__).resolve().parents[1] / "modem" / "src" / "ldpc"

# ---- primitives (fx.zig) -------------------------------------------------------------


def rshr(x, n: int):
    """Round half up, arithmetic shift."""
    x = np.asarray(x, dtype=np.int64)
    return x if n == 0 else (x + (1 << (n - 1))) >> n


def rshr_even(x, n: int):
    """Round half to even."""
    x = np.asarray(x, dtype=np.int64)
    q = x >> n
    r = x - (q << n)
    half = 1 << (n - 1)
    return q + ((r > half) | ((r == half) & ((q & 1) == 1)))


def sat(x, bits: int):
    hi = (1 << (bits - 1)) - 1
    return np.clip(x, -hi - 1, hi)


def sat_sym(x, bits: int):
    hi = (1 << (bits - 1)) - 1
    return np.clip(x, -hi, hi)


def div_round(a, b):
    """a / b rounded half up, b > 0."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    return (2 * a + b) // (2 * b)


# ---- QAM -------------------------------------------------------------------------------


def pam_gray(label: int, m: int) -> int:
    """Sionna's pam_gray: the odd integer level of an m-bit label, MSB first."""
    b0 = (label >> (m - 1)) & 1
    if m == 1:
        return 1 - 2 * b0
    return (1 - 2 * b0) * ((1 << (m - 1)) - pam_gray(label & ((1 << (m - 1)) - 1), m - 1))


def pam_levels(bits_per_symbol: int) -> np.ndarray:
    """Integer level per dimension label for square Gray QAM."""
    m = bits_per_symbol // 2
    return np.array([pam_gray(lab, m) for lab in range(1 << m)], dtype=np.int64)


def qam_norm(bits_per_symbol: int) -> float:
    """Amplitude a with a * levels of unit average symbol energy."""
    lv = pam_levels(bits_per_symbol)
    return 1 / math.sqrt(2 * np.mean(lv.astype(float) ** 2))


def maxlog_llr(y: np.ndarray, no, bits_per_symbol: int) -> np.ndarray:
    """Floating-point max-log LLRs log p(1) / p(0), [..., bits_per_symbol],
    of unit-energy Gray QAM: per dimension, (min distance over the points
    whose bit is 0 - min over those whose bit is 1) / no."""
    m = bits_per_symbol // 2
    pts = pam_levels(bits_per_symbol) * qam_norm(bits_per_symbol)
    y = np.asarray(y)
    no = np.asarray(no, dtype=np.float64)
    out = np.empty(y.shape + (bits_per_symbol,), np.float64)
    for dim, v in enumerate((y.real, y.imag)):
        d = (v[..., None].astype(np.float64) - pts) ** 2                 # [..., 2^m]
        for j in range(m):
            bit = (np.arange(1 << m) >> (m - 1 - j)) & 1
            out[..., 2 * j + dim] = (d[..., bit == 0].min(-1) - d[..., bit == 1].min(-1)) / no
    return out


# ---- FFT (fx.Fft.runFixed) ------------------------------------------------------------------

def twiddles(n: int) -> np.ndarray:
    """round(65536 e^{-2 pi i j / n}), j < n / 2, as [n / 2, 2] int64."""
    a = -2 * np.pi * np.arange(n // 2) / n
    return np.stack([np.floor(np.cos(a) * 65536 + 0.5), np.floor(np.sin(a) * 65536 + 0.5)], -1).astype(np.int64)


def fft(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Fixed-point radix-2 DIT FFT of [..., n, 2] int (16-bit) I/Q pairs:
    unitary, the even stages halve, every output rounded half to even and
    saturated to 16 bits."""
    x = np.asarray(x, dtype=np.int64)
    n = x.shape[-2]
    m = int(math.log2(n))
    rev = np.array([int(f"{i:0{m}b}"[::-1], 2) for i in range(n)])
    x = x[..., rev, :].copy()
    tw = twiddles(n)
    size, stage = 2, 0
    while size <= n:
        half, step = size // 2, n // size
        sh = 16 + (1 if stage % 2 == 0 else 0)
        w = tw[np.arange(half) * step]
        wr, wi = w[:, 0], (-w[:, 1] if inverse else w[:, 1])
        xs = x.reshape(*x.shape[:-2], n // size, size, 2)
        a, b = xs[..., :half, :], xs[..., half:, :]
        tr = wr * b[..., 0] - wi * b[..., 1]
        ti = wr * b[..., 1] + wi * b[..., 0]
        ar, ai = a[..., 0] << 16, a[..., 1] << 16
        top = np.stack([sat(rshr_even(ar + tr, sh), 16), sat(rshr_even(ai + ti, sh), 16)], -1)
        bot = np.stack([sat(rshr_even(ar - tr, sh), 16), sat(rshr_even(ai - ti, sh), 16)], -1)
        x = np.concatenate([top, bot], -2).reshape(x.shape)
        size, stage = size * 2, stage + 1
    return x


# ---- CORDIC (fx.vector, fx.cossin) ------------------------------------------------------------

TURN = 65536
CORDIC_ATAN = [8192, 4836, 2555, 1297, 651, 326, 163, 81, 41, 20, 10, 5, 3, 1]
CORDIC_X0 = 19898


def cordic_vector(x: int, y: int) -> tuple[int, int]:
    """(magnitude times K, angle in (-32768, 32768])."""
    ang = 0
    if x < 0:
        x, y, ang = -x, -y, TURN // 2
    for i, a in enumerate(CORDIC_ATAN):
        dx, dy = y >> i, x >> i
        if y > 0:
            x, y, ang = x + dx, y - dy, ang + a
        else:
            x, y, ang = x - dx, y + dy, ang - a
    if ang > TURN // 2:
        ang -= TURN
    return x, ang


def cossin(ang: int) -> tuple[int, int]:
    """2^15 (cos, sin) of a binary angle."""
    a = ang % TURN
    if a > TURN // 2:
        a -= TURN
    neg = False
    if a > TURN // 4:
        a, neg = a - TURN // 2, True
    elif a < -TURN // 4:
        a, neg = a + TURN // 2, True
    x, y = CORDIC_X0, 0
    for i, at in enumerate(CORDIC_ATAN):
        dx, dy = y >> i, x >> i
        if a >= 0:
            x, y, a = x - dx, y + dy, a - at
        else:
            x, y, a = x + dx, y - dy, a + at
    return (-x, -y) if neg else (x, y)


def rot16(v: np.ndarray, cs: tuple[int, int]) -> np.ndarray:
    """v [..., 2] * (c + j s) / 2^15, rounded half up, 16-bit."""
    c, s = cs
    re = v[..., 0] * c - v[..., 1] * s
    im = v[..., 0] * s + v[..., 1] * c
    return np.stack([sat(rshr(re, 15), 16), sat(rshr(im, 15), 16)], -1)


# ---- 5G LDPC code structure (ldpc.Code) ----------------------------------------------------

S_VAL = [[2, 4, 8, 16, 32, 64, 128, 256], [3, 6, 12, 24, 48, 96, 192, 384], [5, 10, 20, 40, 80, 160, 320],
         [7, 14, 28, 56, 112, 224], [9, 18, 36, 72, 144, 288], [11, 22, 44, 88, 176, 352],
         [13, 26, 52, 104, 208], [15, 30, 60, 120, 240]]


@dataclass
class LdpcCode:
    k: int
    n: int
    bg: int
    z: int
    k_b: int
    k_ldpc: int
    n_ldpc: int
    k_filler: int
    dec_rows: int
    dec_cols: int
    rows: list = field(default_factory=list)     # per base row kept by the decoder: [(col, shift)]


def ldpc_code(k: int, n: int) -> LdpcCode:
    """Base graph, lifting and decoder pruning exactly as ldpc.Code.init."""
    r = k / n
    bg = 2 if (k <= 292 or (k <= 3824 and r <= 0.67) or r <= 0.25) else 1
    sel_kb = 22 if bg == 1 else (10 if k > 640 else 9 if k > 560 else 8 if k > 192 else 6)
    best, z, i_ls = 100000, 0, 0
    for i, s in enumerate(S_VAL):
        for s1 in s:
            if k <= sel_kb * s1 < best:
                best, z, i_ls = sel_kb * s1, s1, i
    k_b = 22 if bg == 1 else 10
    num_cols = 68 if bg == 1 else 52
    rows: dict[int, list] = {}
    row = 0
    for line in (LDPC_DIR / f"5G_bg{bg}.csv").read_text().splitlines()[2:]:
        if not line.strip():
            continue
        f = [x.strip() for x in line.strip().split(";")]
        if f[0]:
            row = int(f[0])
        rows.setdefault(row, []).append((int(f[1]), int(f[2 + i_ls]) % z))
    num_rows = num_cols - k_b
    base = [rows.get(i, []) for i in range(num_rows)]
    deg = np.zeros(num_cols, int)
    for ents in base:
        for c, _ in ents:
            deg[c] += 1
    last_col = num_cols
    while last_col > 1 and deg[last_col - 1] == 1:
        last_col -= 1
    k_ldpc, n_ldpc = k_b * z, num_cols * z
    k_filler = k_ldpc - k
    nb_punc = (n_ldpc - k_filler) - n - 2 * z
    n_pruned = max(last_col * z, n_ldpc - nb_punc)
    dec_cols = min(num_cols, -(-n_pruned // z))
    dec_rows = dec_cols - k_b
    return LdpcCode(k, n, bg, z, k_b, k_ldpc, n_ldpc, k_filler, dec_rows, dec_cols, base[:dec_rows])


def ldpc_decode(code: LdpcCode, logits: np.ndarray, llr_bits: int = 6, app_bits: int = 8, msg_bits: int = 6,
                offset: int = 2, num_iter: int = 20, early_stop: bool = True) -> tuple[np.ndarray, int]:
    """Layered offset min-sum in fixed point, as fxldpc.Decoder(i32): the
    integer logits [n] of one codeword -> (k bits, iterations)."""
    z, k = code.z, code.k
    app_max, msg_max = (1 << (app_bits - 1)) - 1, (1 << (msg_bits - 1)) - 1
    app = np.zeros((code.dec_cols, z), np.int64)
    flat = app.reshape(-1)
    t = np.arange(code.n)
    n_sys = k - 2 * z
    j = np.where(t < n_sys, 2 * z + t, code.k_ldpc + (t - n_sys))
    flat[j] = -sat_sym(np.asarray(logits, dtype=np.int64), llr_bits)
    flat[k:code.k_ldpc] = app_max
    c2v = [np.zeros((len(ents), z), np.int64) for ents in code.rows]
    lanes = np.arange(z)
    it = 0
    while it < num_iter:
        for r, ents in enumerate(code.rows):
            tt = np.stack([sat_sym(np.roll(app[c], -s) - c2v[r][e], app_bits) for e, (c, s) in enumerate(ents)])
            mag = np.minimum(np.abs(tt), msg_max)
            idx = mag.argmin(0)
            min1 = mag[idx, lanes]
            m2 = mag.copy()
            m2[idx, lanes] = np.iinfo(np.int64).max
            min2 = m2.min(0)
            neg = tt < 0
            par = np.bitwise_xor.reduce(neg, axis=0)
            new = np.where(np.arange(len(ents))[:, None] == idx, np.maximum(min2 - offset, 0),
                           np.maximum(min1 - offset, 0))
            msg = np.where(par ^ neg, -new, new)
            c2v[r] = msg
            tn = sat_sym(tt + msg, app_bits)
            for e, (c, s) in enumerate(ents):
                app[c] = np.roll(tn[e], s)
        it += 1
        if early_stop and it < num_iter and _syndrome_zero(code, app):
            break
    return (app.reshape(-1)[:k] <= 0).astype(np.uint8), it


def _syndrome_zero(code: LdpcCode, app: np.ndarray) -> bool:
    hard = (app <= 0).astype(np.uint8)
    for ents in code.rows:
        acc = np.zeros(code.z, np.uint8)
        for c, s in ents:
            acc ^= np.roll(hard[c], -s)
        if acc.any():
            return False
    return True


# ---- the frame (fxmodem.FxModem(true)) -------------------------------------------------------------

def q_round(v, frac: int) -> np.ndarray:
    """floor(v 2^frac + 0.5) in float64, as fxmodem's q()."""
    return np.floor(np.asarray(v, dtype=np.float64) * float(1 << frac) + 0.5).astype(np.int64)


def llr_mantissa(scale: float) -> tuple[int, int]:
    """The per-frame LLR scale as a 16-bit mantissa and a right shift."""
    if scale <= 0:
        return 0, 1
    fr, e = math.frexp(scale)
    mant = math.floor(fr * 65536.0 + 0.5)
    if mant == 65536:
        mant, e = 32768, e + 1
    return mant, int(np.clip(16 - e, 1, 62))


class TwinModem:
    """FxModem(true) in NumPy. Geometry from a Numerology, the codeword plan
    (k, n, num_codewords), the pilot mask [S, F], and the widths of
    harness/fxmodem.FixedPoint as plain values."""

    def __init__(self, num, bits_per_symbol: int, *, k: int, n: int, num_codewords: int, pilot_mask: np.ndarray,
                 tx_scale: int = 512, dac_bits: int = 12, llr_bits: int = 6, app_bits: int = 8, msg_bits: int = 6,
                 offset: int = 2, num_iter: int = 20, early_stop: bool = True, phase_track: bool = False,
                 track_gain_q15: int = 32768, pilot_seed: int = tables.PILOT_SEED, perm_seed: int = tables.PERM_SEED):
        self.num, self.B = num, bits_per_symbol
        self.S, self.N, self.cp = num.num_ofdm_symbols, num.fft_size, num.cp_len
        self.F = num.num_effective_subcarriers
        self.k, self.n, self.num_cw = k, n, num_codewords
        self.code = ldpc_code(k, n)
        self.dec = dict(llr_bits=llr_bits, app_bits=app_bits, msg_bits=msg_bits, offset=offset, num_iter=num_iter,
                        early_stop=early_stop)
        self.dac_bits, self.phase_track, self.track_gain = dac_bits, phase_track, track_gain_q15
        n_ = self.N
        idx = [i for i in range(num.num_guard[0], n_ - num.num_guard[1]) if not (num.dc_null and i == n_ // 2)]
        self.bin = np.array([(i + n_ // 2) % n_ for i in idx])
        mask = np.asarray(pilot_mask).reshape(self.S, self.F).astype(bool)
        self.psym = [s for s in range(self.S) if mask[s].all()]
        self.dsym = [s for s in range(self.S) if not mask[s].any()]
        assert len(self.psym) + len(self.dsym) == self.S, "block pilots only"
        self.signs = tables.pilot_signs(len(self.psym) * self.F, pilot_seed).astype(np.int64).reshape(-1, self.F, 2)
        self.frame_bits = len(self.dsym) * self.F * self.B
        self.perm = tables.interleaver(self.frame_bits, perm_seed).astype(np.int64)
        self.levels = pam_levels(self.B)
        norm = 2 * float(np.mean(self.levels.astype(float) ** 2))
        self.a_q = int(math.floor(65536.0 / math.sqrt(norm) + 0.5))
        self.tx_level = np.floor(tx_scale * self.levels.astype(np.float64) / math.sqrt(norm) + 0.5).astype(np.int64)
        self.tx_pilot = int(math.floor(tx_scale / math.sqrt(2.0) + 0.5))

    def set_filter(self, filt: dict):
        """Quantise the design as FxModem.setFilter does."""
        bt = np.asarray(filt["bt"], np.complex64)
        self.bq = np.stack([sat(q_round(bt.real, 15), 16), sat(q_round(bt.imag, 15), 16)], -1)   # [F, L, 2]
        self.start = np.asarray(filt["start"], np.int64)
        self.aq = sat(q_round(np.asarray(filt["at"], np.float32), 14), 16)                      # [S, Npsym]
        r = np.asarray(filt["r"], np.float32).reshape(self.S, self.F)
        self.rq = np.clip(q_round(r, 15), 0, 32768)[self.dsym].reshape(-1)                      # [num_data]

    # -- TX ------------------------------------------------------------------------------------------

    def tx(self, coded: np.ndarray) -> np.ndarray:
        """The LDPC codewords of one frame [num_codewords, n] (0/1) -> int16
        samples [S (N + cp), 2]. The encoder itself is checked against
        Sionna in tests/test_m3_modem.py."""
        bits = np.zeros(self.frame_bits, np.int64)
        bits[:self.num_cw * self.n] = np.asarray(coded).reshape(-1)
        txb = bits[self.perm].reshape(-1, self.B)
        md = self.B // 2
        w = 1 << np.arange(md - 1, -1, -1)
        li, lq = (txb[:, 0::2] * w).sum(-1), (txb[:, 1::2] * w).sum(-1)
        data = np.stack([self.tx_level[li], self.tx_level[lq]], -1).reshape(len(self.dsym), self.F, 2)
        out = []
        for s in range(self.S):
            grid = np.zeros((self.N, 2), np.int64)
            if s in self.psym:
                grid[self.bin] = self.signs[self.psym.index(s)] * self.tx_pilot
            else:
                grid[self.bin] = data[self.dsym.index(s)]
            t = fft(grid, inverse=True)
            out.append(sat(np.concatenate([t[self.N - self.cp:], t]), self.dac_bits))
        return np.concatenate(out).astype(np.int16)

    # -- RX ------------------------------------------------------------------------------------------

    def rx(self, y: np.ndarray, scale: float, l_min: int = 0) -> dict:
        """One frame of int16 samples [len, 2] -> dict: bits, LLRs
        (deinterleaved, [num_codewords, n]), iterations, and the probes
        y, h, phi of FxModem.rx."""
        S, F, N, cp = self.S, self.F, self.N, self.cp
        y = np.asarray(y, np.int64)
        tw = twiddles(N)
        Y = np.zeros((S, F, 2), np.int64)
        for s in range(S):
            yb = fft(y[s * (N + cp) + cp:][:N])
            if l_min != 0:
                j = (l_min * np.arange(N)) % N
                w = tw[j % (N // 2)] * np.where(j < N // 2, 1, -1)[:, None]
                re = yb[:, 0] * w[:, 0] - yb[:, 1] * w[:, 1]
                im = yb[:, 0] * w[:, 1] + yb[:, 1] * w[:, 0]
                yb = np.stack([sat(rshr(re, 16), 16), sat(rshr(im, 16), 16)], -1)
            Y[s] = yb[self.bin]
        # LS and the frequency filter
        L = self.bq.shape[1]
        win = self.start[:, None] + np.arange(L)[None, :]                     # [F, L]
        hf = np.zeros((len(self.psym), F, 2), np.int64)
        for t, s in enumerate(self.psym):
            a, b = self.signs[t][:, 0], self.signs[t][:, 1]
            lr = a * Y[s, :, 0] + b * Y[s, :, 1]
            li = a * Y[s, :, 1] - b * Y[s, :, 0]
            wr, wi = self.bq[..., 0], self.bq[..., 1]
            ar = (wr * lr[win] - wi * li[win]).sum(-1)
            ai = (wr * li[win] + wi * lr[win]).sum(-1)
            hf[t] = np.stack([sat(rshr(ar, 15), 16), sat(rshr(ai, 15), 16)], -1)
        phi = self._track(Y, hf) if self.phase_track else np.zeros(S, np.int64)
        # the time filter at the data symbols
        h = sat(rshr(np.einsum("st,tfc->sfc", self.aq[self.dsym], hf), 14), 16)       # [Sd, F, 2]
        llr_tx = self._demap(Y[self.dsym].reshape(-1, 2), h.reshape(-1, 2), scale)
        llr = np.empty_like(llr_tx)
        llr[self.perm] = llr_tx
        llr = llr[:self.num_cw * self.n].reshape(self.num_cw, self.n)
        bits, iters = [], 0
        for c in range(self.num_cw):
            b, it = ldpc_decode(self.code, llr[c], **self.dec)
            bits.append(b)
            iters += it
        return {"bits": np.stack(bits), "llr": llr, "iters": iters, "y": Y.reshape(S * F, 2),
                "h": h.reshape(-1, 2), "phi": phi, "hf": hf.reshape(-1, 2)}

    def _track(self, Y: np.ndarray, hf: np.ndarray) -> np.ndarray:
        S = self.S
        npi = len(self.psym)
        hs = hf.sum(0)                                                          # [F, 2]
        zr = [int((hf[t, :, 0] * hs[:, 0] + hf[t, :, 1] * hs[:, 1]).sum()) for t in range(npi)]
        zi = [int((hf[t, :, 1] * hs[:, 0] - hf[t, :, 0] * hs[:, 1]).sum()) for t in range(npi)]
        mx = max(max(abs(a), abs(b)) for a, b in zip(zr, zi))
        sh = max(mx.bit_length() - 30, 0)
        sum_ps = sum(self.psym)
        num = den = 0
        for t in range(npi):
            mag, ang = cordic_vector(zr[t] >> sh, zi[t] >> sh)
            w = int(rshr(mag, 16))
            sv = npi * self.psym[t] - sum_ps
            num += w * sv * ang
            den += w * sv * sv
        phi = np.array([int(div_round(num * (npi * s - sum_ps) * self.track_gain, den << 15)) if den > 0 else 0
                        for s in range(S)])
        for s in range(S):
            Y[s] = rot16(Y[s], cossin(-int(phi[s])))
        for t, s in enumerate(self.psym):
            hf[t] = rot16(hf[t], cossin(-int(phi[s])))
        return phi

    def _demap(self, y: np.ndarray, h: np.ndarray, scale: float) -> np.ndarray:
        md = self.B // 2
        lv = self.levels
        zr = y[:, 0] * h[:, 0] + y[:, 1] * h[:, 1]
        zi = y[:, 1] * h[:, 0] - y[:, 0] * h[:, 1]
        u = rshr((h[:, 0] ** 2 + h[:, 1] ** 2) * self.a_q, 16)
        mant, shift = llr_mantissa(scale)
        mult = rshr(mant * self.rq, 15)
        out = np.empty((y.shape[0], self.B), np.int64)
        labels = np.arange(1 << md)
        for dim, v in enumerate((zr, zi)):
            met = u[:, None] * lv[None, :] ** 2 - 2 * v[:, None] * lv[None, :]
            for j in range(md):
                bit = (labels >> (md - 1 - j)) & 1
                dd = met[:, bit == 0].min(-1) - met[:, bit == 1].min(-1)
                out[:, 2 * j + dim] = sat_sym(rshr(dd * mult, shift), self.dec["llr_bits"])
        return out.reshape(-1)


# ---- synchroniser (fxsync.Sync, M5) ------------------------------------------------------------

def cossin_vec(ang) -> tuple[np.ndarray, np.ndarray]:
    """cossin on an array of binary angles."""
    a = np.mod(np.asarray(ang, dtype=np.int64), TURN)
    a = np.where(a > TURN // 2, a - TURN, a)
    neg = (a > TURN // 4) | (a < -TURN // 4)
    a = np.where(a > TURN // 4, a - TURN // 2, np.where(a < -TURN // 4, a + TURN // 2, a))
    x = np.full(a.shape, CORDIC_X0, np.int64)
    y = np.zeros(a.shape, np.int64)
    for i, at in enumerate(CORDIC_ATAN):
        dx, dy = y >> i, x >> i
        pos = a >= 0
        x, y, a = np.where(pos, x - dx, x + dx), np.where(pos, y + dy, y - dy), np.where(pos, a - at, a + at)
    return np.where(neg, -x, x), np.where(neg, -y, y)


def angle30(re: int, im: int) -> int:
    """CORDIC angle of a wide complex sum, normalised to 30 bits first."""
    re, im = int(re), int(im)
    sh = max(max(abs(re), abs(im)).bit_length() - 30, 0)
    return cordic_vector(re >> sh, im >> sh)[1]


def nco(n: np.ndarray, inc: int) -> tuple[np.ndarray, np.ndarray]:
    """2^15 (cos, sin) of -n inc, from the top 16 bits of the 32-bit phase."""
    return cossin_vec(np.mod(-np.asarray(n, np.int64) * int(inc), 1 << 32) >> 16)


def wrap16(a: int) -> int:
    v = a % TURN
    return v - TURN if v > TURN // 2 else v


class TwinSync:
    """fxsync.Sync: cfg is a sync.SyncConfig (any object with its fields)."""

    def __init__(self, cfg):
        self.c = c = cfg
        N = c.fft_size
        idx = [i for i in range(c.guard_lo, N - c.guard_hi) if not (c.dc_null and i == N // 2)]
        self.offsets = np.array(idx) - N // 2
        self.bins = self.offsets % N
        self.F = len(idx)
        self.psign = tables.pilot_signs(self.F, c.pilot_seed).astype(np.int64)
        self.slot = N + c.cp_len
        self.pre = c.period * c.reps

    def preamble(self) -> np.ndarray:
        c, N = self.c, self.c.fft_size
        k = self.offsets[(self.offsets % (N // c.period) == 0) & (self.offsets != 0)]
        signs = tables.pilot_signs(len(k), c.stf_seed).astype(np.int64)
        amp = math.floor(c.tx_scale * math.sqrt(self.F / len(k)) / math.sqrt(2.0) + 0.5)
        x = np.zeros((N, 2), np.int64)
        x[k % N] = signs * amp
        x = fft(x, inverse=True)
        return sat(x[np.arange(self.pre) % c.period], c.dac_bits).astype(np.int16)

    @staticmethod
    def _mul_conj(a, b):
        return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1], a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1]

    def run_one(self, r: np.ndarray, probe: dict | None = None) -> dict:
        """SyncOut's fields; probe, if given, receives the intermediate
        results: the shifted correlator outputs for every d ("p" [d, 2],
        "e" [d]), the pilot window after the coarse NCO and the FFT
        ("pilot_fft" [N, 2]) and the impulse response ("cir" [N, 2])."""
        c = self.c
        r = np.asarray(r, np.int64)
        L, P, W, N, cp = r.shape[0], c.period, c.window, c.fft_size, c.cp_len
        out = dict.fromkeys(("detected", "start", "peak", "tau", "ang_coarse", "ang_cp", "inc", "inc_coarse",
                             "p_re", "p_im", "e", "cir_peak", "cir_sum"), 0)
        if L < P + W + 1:
            return out
        cr, ci = self._mul_conj(r[P:], r[:-P])
        en = (r[:, 0] ** 2 + r[:, 1] ** 2)
        e = en[P:] + en[:-P]

        def rsum(v):
            s = np.concatenate([[0], np.cumsum(v)])
            return s[W:] - s[:-W]
        pr, pi, es = rsum(cr), rsum(ci), rsum(e)                    # d = 0 .. L - P - W
        ms = c.metric_shift
        psr, psi, ess = pr >> ms, pi >> ms, es >> ms
        if probe is not None:
            probe.update(p=np.stack([psr, psi], -1), e=ess)
        mag = psr * psr + psi * psi
        over = mag * 1024 > c.threshold_q8 * ess * ess
        if not over.any():
            return out
        d0 = int(over.argmax())
        dmax = L - P - W
        d = d0 + int(mag[d0:min(d0 + self.pre, dmax) + 1].argmax())
        out.update(detected=1, peak=d, p_re=int(psr[d]), p_im=int(psi[d]), e=int(ess[d]))
        a1 = angle30(pr[d], pi[d])
        inc1 = a1 * (1 << 32) // (TURN * P)
        out.update(ang_coarse=a1, inc_coarse=inc1)

        coarse = d + self.pre
        n = np.clip(coarse + c.pilot_symbol * self.slot + cp + np.arange(N), 0, L - 1)
        x = rot16(r[n], nco(n, inc1))
        y = fft(x)
        sa, sb = self.psign[:, 0], self.psign[:, 1]
        yb = y[self.bins]
        z = np.zeros((N, 2), np.int64)
        z[self.bins, 0] = sat(rshr(sa * yb[:, 0] + sb * yb[:, 1], 1), 16)
        z[self.bins, 1] = sat(rshr(sa * yb[:, 1] - sb * yb[:, 0], 1), 16)
        h = fft(z, inverse=True)
        if probe is not None:
            probe.update(pilot_fft=y, cir=h)
        pw = h[:, 0] ** 2 + h[:, 1] ** 2
        total = int(pw.sum())
        pk = int(pw.argmax())
        near = int(pw[(pk - cp + np.arange(2 * cp + 1)) % N].sum())
        floor_sum, floor_n = total - near, N - (2 * cp + 1)
        pmax = int(pw[pk])
        cand = (pk - (cp - 1) + np.arange(cp)) % N
        pc = pw[cand]
        ok = ((pc << c.first_path_shift) >= pmax) & (pc * floor_n >= c.noise_floor * floor_sum)
        ok[-1] = True
        first = int(cand[ok.argmax()])
        tau = first - N if first >= N // 2 else first
        det = 1 | (2 if pmax * N >= c.cir_min_ratio * total else 0)
        start = coarse + tau
        out.update(detected=det, tau=tau, start=start, cir_peak=pmax, cir_sum=total)

        inc = inc1
        if c.cp_cfo:
            n0 = (start + np.arange(c.num_symbols)[:, None] * self.slot + np.arange(cp)[None, :]).reshape(-1)
            n0 = np.clip(n0, 0, L - N - 1)
            qr, qi = self._mul_conj(r[n0 + N], r[n0])
            delta = wrap16(angle30(qr.sum(), qi.sum()) - a1 * (N // P))
            inc = inc1 + delta * ((1 << 32) // (TURN * N))
            out["ang_cp"] = delta
        out["inc"] = inc
        return out

    def run(self, buf: np.ndarray) -> np.ndarray:
        from .sync import SYNC_OUT_FIELDS
        recs = [self.run_one(b) for b in np.asarray(buf)]
        return np.array([tuple(o[f] for f in SYNC_OUT_FIELDS) for o in recs],
                        dtype=[(f, np.int64) for f in SYNC_OUT_FIELDS])

    def align(self, r: np.ndarray, start: int, inc: int, length: int) -> np.ndarray:
        r = np.asarray(r, np.int64)
        n = start - self.c.backoff + np.arange(length)
        valid = (n >= 0) & (n < r.shape[0])
        v = rot16(r[np.clip(n, 0, r.shape[0] - 1)], nco(n, inc))
        return np.where(valid[:, None], v, 0).astype(np.int16)

    def stream(self, r: np.ndarray, rearm: int = 64) -> list[dict]:
        """The synchroniser on an endless sample stream (rtl/sync/rx_sync):
        run_one on the stream from a search base, which starts at 0. A frame
        whose CIR check fails moves the base past its preamble (base + peak
        + pre); a synchronised frame is aligned and moves the base to rearm
        samples before its end (start + S slot - rearm), where the next
        preamble may begin. Each found frame: run_one's fields with start and
        peak absolute, its base, and the aligned frame ("frame", S slot
        samples, rotated as run_one's buffer counts: from the base)."""
        c = self.c
        r = np.asarray(r, np.int64)
        P, W, N = c.period, c.window, c.fft_size
        flen = c.num_symbols * self.slot
        need = self.pre + c.pilot_symbol * self.slot + flen + 2 * N       # samples past the crossing
        out, base = [], 0
        while base + P + W + 1 <= r.shape[0]:
            # the first crossing from the base, in a window that grows until it has one
            span, d0 = 4 * (self.pre + flen), None
            while d0 is None:
                seg = r[base:base + span]
                cr, ci = self._mul_conj(seg[P:], seg[:-P])
                en = seg[:, 0] ** 2 + seg[:, 1] ** 2
                e = en[P:] + en[:-P]
                cs = [np.concatenate([[0], np.cumsum(v)]) for v in (cr, ci, e)]
                pr, pi, es = (s[W:] - s[:-W] for s in cs)
                ms = c.metric_shift
                psr, psi, ess = pr >> ms, pi >> ms, es >> ms
                over = (psr * psr + psi * psi) * 1024 > c.threshold_q8 * ess * ess
                if over.any():
                    d0 = int(over.argmax())
                elif base + span >= r.shape[0]:
                    break
                else:
                    span *= 2
            if d0 is None:
                break
            if base + d0 + need > r.shape[0]:
                break                                   # the stream ends inside this frame
            o = self.run_one(r[base:base + d0 + need])
            if not o["detected"] & 2:
                base += o["peak"] + self.pre
                continue
            o = {**o, "base": base, "start": base + o["start"], "peak": base + o["peak"]}
            o["frame"] = self.align(r[base:], o["start"] - base, o["inc"], flen)
            out.append(o)
            base = o["start"] + flen - rearm
        return out
