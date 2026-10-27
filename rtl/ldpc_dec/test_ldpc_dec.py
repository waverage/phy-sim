"""cocotb tests of rtl/ldpc_dec against the golden vectors and the twin:
the decoded bits and the iteration count must match exactly."""
import os

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, ReadOnly

from tbutil import FRAME_FILES, load, start
from harness import twin
from harness.rtlcfg import ldpc_cfg, ldpc_cycles, ldpc_llr_slot, ldpc_out_bits

ICARUS = os.environ.get("SIM", "verilator") == "icarus"


async def configure(dut, cfg):
    for a, d in cfg:
        dut.cfg_we.value = 1
        dut.cfg_addr.value = a
        dut.cfg_data.value = d
        await RisingEdge(dut.clk)
    dut.cfg_we.value = 0


BASE = 37        # the codeword's first word in the LLR memory: any offset must work


def llr_memory(code, llr: np.ndarray) -> list[int]:
    """The LLR memory words (lane layout, 6 bits a lane) of one codeword;
    the unused lanes hold noise, which the decoder must ignore."""
    rng = np.random.default_rng(len(llr))
    lanes = rng.integers(-32, 32, (1 << 10, 16))
    word, lane = ldpc_llr_slot(code)
    lanes[BASE + word, lane] = llr
    return [sum((int(v) & 63) << (6 * j) for j, v in enumerate(row)) for row in lanes]


async def decode(dut, code, llr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """One codeword: (bits, iterations, clocks from start to done)."""
    mem = llr_memory(code, llr)
    out = ldpc_out_bits(code)
    dut.llr_base.value = BASE
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    bits = np.full(code.k, 2, np.uint8)
    clocks = 1
    while True:
        await ReadOnly()
        re, addr = int(dut.llr_re.value), int(dut.llr_addr.value)
        if int(dut.out_valid.value):
            a, wd, mk = int(dut.out_addr.value), int(dut.out_word.value), int(dut.out_mask.value)
            for j in range(16):
                assert ((mk >> j) & 1) == (out[a, j] >= 0), f"output word {a}: mask {mk:04x}"
                if out[a, j] >= 0:
                    bits[out[a, j]] = (wd >> j) & 1
        if int(dut.done.value):
            it = int(dut.iters.value)
            await RisingEdge(dut.clk)
            return bits, it, clocks
        await RisingEdge(dut.clk)
        clocks += 1
        if re:
            dut.llr_data.value = mem[addr]


async def setup(dut):
    dut.cfg_we.value = 0
    dut.start.value = 0
    dut.llr_base.value = 0
    dut.llr_data.value = 0
    dut.out_ready.value = 1
    await start(dut)


async def check(dut, k, n, llrs, want_bits=None, **dec):
    code = twin.ldpc_code(k, n)
    await configure(dut, ldpc_cfg(code, **{key: dec[key] for key in ("num_iter", "offset", "early_stop") if key in dec}))
    for c, llr in enumerate(llrs):
        tb, tit = twin.ldpc_decode(code, llr, **dec)
        if want_bits is not None:
            assert np.array_equal(tb, want_bits[c])
        bits, it, clocks = await decode(dut, code, llr)
        assert np.array_equal(bits, tb), f"k={k} n={n} codeword {c}: {int((bits != tb).sum())} bits differ"
        assert it == tit, f"k={k} n={n} codeword {c}: {it} iterations, twin {tit}"
        est = ldpc_cycles(code, it, dec.get("early_stop", True), dec.get("num_iter", 20))
        dut._log.info(f"k={k} n={n} codeword {c}: {it} iterations, {int(tb.sum())} ones, match; "
                      f"{clocks} clocks (model {est})")


@cocotb.test()
async def blocks_vectors(dut):
    """blocks.npz: four codewords of (4776, 9552) at 1.5 dB."""
    await setup(dut)
    v = load("blocks")
    k, n = int(v["ldpc_k"]), int(v["ldpc_n"])
    rows = range(1) if ICARUS else range(len(v["ldpc_llr"]))
    await check(dut, k, n, [v["ldpc_llr"][i] for i in rows], [v["ldpc_bits"][i] for i in rows])
    for i in rows:
        assert int(v["ldpc_iters"][i]) == twin.ldpc_decode(twin.ldpc_code(k, n), v["ldpc_llr"][i])[1]


@cocotb.test()
async def frame_vectors(dut):
    """The deinterleaved LLRs of every frame vector -> its decoded bits."""
    await setup(dut)
    for name in FRAME_FILES[:1] if ICARUS else FRAME_FILES:
        v = load(name)
        m = v["meta"]
        for f in range(v["llr"].shape[0]):
            await check(dut, m["k"], m["n"], list(v["llr"][f]), list(v["bits"][f]))


@cocotb.test(skip=ICARUS)
async def every_code(dut):
    """Every code of the MCS table (both pilot layouts), noisy LLRs near
    the waterfall, fixed 4 iterations and the early stop."""
    import dataclasses
    from harness import modem as zm
    from harness.waveform import MCS_TABLE, NUM_10MHZ, plan_codewords
    await setup(dut)
    rng = np.random.default_rng(8)
    p3 = dataclasses.replace(NUM_10MHZ, pilot_ofdm_symbols=(1, 6, 11))
    seen = set()
    for num in (NUM_10MHZ, p3):
        for mcs in MCS_TABLE:
            p = plan_codewords(num, mcs)
            if (p.k, p.n) in seen:
                continue
            seen.add((p.k, p.n))
            u = rng.integers(0, 2, p.k).astype(np.uint8)
            cw = zm.ldpc_encode(u, p.n).astype(np.float64)
            sigma = 0.8 if p.k / p.n < 0.6 else 0.55
            s = 1 - 2 * cw + rng.standard_normal(p.n) * sigma
            llr = np.clip(np.floor(-2 * s / sigma ** 2 / 0.25 + 0.5), -31, 31).astype(np.int64)
            await check(dut, p.k, p.n, [llr])
            await check(dut, p.k, p.n, [llr], num_iter=4, offset=1, early_stop=False)


@cocotb.test()
async def saturated_inputs(dut):
    """Full-scale and -32 LLRs, all zero LLRs (no early stop)."""
    await setup(dut)
    k, n = 2388, 4776
    rng = np.random.default_rng(9)
    llr = rng.choice([-32, -31, 31, 0, 1, -1], n)
    await check(dut, k, n, [llr, np.zeros(n, np.int64)], num_iter=3)
