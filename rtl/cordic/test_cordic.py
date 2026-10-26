"""cocotb: the CORDIC cores against the golden vectors and the twin.

  blocks.npz   cordic_xy -> cordic_vector, cossin_ang -> cossin;
  sync_*.npz   angle30 of the correlation at the peak -> sync_ang_coarse,
               and of the cyclic-prefix sum -> sync_ang_cp (the sums are
               formed in NumPy from the ADC buffer, the angles in RTL);
  exhaustive   every rotation angle of a turn, and random vectoring and
               angle30 inputs, against harness/twin.py.
"""
import numpy as np
import cocotb
from cocotb.triggers import RisingEdge

from tbutil import SYNC_FILES, load, s, start
from harness import twin


async def stream(dut, prefix: str, inputs: list[dict], outputs: tuple[str, ...]) -> list[tuple]:
    """Drive {port: value} dicts one per clock on `prefix`_*, with the index
    as the tag; return the outputs in input order."""
    out = [None] * len(inputs)
    i = 0
    got = 0
    vo, to = getattr(dut, f"{prefix}o_valid"), getattr(dut, f"{prefix}o_tag")
    while got < len(inputs):
        if i < len(inputs):
            getattr(dut, f"{prefix}_valid").value = 1
            getattr(dut, f"{prefix}_tag").value = i
            for k, v in inputs[i].items():
                getattr(dut, f"{prefix}_{k}").value = int(v)
            i += 1
        else:
            getattr(dut, f"{prefix}_valid").value = 0
        await RisingEdge(dut.clk)
        if vo.value:
            out[int(to.value)] = tuple(s(getattr(dut, f"{prefix}o_{k}")) for k in outputs)
            got += 1
    getattr(dut, f"{prefix}_valid").value = 0
    return out


async def init(dut):
    for p in ("v", "r", "a"):
        getattr(dut, f"{p}_valid").value = 0
    await start(dut)


@cocotb.test()
async def blocks_vectors(dut):
    await init(dut)
    v = load("blocks")
    got = await stream(dut, "v", [{"x": x, "y": y} for x, y in v["cordic_xy"]], ("mag", "ang"))
    assert np.array_equal(np.array(got), v["cordic_vector"]), "cordic_vector"
    # the core takes the angle modulo a turn (its low 16 bits)
    got = await stream(dut, "r", [{"ang": a % 65536} for a in v["cossin_ang"]], ("cos", "sin"))
    assert np.array_equal(np.array(got), v["cossin"]), "cossin"


@cocotb.test()
async def every_rotation_angle(dut):
    await init(dut)
    ang = np.concatenate([np.arange(-32768, 32768), [65536, -65536, 2 ** 31 - 1, -2 ** 31, 98304, -98304]])
    got = np.array(await stream(dut, "r", [{"ang": a % 65536} for a in ang], ("cos", "sin")))
    c, sn = twin.cossin_vec(ang)
    bad = np.nonzero((got[:, 0] != c) | (got[:, 1] != sn))[0]
    assert len(bad) == 0, f"{len(bad)} angles differ, first {ang[bad[:5]]}"


@cocotb.test()
async def random_vectoring(dut):
    await init(dut)
    rng = np.random.default_rng(11)
    xy = np.concatenate([rng.integers(-2 ** 30, 2 ** 30, (4000, 2)), rng.integers(-300, 300, (1000, 2)),
                         [[0, 0], [0, 5], [0, -5], [-7, 0], [2 ** 30 - 1, -2 ** 30], [-2 ** 30, -2 ** 30]]])
    got = await stream(dut, "v", [{"x": x, "y": y} for x, y in xy], ("mag", "ang"))
    want = [twin.cordic_vector(int(x), int(y)) for x, y in xy]
    bad = [i for i in range(len(xy)) if got[i] != want[i]]
    assert not bad, f"{len(bad)} differ, first {xy[bad[0]]}: {got[bad[0]]} != {want[bad[0]]}"
    # angle30 over the whole 48-bit range
    ws = np.concatenate([rng.integers(-2 ** 46, 2 ** 46, (2000, 2)), rng.integers(-2 ** 31, 2 ** 31, (1000, 2)),
                         rng.integers(-1000, 1000, (500, 2))])
    got = await stream(dut, "a", [{"re": a, "im": b} for a, b in ws], ("ang",))
    want = [(twin.angle30(int(a), int(b)),) for a, b in ws]
    bad = [i for i in range(len(ws)) if got[i] != want[i]]
    assert not bad, f"angle30: {len(bad)} differ, first {ws[bad[0]]}"


@cocotb.test()
async def sync_angles(dut):
    """The synchroniser's two angles from the ADC buffers of the sync vectors."""
    await init(dut)
    ins, want = [], []
    for name in SYNC_FILES:
        v = load(name)
        c = v["meta"]["sync_config"]
        P, W, N, cp, S = c["period"], c["window"], c["fft_size"], c["cp_len"], c["num_symbols"]
        for f in range(v["rx"].shape[0]):
            r = v["rx"][f].astype(np.int64)
            d = int(v["sync_peak"][f])
            a, b = r[d + P:d + P + W], r[d:d + W]
            pr = int((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum())
            pi = int((a[:, 1] * b[:, 0] - a[:, 0] * b[:, 1]).sum())
            ins.append({"re": pr, "im": pi})
            want.append(("coarse", name, f, int(v["sync_ang_coarse"][f]), None))
            st = int(v["sync_start"][f])
            n0 = (st + np.arange(S)[:, None] * (N + cp) + np.arange(cp)[None, :]).reshape(-1)
            n0 = np.clip(n0, 0, len(r) - N - 1)
            qa, qb = r[n0 + N], r[n0]
            ins.append({"re": int((qa[:, 0] * qb[:, 0] + qa[:, 1] * qb[:, 1]).sum()),
                        "im": int((qa[:, 1] * qb[:, 0] - qa[:, 0] * qb[:, 1]).sum())})
            want.append(("cp", name, f, int(v["sync_ang_cp"][f]), int(v["sync_ang_coarse"][f]) * (N // P)))
    got = await stream(dut, "a", ins, ("ang",))
    for (kind, name, f, ref, sub), (ang,) in zip(want, got):
        val = ang if kind == "coarse" else twin.wrap16(ang - sub)
        assert val == ref, f"{name} frame {f} {kind}: {val} != {ref}"
    dut._log.info(f"{len(got)} synchroniser angles match")
