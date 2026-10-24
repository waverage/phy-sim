"""M0 gate: the toolchain the plan relies on is present and importable."""
import os
import shutil
import subprocess

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ["PATH"]


def test_sionna_imports():
    import sionna
    import torch
    assert sionna.__version__.startswith("2.")
    assert int(torch.__version__.split(".")[0]) >= 2


def test_zig_present():
    assert shutil.which("zig"), "zig not on PATH (official tarball in ~/.local/bin)"
    out = subprocess.run(["zig", "version"], capture_output=True, text=True).stdout
    assert out.strip()


def test_verilator_present():
    assert shutil.which("verilator"), "verilator not on PATH (brew install verilator)"


def test_cocotb_present():
    import cocotb
    assert int(cocotb.__version__.split(".")[0]) >= 2
