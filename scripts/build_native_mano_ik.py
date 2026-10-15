#!/usr/bin/env python3
"""Build the CPython/NumPy native MANO-to-q20 extension."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sysconfig

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = root / "mano/native"
    source = package / "_native_mcp_ik.cpp"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python did not report EXT_SUFFIX")
    output = package / f"_native_mcp_ik{suffix}"
    if (
        output.exists()
        and not args.force
        and output.stat().st_mtime >= source.stat().st_mtime
    ):
        print(output)
        return
    compiler = shlex.split(sysconfig.get_config_var("CXX") or "c++")
    command = [
        *compiler,
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fopenmp",
        f"-I{sysconfig.get_paths()['include']}",
        f"-I{np.get_include()}",
        str(source),
        "-o",
        str(output),
    ]
    subprocess.run(command, cwd=root, check=True)
    print(output)


if __name__ == "__main__":
    main()

