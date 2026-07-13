#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from spatial_grid import generate_design


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--wout", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--surfaces", type=float, nargs="+", required=True)
    root.add_argument("--ntheta", type=int, default=16)
    root.add_argument("--nzeta", type=int, default=16)
    root.add_argument("--nmu", type=int, default=9)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    generate_design(
        args.wout.resolve(),
        args.out.resolve(),
        np.asarray(args.surfaces),
        ntheta=args.ntheta,
        nzeta=args.nzeta,
        nmu=args.nmu,
    )
