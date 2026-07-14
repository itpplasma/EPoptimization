#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from spatial_grid import _to_reference_native


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--field", type=Path, required=True)
    root.add_argument("--points", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    points = np.load(args.points)
    np.save(args.out, _to_reference_native(args.field.resolve(), points))
