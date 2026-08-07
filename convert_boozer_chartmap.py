#!/usr/bin/env python3
import argparse
from pathlib import Path

from spatial_grid import write_compatible_chartmap


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--source", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    write_compatible_chartmap(args.source.resolve(), args.out.resolve())
