#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def response(candidate_id: int, unit_x: list[float], metrics: dict, source: str) -> dict:
    total = metrics["total"]
    late = metrics["late"]
    return {
        "candidate_id": candidate_id,
        "unit_x": unit_x,
        "status": "ok",
        "failure_kind": None,
        "observation": {
            "value": total["change"],
            "variance": total["paired_se"] ** 2,
            "constraints": [late["change"] + 0.01],
            "constraint_variances": [late["paired_se"] ** 2],
        },
        "metrics": {"source": source, "total": total, "late": late},
    }


def promotion_rows(document: dict, search_root: Path) -> list[dict]:
    rows = []
    for key in sorted(document["candidates"], key=int):
        paths = list(search_root.glob(f"wave*/candidates/candidate-{int(key):08d}/request.json"))
        if len(paths) != 1:
            raise ValueError(f"candidate {key} request inventory differs")
        request = json.loads(paths[0].read_text())
        aggregate = document["candidates"][key]["aggregate"]
        rows.append(response(len(rows) + 1, request["unit_x"], aggregate, "promotion1024"))
    return rows


def blend_rows(document: dict, start: int) -> list[dict]:
    rows = []
    for case in document["cases"]:
        metrics = {name: case[name] for name in ("total", "late")}
        rows.append(response(start + len(rows), case["unit"], metrics, "blend256"))
    return rows


def build(args: argparse.Namespace) -> list[dict]:
    promotion = json.loads(args.promotion.read_text())
    blend = json.loads(args.blend.read_text())
    rows = promotion_rows(promotion, args.search_root)
    rows.extend(blend_rows(blend, len(rows) + 1))
    if len({tuple(row["unit_x"]) for row in rows}) != len(rows):
        raise ValueError("resolved priming coordinates contain a duplicate")
    return rows


def write_rows(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        case = output / f"candidate-{row['candidate_id']:08d}"
        case.mkdir()
        (case / "response.json").write_text(
            json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--promotion", type=Path, required=True)
    root.add_argument("--blend", type=Path, required=True)
    root.add_argument("--search-root", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    write_rows(build(args), args.out)


if __name__ == "__main__":
    main()
