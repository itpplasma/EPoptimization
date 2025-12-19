#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


_ASSIGN_RE = re.compile(
    r"(?P<key>RBC|RBS|ZBC|ZBS)\(\s*(?P<n>-?\d+)\s*,\s*(?P<m>-?\d+)\s*\)\s*=\s*(?P<val>[^,!\n]+)"
)


def _format_like(original: str, value: float) -> str:
    text = original.strip()
    if "E" in text or "e" in text or "D" in text or "d" in text:
        exp_char = "E"
        if "d" in original or "D" in original:
            exp_char = "D"
        mantissa = re.split(r"[eEdD]", text, maxsplit=1)[0]
        digits = 16
        if "." in mantissa:
            digits = len(mantissa.split(".", maxsplit=1)[1])
        digits = max(6, digits)
        formatted = f"{value:.{digits}E}"
        if exp_char == "D":
            formatted = formatted.replace("E", "D", 1)
        return formatted
    return f"{value:.16g}"


def _parse_float_fortran(text: str) -> float:
    return float(text.strip().replace("d", "e").replace("D", "E"))


def scale_vmec_input(
    *,
    input_path: Path,
    output_path: Path,
    geo_scale: float,
    phiedge_scale: float | None = None,
    set_ns: int | None = None,
    set_niter: int | None = None,
    set_ns_array: list[int] | None = None,
) -> None:
    if geo_scale <= 0.0:
        raise ValueError("geo_scale must be positive")
    if phiedge_scale is not None and phiedge_scale <= 0.0:
        raise ValueError("phiedge_scale must be positive")
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []

    out_lines.append("&INDATA")
    out_lines.append(f"! Scaled from {input_path.name} with geo_scale={geo_scale:g}.")
    out_lines.append("! Boundary Fourier coefficients (RBC/RBS/ZBC/ZBS) were scaled.")
    if phiedge_scale is not None:
        out_lines.append(f"! PHIEDGE was scaled by phiedge_scale={phiedge_scale:g}.")
    out_lines.append("! License and provenance: see initial_configs/NOTICE.ALPES_QUASR_0021326.")

    in_indata = False
    for raw in lines:
        line = raw.rstrip("\n")
        if line.strip().upper().startswith("&INDATA"):
            in_indata = True
            continue
        if not in_indata:
            continue

        if line.strip().startswith("!"):
            continue

        if phiedge_scale is not None and re.match(r"^\s*PHIEDGE\b", line, flags=re.IGNORECASE):
            key, val_text = line.split("=", maxsplit=1)
            val = _parse_float_fortran(val_text)
            scaled = phiedge_scale * val
            out_lines.append(f"{key.strip()} = {_format_like(val_text, scaled)}")
            continue

        if set_ns_array is not None and re.match(r"^\s*NS_ARRAY\b", line, flags=re.IGNORECASE):
            ns_text = ", ".join(str(v) for v in set_ns_array)
            out_lines.append(f"  NS_ARRAY = {ns_text}")
            continue
        if set_ns is not None and re.match(r"^\s*NS_ARRAY\b", line, flags=re.IGNORECASE):
            out_lines.append(f"  NS_ARRAY = {set_ns:d}")
            continue
        if set_niter is not None and re.match(r"^\s*NITER\b", line, flags=re.IGNORECASE):
            out_lines.append(f"  NITER = {set_niter:d}")
            continue

        def repl(match: re.Match[str]) -> str:
            key = match.group("key")
            val_text = match.group("val")
            val = _parse_float_fortran(val_text)
            scaled = geo_scale * val
            formatted = _format_like(val_text, scaled)
            return f"{key}({match.group('n')},{match.group('m')}) = {formatted}"

        if any(k in line for k in ("RBC(", "RBS(", "ZBC(", "ZBS(")):
            new_line = _ASSIGN_RE.sub(repl, line)
            out_lines.append(new_line)
        else:
            out_lines.append(line)

        if line.strip().startswith("/"):
            break

    if not out_lines or out_lines[-1].strip() != "/":
        out_lines.append("/")

    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale a VMEC fixed-boundary input file by scaling boundary Fourier coefficients."
    )
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--out", dest="output_path", type=Path, required=True)
    parser.add_argument("--geo-scale", dest="geo_scale", type=float, required=True)
    parser.add_argument(
        "--scale-phiedge",
        dest="scale_phiedge",
        action="store_true",
        help="Also scale PHIEDGE by geo_scale^2 (keeps B approximately constant under geometric scaling).",
    )
    parser.add_argument("--set-ns", dest="set_ns", type=int, default=None)
    parser.add_argument(
        "--set-ns-array",
        dest="set_ns_array",
        type=str,
        default=None,
        help="Set NS_ARRAY explicitly, e.g. 16,32,64",
    )
    parser.add_argument("--set-niter", dest="set_niter", type=int, default=None)
    args = parser.parse_args()

    set_ns_array: list[int] | None = None
    if args.set_ns_array is not None:
        set_ns_array = [int(x.strip()) for x in args.set_ns_array.split(",") if x.strip()]

    scale_vmec_input(
        input_path=args.input_path,
        output_path=args.output_path,
        geo_scale=args.geo_scale,
        phiedge_scale=(args.geo_scale * args.geo_scale) if args.scale_phiedge else None,
        set_ns=args.set_ns,
        set_niter=args.set_niter,
        set_ns_array=set_ns_array,
    )


if __name__ == "__main__":
    main()
