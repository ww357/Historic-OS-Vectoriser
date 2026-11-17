"""
Single entry point for every pipeline step.

    python run.py <command> [--sheet SHEET] [any other flags...]

Saves having to remember script paths; all flags after the command are passed
straight through to the underlying script, so anything the script accepts works
here unchanged.

This CANNOT switch conda environment for you - a running process cannot change
its own interpreter. What it does instead is check the active environment before
launching and stop with the exact activate command if it is wrong, which turns a
confusing ImportError deep in a script into a one-line fix. Use --force-env to
skip that check (e.g. if your environments are named differently).

    python run.py list                     # every command, with its environment
    python run.py status --sheet SHEET     # how far this sheet has progressed
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# command -> (script path relative to repo root, required conda env)
# Environments are per CONTEXT.md: maptools = geospatial, lines = TensorFlow
# U-Net (solid + dashed), polygons = PyTorch (MapSAM / text / parcels).
COMMANDS: dict[str, tuple[str, str | None]] = {
    # 00 download
    "download":           ("steps/00_download/download.py",               "maptools"),
    # 01 patchify
    "reproject":          ("steps/01_patchify/reproject.py",              "maptools"),
    "draw-mask":          ("steps/01_patchify/draw_mask.py",              "maptools"),
    "patchify":           ("steps/01_patchify/patchify.py",               "maptools"),
    # 02 annotate
    "pick-patches":       ("steps/02_annotate/pick_patches.py",           "maptools"),
    "autolines":          ("steps/02_annotate/autolines.py",              "maptools"),
    "annotate":           ("steps/02_annotate/annotate.py",               "maptools"),
    "export-manual":      ("steps/02_annotate/export_manual.py",          "maptools"),
    "export-masks":       ("steps/02_annotate/export_masks.py",           "maptools"),
    # 03 finetune
    "train-lines":        ("steps/03_finetune/lines/train.py",            "lines"),
    "evaluate-lines":     ("steps/03_finetune/lines/evaluate.py",         "lines"),
    "dashed-masks":       ("steps/03_finetune/dashed/gaussian_masks.py",  "lines"),
    "train-dashed":       ("steps/03_finetune/dashed/train.py",           "lines"),
    "train-polygons":     ("steps/03_finetune/polygons/train.py",         "polygons"),
    # 04 predict
    "predict-lines":      ("steps/04_predict/lines/predict.py",           "lines"),
    "predict-dashed":     ("steps/04_predict/dashed/predict.py",          "lines"),
    "predict-polygons":   ("steps/04_predict/polygons/predict.py",        "polygons"),
    "predict-text":       ("steps/04_predict/text/predict.py",            "polygons"),
    "predict-parcels":    ("steps/04_predict/parcels/predict.py",         "polygons"),
    # 05 vectorise
    "vectorise-lines":    ("steps/05_vectorise/lines/vectorise.py",       "maptools"),
    "vectorise-dashed":   ("steps/05_vectorise/dashed/vectorise.py",      "maptools"),
    "vectorise-polygons": ("steps/05_vectorise/polygons/vectorise.py",    "maptools"),
    "vectorise-text":     ("steps/05_vectorise/text/vectorise.py",        "maptools"),
    "vectorise-parcels":  ("steps/05_vectorise/parcels/vectorise.py",     "polygons"),
    "import-manual":      ("steps/05_vectorise/manual/import_layers.py",  "maptools"),
    "mark-manual":        ("steps/05_vectorise/manual/mark_manual.py",    "maptools"),
    # 06 feedback
    "prepare-lines":      ("steps/06_feedback/lines/prepare.py",          "maptools"),
    "feedback-lines":     ("steps/06_feedback/lines/train.py",            "lines"),
    "prepare-polygons":   ("steps/06_feedback/polygons/prepare.py",       "polygons"),
    "feedback-polygons":  ("steps/06_feedback/polygons/train.py",         "polygons"),
    # utility - run in any environment
    "status":             ("steps/status.py",                             None),
    "fetch-weights":      ("steps/fetch_weights.py",                      None),
}

# Printed by `list`, grouped so the pipeline order is obvious.
GROUPS = [
    ("00 download",  ["download"]),
    ("01 patchify", ["reproject", "draw-mask", "patchify"]),
    ("02 annotate",  ["pick-patches", "autolines", "annotate", "export-manual", "export-masks"]),
    ("03 finetune",  ["train-lines", "train-dashed", "dashed-masks",
                      "train-polygons", "evaluate-lines"]),
    ("04 predict",   ["predict-lines", "predict-dashed", "predict-polygons",
                      "predict-text", "predict-parcels"]),
    ("05 vectorise", ["vectorise-lines", "vectorise-dashed", "vectorise-polygons",
                      "vectorise-text", "vectorise-parcels", "import-manual", "mark-manual"]),
    ("06 feedback",  ["prepare-lines", "feedback-lines",
                      "prepare-polygons", "feedback-polygons"]),
    ("utility",      ["fetch-weights", "status"]),
]


def current_env() -> str | None:
    """Active conda environment name, or None if not in one."""
    name = os.environ.get("CONDA_DEFAULT_ENV")
    if name:
        return name
    prefix = os.environ.get("CONDA_PREFIX")
    return Path(prefix).name if prefix else None


# Commands that work on one sheet at a time and are independent between sheets, so
# `--sheet AREA` can safely be run once per block of a downloaded area. Training,
# annotation and the interactive tools are NOT in this list: they need you to pick
# which block(s) to work on.
PER_BLOCK_OK = {
    "reproject", "patchify", "status",
    "predict-lines", "predict-dashed", "predict-polygons", "predict-text", "predict-parcels",
    "vectorise-lines", "vectorise-dashed", "vectorise-polygons", "vectorise-text", "vectorise-parcels",
}


# Commands that take an area name themselves and pool the annotations of all its blocks.
POOLED = {"train-lines", "train-polygons"}


def _path_setting(key: str, default: str) -> Path:
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
        return ROOT / cfg.get("paths", {}).get(key, default)
    except Exception:
        return ROOT / default


def area_blocks(name: str) -> list[str]:
    """Blocks of a downloaded area (data/areas/<name>/aoi.json) that have a raw map on disk.
    Returns [] when `name` is not an area, or is itself an existing sheet."""
    import json
    raw = _path_setting("raw", "data/raw/")
    if (raw / name).exists():
        return []
    aoi = _path_setting("areas", "data/areas/") / name / "aoi.json"
    if not aoi.exists():
        return []
    try:
        blocks = json.loads(aoi.read_text(encoding="utf-8")).get("blocks", {})
    except (OSError, ValueError):
        return []
    return sorted(b for b in blocks if (raw / b / f"{b}.tif").exists())


def find_sheet_arg(args: list[str]) -> tuple[int, str] | None:
    """(index of the value in args, value) for `--sheet X` / `--sheet=X`; the value index is
    negative-encoded as -(i+1) for the `--sheet=X` form so the caller can rebuild it."""
    for i, a in enumerate(args):
        if a == "--sheet" and i + 1 < len(args):
            return i + 1, args[i + 1]
        if a.startswith("--sheet="):
            return -(i + 1), a.split("=", 1)[1]
    return None


def run_per_block(command: str, script: Path, passthrough: list[str], found, blocks: list[str]) -> int:
    pos, area = found
    n = len(blocks)
    print(f"\n'{area}' is a downloaded area of {n} sheets: {', '.join(blocks)}\n"
          f"Running '{command}' on each in turn (Ctrl+C stops; finished sheets keep their results).")
    failed = []
    i = 0
    try:
        for i, b in enumerate(blocks, 1):
            args = list(passthrough)
            if pos < 0:
                args[-pos - 1] = f"--sheet={b}"
            else:
                args[pos] = b
            print(f"\n{'=' * 70}\n[{i}/{n}] {command} --sheet {b}\n{'=' * 70}", flush=True)
            if subprocess.run([sys.executable, str(script), *args]).returncode != 0:
                failed.append(b)
    except KeyboardInterrupt:
        print(f"\nStopped after {i - 1} of {n} sheets.")
        return 130
    print(f"\n{'=' * 70}\n{command}: {n - len(failed)} of {n} sheets finished"
          + (f"; failed: {', '.join(failed)}" if failed else "") + ".")
    return 1 if failed else 0


def print_list() -> None:
    print("\nUsage:  python run.py <command> [--sheet SHEET] [flags...]\n")
    for title, names in GROUPS:
        print(f"  {title}")
        for n in names:
            script, env = COMMANDS[n]
            print(f"    {n:<20} {env or 'any':<9} {script}")
        print()
    print("All flags after the command are passed through to the script.\n"
          "Example:  python run.py predict-parcels --sheet Selworthy "
          "--extent boundaries dashed\n")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help", "list"):
        print_list()
        sys.exit(0 if argv else 1)

    command, passthrough = argv[0], argv[1:]
    if command not in COMMANDS:
        close = [c for c in COMMANDS if command in c or c.startswith(command[:4])]
        print(f"Unknown command: {command}", file=sys.stderr)
        if close:
            print(f"Did you mean: {', '.join(sorted(close))}", file=sys.stderr)
        print("Run 'python run.py list' to see all commands.", file=sys.stderr)
        sys.exit(1)

    script_rel, needed_env = COMMANDS[command]
    script = ROOT / script_rel
    if not script.exists():
        sys.exit(f"Script not found: {script}")

    # Environment guard - catches the most common failure in a 3-env pipeline
    # before it turns into an ImportError halfway through a script.
    force = "--force-env" in passthrough
    passthrough = [a for a in passthrough if a != "--force-env"]
    active = current_env()
    if needed_env and not force and active != needed_env:
        print(
            f"'{command}' needs the '{needed_env}' environment "
            f"(active: {active or 'none'}).\n\n"
            f"  conda activate {needed_env}\n"
            f"  python run.py {command} {' '.join(passthrough)}\n\n"
            f"Pass --force-env to run anyway.",
            file=sys.stderr,
        )
        sys.exit(1)

    # `--sheet AREA` where AREA is a downloaded area split into blocks (AREA_r01c01, ...).
    found = find_sheet_arg(passthrough)
    if found and command != "download":
        blocks = area_blocks(found[1])
        if blocks and command in POOLED:
            blocks = []      # the script pools the area's annotations itself
        if blocks and command in PER_BLOCK_OK:
            raise SystemExit(run_per_block(command, script, passthrough, found, blocks))
        if blocks:
            sys.exit(f"'{found[1]}' is a downloaded area split into {len(blocks)} sheets, and "
                     f"'{command}' works on one sheet at a time.\nPick one:  --sheet "
                     + "\n           --sheet ".join(blocks))

    # Same interpreter, so the script runs in whatever env is active.
    raise SystemExit(subprocess.run([sys.executable, str(script), *passthrough]).returncode)


if __name__ == "__main__":
    main()
