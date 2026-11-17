"""
Shared helpers used across the pipeline steps.

Kept deliberately dependency-light - standard library + PyYAML only - so it
imports cleanly in every conda environment (maptools / lines / polygons). The one
torch helper imports torch lazily, inside the function, so this module still loads
fine where torch is absent.

Import it by putting the steps/ directory on sys.path, e.g.:

    ROOT = Path(__file__).resolve().parents[2]     # repo root
    sys.path.insert(0, str(ROOT / "steps"))
    from common import load_config, find_raw
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import yaml

# Repo root - steps/ sits one level below it, so parents[1] of this file.
ROOT = Path(__file__).resolve().parents[1]

# Raw map formats in resolution-priority order: georeferenced / wrapped forms
# first, plain images last. GDAL reads world-file + .prj sidecars for jpg/png/tif.
RAW_EXTENSIONS = (".tif", ".tiff", ".vrt", ".jpg", ".jpeg", ".png")


# -- Config ------------------------------------------------------------------

def load_config() -> dict:
    """Load config.yaml from the repo root."""
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def rel_to_root(p: Path) -> str:
    """Path relative to the repo root for tidy printing; absolute if outside it."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# -- Input file resolution ---------------------------------------------------

def find_raw(raw_root: Path, sheet_id: str) -> Path | None:
    """Return data/raw/<sheet>/<sheet>.<ext> for the first supported extension."""
    for ext in RAW_EXTENSIONS:
        p = raw_root / sheet_id / f"{sheet_id}{ext}"
        if p.exists():
            return p
    return None


def selection_file(cfg: dict, sheet_id: str, name: str | None = None) -> Path:
    """Patch list saved by pick_patches.py (one patch id per line):
    data/patches/selections/<sheet>.txt for `autolines --select`, or
    <sheet>_<name>.txt for a named list (e.g. name="manual" for `export-manual --select`)."""
    stem = sheet_id if not name else f"{sheet_id}_{name}"
    return ROOT / cfg["paths"]["patches"] / "selections" / f"{stem}.txt"


def manual_layers_dir(cfg: dict, sheet_id: str) -> Path:
    """Folder holding fully hand-made vector layers for a sheet: data/manual_layers/<sheet>/"""
    return ROOT / cfg["paths"].get("manual_layers", "data/manual_layers/") / sheet_id


def find_mask(mask_dir: Path, sheet_id: str) -> Path | None:
    """Return a map-area mask (data/map_area_masks/<sheet>/<sheet>.png/.tif), or None."""
    for ext in (".png", ".tif", ".tiff"):
        p = mask_dir / f"{sheet_id}{ext}"
        if p.exists():
            return p
    return None



# -- Downloaded areas (a big area is split by step 00 into blocks <AREA>_r01c01, ...) ---
#
# Each block is an ordinary sheet. These helpers let training pool the annotations of all
# the blocks of one area, and let prediction on any block use weights trained on its area.

def _areas_root(cfg: dict) -> Path:
    return ROOT / cfg["paths"].get("areas", "data/areas/")


def area_blocks(cfg: dict, name: str) -> list[str]:
    """Block names listed in data/areas/<name>/aoi.json (sorted), or [] if `name` is no area."""
    import json
    f = _areas_root(cfg) / name / "aoi.json"
    if not f.is_file():
        return []
    try:
        return sorted(json.loads(f.read_text(encoding="utf-8")).get("blocks", {}))
    except (OSError, ValueError):
        return []


def is_area(cfg: dict, name: str) -> bool:
    """True if `name` is a downloaded multi-block area and not itself a sheet."""
    sheet_exists = ((ROOT / cfg["paths"]["raw"] / name).exists()
                    or (ROOT / cfg["paths"]["patches"] / "images" / name).exists())
    return not sheet_exists and bool(area_blocks(cfg, name))


def area_of_sheet(cfg: dict, sheet_id: str) -> str | None:
    """The downloaded area this sheet is a block of (e.g. 'Stackpole_r01c02' -> 'Stackpole')."""
    root = _areas_root(cfg)
    if not root.is_dir():
        return None
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if sheet_id in area_blocks(cfg, d.name):
            return d.name
    return None


def export_masks_if_stale(cfg: dict, sheet_id: str, label: str) -> None:
    """Run export_masks.py for a sheet when it has labelme annotations but no up-to-date
    masks for `label` (missing, or older than the newest annotation file)."""
    import subprocess
    ann = ROOT / cfg["paths"]["annotations"]
    jsons = list((ann / "labelme_json" / sheet_id).glob("*.json"))
    if not jsons:
        return
    masks = list((ann / label / sheet_id / "masks").glob("*.png"))
    if masks and max(p.stat().st_mtime for p in masks) >= max(p.stat().st_mtime for p in jsons):
        return
    print(f"  exporting masks for {sheet_id} ...")
    subprocess.run([sys.executable, str(ROOT / "steps" / "02_annotate" / "export_masks.py"),
                    "--sheet", sheet_id], check=False, stdout=subprocess.DEVNULL)


def annotated_blocks(cfg: dict, area: str, label: str, need: str = "masks") -> list[str]:
    """Blocks of `area` that have annotation data for class `label` (exporting masks first
    where the labelme files are newer). `need` is the sub-folder that must hold PNGs."""
    ann = ROOT / cfg["paths"]["annotations"]
    out = []
    for b in area_blocks(cfg, area):
        export_masks_if_stale(cfg, b, label)
        if any((ann / label / b / need).glob("*.png")):
            out.append(b)
    return out


# -- Class names -------------------------------------------------------------------

NOT_CLASSES = {"labelme_json", "original_weights", "feedback"}   # folders that are not feature classes


def known_classes(cfg: dict) -> set[str]:
    """Class names the toolkit already uses (annotations, predictions, MapSAM adapters, labelme)."""
    names = {"boundaries", "dashed", "text", "parcels"}
    for root in (ROOT / cfg["paths"]["annotations"], ROOT / cfg["paths"]["predictions"],
                 ROOT / cfg["paths"]["models_base"] / "MapSAM"):
        if root.is_dir():
            names |= {d.name for d in root.iterdir() if d.is_dir()}
    lm = ROOT / "steps" / "02_annotate" / "labelme_config.yaml"
    if lm.exists():
        names |= set(yaml.safe_load(lm.read_text()).get("labels") or [])
    return names - NOT_CLASSES


def _forms(s: str) -> set[str]:
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return {s} | ({s[:-1]} if s.endswith("s") else set()) | ({s[:-2]} if s.endswith("es") else set())


def classify_name(name: str, classes: set[str]) -> tuple[str, str | None]:
    """('existing'|'near'|'new', closest class). 'near' = same word in another spelling
    (plural, case, small typo): it would become a separate class, so it is worth a look."""
    if name in classes:
        return "existing", name
    for c in sorted(classes):
        if c.lower() == name.lower() or _forms(c) & _forms(name):
            return "near", c
    close = difflib.get_close_matches(name.lower(), [c.lower() for c in classes], n=1, cutoff=0.8)
    if close:
        return "near", next(c for c in classes if c.lower() == close[0])
    return "new", None

def model_predicted(cfg: dict, sheet_id: str, name: str) -> bool:
    """True if a model step has produced predictions for this layer / class on this sheet."""
    d = ROOT / cfg["paths"]["predictions"] / name / sheet_id
    return d.is_dir() and any(d.iterdir())


# -- GeoPackage target resolution (the --mended / --gpkg rule) ----------------
#
# One rule everywhere: default is paths.outputs; --mended uses the hand-corrected
# GeoPackage in paths.outputs_mended; --gpkg overrides both. Writers use
# resolve_output_gpkg, feedback readers use resolve_input_gpkg (which warns rather
# than silently ignoring a mended file that exists).

def find_mended(sheet_id: str, cfg: dict) -> Path | None:
    """Hand-corrected GeoPackage for the sheet in paths.outputs_mended, or None.
    Exact name first, then any *.gpkg whose name contains the sheet id
    (e.g. 'Porlock mended.gpkg')."""
    d = ROOT / cfg["paths"].get("outputs_mended", "data/mended outputs")
    if not d.is_dir():
        return None
    exact = d / f"{sheet_id}.gpkg"
    if exact.exists():
        return exact
    hits = sorted(p for p in d.glob("*.gpkg") if sheet_id.lower() in p.stem.lower())
    return hits[0] if hits else None


def _mended_dir(cfg: dict) -> Path:
    return ROOT / cfg["paths"].get("outputs_mended", "data/mended outputs")


def resolve_output_gpkg(sheet_id: str, cfg: dict, gpkg_arg: str | None,
                        mended: bool) -> Path:
    """
    Pick the GeoPackage a vectorise step should WRITE to.

    The target is never inferred from what happens to exist on disk: these steps
    drop and rewrite their own layers, so silently redirecting into a
    hand-corrected file would destroy mending.
    """
    if gpkg_arg:
        p = Path(gpkg_arg)
        return p if p.is_absolute() else ROOT / p
    if not mended:
        return ROOT / cfg["paths"]["outputs"] / f"{sheet_id}.gpkg"
    found = find_mended(sheet_id, cfg)
    if found is None:
        sys.exit(
            f"--mended: no GeoPackage for sheet '{sheet_id}' in {_mended_dir(cfg)}\n"
            f"Looked for '{sheet_id}.gpkg' and any *.gpkg with '{sheet_id}' in the name.\n"
            f"Put the mended file there, or drop --mended to write to "
            f"{cfg['paths']['outputs']}{sheet_id}.gpkg."
        )
    return found


def resolve_input_gpkg(sheet_id: str, cfg: dict, gpkg_arg: str | None,
                       mended: bool) -> Path:
    """
    Pick the GeoPackage a feedback step should READ from.

    If a mended file exists but --mended was not passed, warn loudly rather than
    silently training on the un-mended file - that would discard the corrections
    the feedback loop exists to capture.
    """
    if gpkg_arg:
        p = Path(gpkg_arg)
        return p if p.is_absolute() else ROOT / p
    if mended:
        found = find_mended(sheet_id, cfg)
        if found is None:
            sys.exit(
                f"--mended: no GeoPackage for sheet '{sheet_id}' in {_mended_dir(cfg)}\n"
                f"Looked for '{sheet_id}.gpkg' and any *.gpkg with '{sheet_id}' in the name."
            )
        return found

    default = ROOT / cfg["paths"]["outputs"] / f"{sheet_id}.gpkg"
    available = find_mended(sheet_id, cfg)
    if available is not None:
        print(
            f"\n  ! A mended GeoPackage exists for this sheet:\n"
            f"      {available}\n"
            f"    but --mended was not passed, so corrections in it will be IGNORED\n"
            f"    and training data will come from {default.name} instead.\n"
            f"    Re-run with --mended to use the corrected layers.\n"
        )
    return default


# -- NumPy 2.x / PyTorch compatibility ----------------------------------------

def tensor_from_numpy(arr):
    """torch.from_numpy replacement compatible with NumPy 2.x.

    torch.from_numpy checks the C-level numpy.ndarray type, which changed in
    NumPy 2.0, so it fails with "expected np.ndarray (got numpy.ndarray)" when
    PyTorch was compiled against NumPy 1.x. Routing through memoryview /
    torch.frombuffer bypasses that check regardless of NumPy version.

    torch/numpy are imported lazily so this module still loads where torch is
    absent (only the polygons-env training scripts call this).
    """
    import numpy as np
    import torch

    arr = np.ascontiguousarray(arr)
    return (torch.frombuffer(memoryview(arr), dtype=torch.float32)
                 .reshape(arr.shape)
                 .clone())
