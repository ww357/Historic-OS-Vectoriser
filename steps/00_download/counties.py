"""
County-layer geometry for step 00: which layer supplies which part of an area.

Pure geometry, no downloading. Used by download.py (the real run) and plan_window.py (the
interactive preview), so both always agree on where the seams are.

The idea
--------
A county tileset only holds real map INSIDE its own county: OS left the neighbouring county's
land blank on the sheet, even where the sheet rectangle extends over it. So each county becomes a
Layer limited to its sheets' outlines intersected with its own county polygon (NLS publishes
historic county boundaries). Where two layers still both have map (the boundary tolerance strip),
the pixel is taken from

    * a PREFERRED layer if you named one (wins wherever it has map), otherwise
    * the layer with more map around the pixel ("margin": distance to the edge of its own county
      and sheets), which puts the seam on the county line, away from every sheet edge.

If a county has no boundary polygon the layer falls back to its sheet outlines alone (and the
plan says so). Margins are capped at `cap_m` and measured on a padded coarse grid, so the answer
for a given place is identical whichever block it falls in - blocks therefore join without seams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from nls_index import norm

BNG, WGS84 = "EPSG:27700", "EPSG:4326"
RANK_STEP = 2048           # score = rank * RANK_STEP + 1 + margin level (0..1023)
LEVELS = 1023
DEFAULT_TOL_M = 75.0       # how far past its county line a layer is still allowed to supply map


@dataclass
class Layer:
    """One county tileset restricted to the sheets (and county) that cover the area."""
    tileset: str
    url: str
    counties: list
    max_zoom: int
    sheets: list                      # property dicts of the sheets (county, sheet, years, ...)
    foot_wgs: BaseGeometry            # union of the sheet outlines, lon/lat (for display)
    foot_bng: BaseGeometry            # CORE: sheet outlines inside the county, BNG metres (margins are measured from here)
    valid_bng: BaseGeometry = None    # core widened by the boundary tolerance: where this layer may supply map
    valid_wgs: BaseGeometry = None    # same, lon/lat (which tiles are worth fetching)
    territory_bng: BaseGeometry = None   # the county polygon itself (None if no boundary was found)
    rank: int = 0                     # higher = preferred wherever it has map (0 = automatic)
    index: int = 0                    # 1-based position among the area's layers: the value written in the
                                      #   source-index raster, the same in every block of the area
    sheet_shapes: list = field(default_factory=list, repr=False)   # each sheet's outline, lon/lat
    _prep: object = field(default=None, repr=False)

    def hits_tile(self, z: int, x: int, y: int) -> bool:
        """Could this layer supply map inside the Web Mercator tile z/x/y?"""
        if self._prep is None:
            self._prep = prep(self.valid_wgs)
        n = 2 ** z
        lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
        lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        return self._prep.intersects(box(lon0, lat0, lon1, lat1))

    @property
    def label(self) -> str:
        return ", ".join(self.counties)

    @property
    def has_boundary(self) -> bool:
        return self.territory_bng is not None

    def years(self) -> str:
        """e.g. 'surveyed 1873-1889, published 1888-1902' (from the sheet records)."""
        def span(key):
            vals = []
            for s in self.sheets:
                try:
                    vals.append(int(str(s.get(key) or "").strip()[:4]))
                except ValueError:
                    pass
            if not vals:
                return None
            return f"{min(vals)}" if min(vals) == max(vals) else f"{min(vals)}-{max(vals)}"
        sv, pb = span("SUR_STA"), span("PUB_STA")
        return ", ".join(x for x in (f"surveyed {sv}" if sv else None, f"published {pb}" if pb else None) if x)


def to_bng(geom: BaseGeometry) -> BaseGeometry:
    """lon/lat geometry -> BNG, through the same PROJ as the tile warp so edges line up."""
    return shape(transform_geom(WGS84, BNG, mapping(geom)))


def to_wgs(geom: BaseGeometry) -> BaseGeometry:
    return shape(transform_geom(BNG, WGS84, mapping(geom)))


def _key(name: str) -> str:
    """County name reduced for matching between NLS data sets: 'Devonshire' and 'Devon' -> 'devon'."""
    n = norm(name)
    return n[:-5] if n.endswith("shire") and len(n) > 8 else n


def match_territory(counties_: list, boundaries: list[dict]) -> BaseGeometry | None:
    """Union (lon/lat) of the boundary polygons that belong to these county names, or None."""
    keys = [_key(c) for c in counties_]
    named = [(_key((f.get("properties") or {}).get("NAME") or (f.get("properties") or {}).get("COUNTY") or ""), f)
             for f in boundaries or []]
    named = [(nm, f) for nm, f in named if nm]
    # Exact (or 'Yorkshire' -> 'Yorkshire East Riding' style prefix) matches win. Fuzzy matching is only a
    # fallback for spelling differences (Caernarvon / Carnarvon) and must never pair two different
    # counties (Herefordshire / Hertfordshire both have an exact match, so fuzzy is not tried for them).
    hits = [f for nm, f in named
            if any(nm == k or (min(len(nm), len(k)) >= 4 and (nm.startswith(k) or k.startswith(nm))) for k in keys)]
    if not hits:
        scored = [(max(SequenceMatcher(None, nm, k).ratio() for k in keys), f) for nm, f in named]
        best = max((s for s, _ in scored), default=0.0)
        if best >= 0.88:
            hits = [f for s, f in scored if s == best]
    return unary_union([shape(f["geometry"]) for f in hits]) if hits else None


def build_layers(family: dict, features: list[dict], prefer=(), exclude=(), boundaries=None,
                 tol_m: float = DEFAULT_TOL_M) -> tuple[list[Layer], dict]:
    """Group sheet outlines into one Layer per county tileset.

    prefer / exclude take county names (any spelling of the letters) or tileset ids.
    boundaries: county polygon features (lon/lat) used to limit each layer to its own county.
    Returns (layers sorted by tileset id, {county: n_sheets} for sheets whose county has no tileset).
    """
    county_to_ts = {}
    for ts_id, ts in family["tilesets"].items():
        for c in ts["counties"]:
            county_to_ts[norm(c)] = ts_id
    groups: dict[str, list] = {}
    unmatched: dict[str, int] = {}
    for f in features:
        c = (f.get("properties") or {}).get("COUNTY")
        ts_id = county_to_ts.get(norm(c))
        if ts_id is None:
            unmatched[str(c)] = unmatched.get(str(c), 0) + 1
            continue
        groups.setdefault(ts_id, []).append(f)

    def resolve(names, what):
        ids = []
        for n in names:
            hit = [t for t in groups if norm(t) == norm(n) or any(norm(c) == norm(n) for c in family["tilesets"][t]["counties"])]
            if not hit:
                avail = ", ".join(sorted(c for t in groups for c in family["tilesets"][t]["counties"]))
                raise SystemExit(f"--{what} '{n}': no such county among the sheets covering this area.\n"
                                 f"Counties here: {avail}")
            ids += hit
        return ids

    pref_ids, excl_ids = resolve(prefer, "prefer"), resolve(exclude, "exclude")
    if set(pref_ids) & set(excl_ids):
        raise SystemExit("A county cannot be both --prefer and --exclude.")
    layers = []
    for ts_id in sorted(groups):
        if ts_id in excl_ids:
            continue
        ts = family["tilesets"][ts_id]
        shapes = [shape(f["geometry"]) for f in groups[ts_id]]
        foot_wgs = unary_union(shapes)
        foot_bng = to_bng(foot_wgs)
        terr_wgs = match_territory(ts["counties"], boundaries) if boundaries else None
        territory = to_bng(terr_wgs) if terr_wgs is not None else None
        if territory is not None:
            core = foot_bng.intersection(territory)
            valid = foot_bng.intersection(territory.buffer(tol_m))
            if core.is_empty or valid.is_empty:         # boundary does not overlap the sheets: do not trust it
                core, valid, territory = foot_bng, foot_bng, None
        else:
            core, valid = foot_bng, foot_bng
        rank = (len(pref_ids) - pref_ids.index(ts_id)) if ts_id in pref_ids else 0
        layers.append(Layer(ts_id, ts["url"], list(ts["counties"]), int(ts.get("max_zoom", 16)),
                            [f["properties"] for f in groups[ts_id]], foot_wgs, core, valid, to_wgs(valid),
                            territory, rank, sheet_shapes=shapes))
    for i, ly in enumerate(layers, 1):
        ly.index = i
    return layers, unmatched


# -- Seam scoring ----------------------------------------------------------------------

def layer_grids(layers: list[Layer], extent, cell_m: float, pad_m: float):
    """Coarse validity and margin grids for each layer over `extent` (BNG), padded by `pad_m`.

    Returns (valid [L,H,W] bool, margin [L,H,W] float32 metres capped at pad_m, origin (x0, y0 = top-left),
    n_pad cells). `valid` = where the layer may supply map; `margin` = distance inside its core
    (own county and sheets), so cells in the boundary-tolerance strip have margin 0. Cell centres are tested.
    """
    l, b, r, t = extent
    n = int(math.ceil(pad_m / cell_m))
    x0, y0 = l - n * cell_m, t + n * cell_m
    W = int(math.ceil((r - l) / cell_m)) + 2 * n
    H = int(math.ceil((t - b) / cell_m)) + 2 * n
    tf = from_origin(x0, y0, cell_m, cell_m)
    window = box(x0, y0 - H * cell_m, x0 + W * cell_m, y0)
    valid = np.zeros((len(layers), H, W), dtype=bool)
    margin = np.zeros((len(layers), H, W), dtype=np.float32)

    def ras(g):
        g = g.intersection(window)
        if g.is_empty:
            return np.zeros((H, W), dtype=bool)
        return rasterize([(mapping(g), 1)], out_shape=(H, W), transform=tf, fill=0, dtype="uint8").astype(bool)

    for i, ly in enumerate(layers):
        v = ras(ly.valid_bng)
        valid[i] = v
        core = v if ly.valid_bng is ly.foot_bng else ras(ly.foot_bng)
        if core.all():
            margin[i] = pad_m
        elif core.any():
            margin[i] = np.minimum(distance_transform_edt(core) * cell_m, pad_m)
    return valid, margin, (x0, y0), n


def coarse_scores(layers: list[Layer], valid, margin, pad_m: float) -> np.ndarray:
    """uint16 score per layer and cell; higher wins, 0 = layer has no map there."""
    out = np.zeros(valid.shape, dtype=np.uint16)
    for i, ly in enumerate(layers):
        lev = np.minimum(LEVELS, (margin[i] / max(pad_m, 1e-9) * LEVELS)).astype(np.int32)
        out[i] = np.where(valid[i], ly.rank * RANK_STEP + 1 + lev, 0).astype(np.uint16)
    return out


def winners(scores: np.ndarray) -> np.ndarray:
    """Index of the winning layer per cell (-1 where no layer has map). Ties go to the lower index."""
    w = scores.argmax(axis=0).astype(np.int16)
    w[scores.max(axis=0) == 0] = -1
    return w


def preview_winners(layers: list[Layer], bbox, cell_m: float, pad_m: float):
    """Winner map over `bbox` (BNG) for display and for planning. Returns (winner [H,W], extent)."""
    valid, margin, (x0, y0), n = layer_grids(layers, bbox, cell_m, pad_m)
    w = winners(coarse_scores(layers, valid, margin, pad_m))
    return w[n:w.shape[0] - n, n:w.shape[1] - n], bbox


def share_of_area(layers: list[Layer], bbox, polygon=None, cell_m: float | None = None, pad_m: float = 3000.0):
    """Fraction of the area each layer supplies, as {tileset: fraction} (and 'none' for no map)."""
    l, b, r, t = bbox
    cell = cell_m or max(10.0, math.sqrt((r - l) * (t - b) / 250_000))
    w, _ = preview_winners(layers, bbox, cell, pad_m)
    if polygon is not None:
        H, W = w.shape
        inside = rasterize([(mapping(polygon), 1)], out_shape=(H, W), transform=from_origin(l, t, cell, cell),
                           fill=0, dtype="uint8").astype(bool)
        w = np.where(inside, w, -2)
    tot = int((w != -2).sum()) or 1
    res = {ly.tileset: float((w == i).sum()) / tot for i, ly in enumerate(layers)}
    res["none"] = float((w == -1).sum()) / tot
    return res
