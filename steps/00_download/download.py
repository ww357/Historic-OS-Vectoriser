"""
Step 00 - download an area of a historic OS map series from the NLS tile server and
turn it into pipeline-ready "sheets".

The National Library of Scotland publishes georeferenced OS maps as XYZ (Web Mercator)
tile layers. This step downloads the tiles covering an area you
choose, mosaics them, warps them ONCE to British National Grid (EPSG:27700) and writes
GeoTIFFs in exactly the layout the rest of the toolkit already reads:

    data/raw/<SHEET>/<SHEET>.tif            RGB GeoTIFF, BNG, tiled
    data/raw/<SHEET>/source.json            series, zoom, resolution, extent, attribution
    data/map_area_masks/<SHEET>/<SHEET>.tif map-area mask built from tile transparency
    data/areas/<AREA>/aoi.json              the request + per-block progress (resume state)
    data/tiles/<series>/<z>/<x>/<y>.png     shared tile cache (never downloaded twice)

A small area becomes ONE sheet named <AREA>. A large area is split into equal blocks
named <AREA>_r01c01, <AREA>_r01c02, ... (row/column from the north-west), each a normal
sheet for patchify onwards, so memory stays bounded however large the area is. Blocks
overlap slightly (download.overlap_m) so a later merge can clip edge artefacts; each
block's un-padded "core" is recorded in source.json for that.

Crash / stop safe, resumable
----------------------------
  * every tile and every output file is written to a temp name and atomically renamed,
    so an interrupted run never leaves a half-written file that looks finished;
  * a sheet's .tif is written LAST, so "the .tif exists" means "this block is complete";
  * tiles are validated (PNG trailer) before being trusted, retries use exponential
    backoff, and a dead network stops the run early instead of grinding on;
  * a block is only built when every one of its tiles is resolved (downloaded, or
    confirmed absent) - failed tiles are never silently turned into holes;
  * re-running the same command (or just  --area NAME)  picks up where it stopped.

Usage:
    python steps/00_download/download.py --series help          # list the available series
    python steps/00_download/download.py --area Porlock --series 6in_2nd \\
           --bbox 285000 143000 291000 149000                 # BNG metres: minE minN maxE maxN
    python steps/00_download/download.py --area Porlock        # resume / finish a previous run
    python steps/00_download/download.py --key 10k_natgrid     # key image for annotating
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform as warp_points, transform_bounds
from shapely.geometry import box, mapping, shape
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "steps"))   # shared helpers
from common import load_config, rel_to_root   # noqa: E402

BNG, MERC, WGS84 = "EPSG:27700", "EPSG:3857", "EPSG:4326"

# Web Mercator tile maths (256 px tiles, origin top-left of the world square).
EARTH_R = 6378137.0
WORLD_M = 2 * math.pi * EARTH_R            # width of the Web Mercator world in "mercator metres"
TILE_PX = 256
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_TRAILER = b"IEND\xaeB`\x82"           # last 8 bytes of every complete PNG
USER_AGENT = "tile-downloader/1.0"         # deliberately generic
AVG_TILE_KB = 55                           # rough, for the size estimate only

RESAMPLING = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear,
              "cubic": Resampling.cubic, "lanczos": Resampling.lanczos}


class DiskError(RuntimeError):
    """Writing to disk failed (full / permissions) - abort; a re-run resumes."""


# -- Web Mercator helpers ------------------------------------------------------

def merc_tile_m(z: int) -> float:
    return WORLD_M / 2 ** z


def native_ground_res(z: int, lat_deg: float) -> float:
    """Real ground metres per tile pixel at zoom z and latitude (Mercator stretches
    distances by 1/cos(lat), so a 'mercator metre' is NOT a ground metre in Britain)."""
    return merc_tile_m(z) / TILE_PX * math.cos(math.radians(lat_deg))


def tiles_for_merc_bounds(minx, miny, maxx, maxy, z):
    """Inclusive tile index ranges (x0, x1, y0, y1) covering Web Mercator bounds."""
    tm = merc_tile_m(z)
    n = 2 ** z
    x0 = max(0, math.floor((minx + WORLD_M / 2) / tm))
    x1 = min(n - 1, math.floor((maxx + WORLD_M / 2) / tm))
    y0 = max(0, math.floor((WORLD_M / 2 - maxy) / tm))
    y1 = min(n - 1, math.floor((WORLD_M / 2 - miny) / tm))
    return x0, x1, y0, y1


# -- Atomic file helpers -------------------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError as e:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise DiskError(f"could not write {path}: {e}") from e


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def area_lock(path: Path):
    """OS-level exclusive lock (released automatically if the process dies, so a crash
    never leaves a stale lock). Stops two runs writing the same area at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    locked = False
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
    except OSError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK, 36):
            f.close()
            sys.exit(f"Another download for this area is already running ({rel_to_root(path)} is locked).")
        print(f"  (note: file locking unavailable here - {e}; continuing without it)")
    try:
        yield
    finally:
        if locked:
            with contextlib.suppress(OSError):
                if os.name == "nt":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


# -- Tile store: cache + resilient fetching ------------------------------------

class TileStore:
    """On-disk tile cache with resilient parallel fetching.

    cache/<z>/<x>/<y>.png   a tile      cache/<z>/<x>/<y>.none   confirmed absent (outside coverage)
    """

    def __init__(self, cache_dir: Path, url_template: str, scheme: str,
                 workers: int, timeout: float, retries: int):
        self.dir, self.url_t, self.scheme = cache_dir, url_template, scheme
        self.workers, self.timeout, self.retries = workers, timeout, retries
        self._local = threading.local()
        self.streak = 0          # consecutive failed tiles across the WHOLE run (reset by any success)

    # --- cache state
    def png_path(self, z, x, y) -> Path:
        return self.dir / str(z) / str(x) / f"{y}.png"

    def none_path(self, z, x, y) -> Path:
        return self.dir / str(z) / str(x) / f"{y}.none"

    def state(self, z, x, y) -> str | None:
        """'ok' (valid cached tile), 'missing' (confirmed absent) or None (needs fetching)."""
        p = self.png_path(z, x, y)
        if p.exists():
            if self._tail_ok(p):
                return "ok"
            with contextlib.suppress(OSError):      # corrupt/truncated leftovers: refetch
                p.unlink()
        if self.none_path(z, x, y).exists():
            return "missing"
        return None

    @staticmethod
    def _tail_ok(p: Path) -> bool:
        try:
            size = p.stat().st_size
            if size < 67:
                return False
            with open(p, "rb") as f:
                f.seek(-8, os.SEEK_END)
                return f.read(8) == PNG_TRAILER
        except OSError:
            return False

    def forget_missing(self, tiles) -> int:
        n = 0
        for z, x, y in tiles:
            p = self.none_path(z, x, y)
            if p.exists():
                with contextlib.suppress(OSError):
                    p.unlink()
                    n += 1
        return n

    # --- fetching
    def _url(self, z, x, y) -> str:
        yy = (2 ** z - 1 - y) if self.scheme == "tms" else y
        return (self.url_t.replace("{z}", str(z)).replace("{x}", str(x))
                .replace("{-y}", str(2 ** z - 1 - y)).replace("{y}", str(yy)))

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            self._local.s = s
        return s

    def fetch(self, tile, stop: threading.Event) -> str:
        """Fetch one tile. Returns 'ok' | 'missing' | 'failed' | 'stopped'."""
        z, x, y = tile
        url = self._url(z, x, y)
        for attempt in range(self.retries + 1):
            if stop.is_set():
                return "stopped"
            wait = min(60.0, 2.0 ** attempt) * (0.75 + 0.5 * random.random())
            try:
                r = self._session().get(url, timeout=(10, self.timeout))
                code = r.status_code
                if code == 200:
                    data = r.content
                    clen = r.headers.get("Content-Length")
                    if not (data.startswith(PNG_SIGNATURE) and data.endswith(PNG_TRAILER)) \
                            or (clen and clen.isdigit() and int(clen) != len(data)):
                        raise ValueError("truncated or non-PNG response")
                    atomic_write_bytes(self.png_path(z, x, y), data)
                    return "ok"
                if code in (403, 404):             # S3: key does not exist -> outside coverage
                    atomic_write_bytes(self.none_path(z, x, y), b"")
                    return "missing"
                if code == 429 or code >= 500:
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        wait = max(wait, float(ra))
                    raise ValueError(f"HTTP {code}")
                return "failed"                    # other 4xx: retrying will not help
            except DiskError:
                raise
            except (requests.RequestException, ValueError):
                if attempt == self.retries:
                    return "failed"
                stop.wait(wait)
        return "failed"

    def ensure(self, tiles, desc: str, abort_after: int = 20) -> tuple[int, list, bool]:
        """Fetch every tile that is not yet cached.
        Returns (n_fetched, unresolved_tiles, aborted). `aborted` is True when the network
        looked dead (abort_after consecutive failures) and the run should stop."""
        todo = [t for t in tiles if self.state(*t) is None]
        if not todo:
            return 0, [], False
        stop = threading.Event()
        fetched, aborted = 0, False
        ex = ThreadPoolExecutor(max_workers=self.workers)
        futs = {ex.submit(self.fetch, t, stop): t for t in todo}
        try:
            for f in tqdm(as_completed(futs), total=len(futs), desc=desc, unit="tile", leave=False):
                res = f.result()
                if res == "failed":
                    self.streak += 1
                    if self.streak >= abort_after and not aborted:
                        aborted = True
                        stop.set()
                        print(f"\n  {self.streak} tiles in a row failed - network or server problem. "
                              f"Stopping; progress is saved, re-run to continue.")
                elif res in ("ok", "missing"):
                    fetched += 1
                    self.streak = 0
        except BaseException:
            stop.set()                    # Ctrl-C or error: let in-flight requests wind down quickly
            raise
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
        unresolved = [t for t in todo if self.state(*t) is None]
        return fetched, unresolved, aborted


# -- AOI + block planning ------------------------------------------------------

def read_polygon(path: Path):
    import geopandas as gpd
    g = gpd.read_file(path)
    if g.empty:
        sys.exit(f"No features in {path}")
    if g.crs is None:
        print(f"  Warning: {path.name} has no CRS - assuming British National Grid (EPSG:27700).")
        g = g.set_crs(BNG)
    g = g.to_crs(BNG)
    return g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union


def centre_latitude(bbox) -> float:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    _, lats = warp_points(BNG, WGS84, [cx], [cy])
    return float(lats[0])


def make_blocks(area: str, bbox, polygon, block_m: float, overlap_m: float, res: float) -> list[dict]:
    """Split the AOI bbox into equal blocks no larger than block_m (a single block if it fits)."""
    minx, miny, maxx, maxy = bbox
    w, h = maxx - minx, maxy - miny
    nx, ny = max(1, math.ceil(w / block_m - 1e-9)), max(1, math.ceil(h / block_m - 1e-9))
    single = nx == 1 and ny == 1
    ov = 0.0 if single else overlap_m
    dw, dh = w / nx, h / ny
    digits = max(2, len(str(max(nx, ny))))
    blocks = []
    for r in range(ny):
        for c in range(nx):
            core = (minx + c * dw, maxy - (r + 1) * dh, minx + (c + 1) * dw, maxy - r * dh)
            if polygon is not None and not box(*core).intersects(polygon):
                continue
            name = area if single else f"{area}_r{r + 1:0{digits}d}c{c + 1:0{digits}d}"
            # Snap the padded extent to a GLOBAL pixel grid (multiples of res), so neighbouring
            # blocks share exactly the same pixel lattice and can be merged without resampling.
            ix0, ix1 = math.floor((core[0] - ov) / res), math.ceil((core[2] + ov) / res)
            iy0, iy1 = math.floor((core[1] - ov) / res), math.ceil((core[3] + ov) / res)
            ext = (ix0 * res, iy0 * res, ix1 * res, iy1 * res)
            blocks.append({"name": name, "row": r + 1, "col": c + 1, "rows": ny, "cols": nx,
                           "core": [round(v, 3) for v in core], "extent": [round(v, 6) for v in ext],
                           "width": ix1 - ix0, "height": iy1 - iy0})
    return blocks


def block_tiles(blk: dict, z: int) -> list[tuple[int, int, int]]:
    """Tiles (z, x, y) covering a block, with a small margin for resampling."""
    l, b, r, t = blk["extent"]
    px = merc_tile_m(z) / TILE_PX
    mb = transform_bounds(BNG, MERC, l, b, r, t, densify_pts=21)
    x0, x1, y0, y1 = tiles_for_merc_bounds(mb[0] - 2 * px, mb[1] - 2 * px, mb[2] + 2 * px, mb[3] + 2 * px, z)
    return [(z, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


# -- Building a block ----------------------------------------------------------

def build_block(store: TileStore, blk: dict, z: int, res: float, polygon, cfg: dict,
                series_id: str, series: dict, request: dict) -> dict:
    """Mosaic + warp one block and write its sheet. Returns a manifest entry."""
    paths, dcfg = cfg["paths"], cfg.get("download", {})
    name = blk["name"]
    l, b, r, t = blk["extent"]
    w, h = blk["width"], blk["height"]
    dst_tf = from_origin(l, t, res, res)

    tiles = block_tiles(blk, z)
    xs, ys = [x for _, x, _ in tiles], [y for _, _, y in tiles]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sw, sh = (x1 - x0 + 1) * TILE_PX, (y1 - y0 + 1) * TILE_PX

    src = np.zeros((4, sh, sw), dtype=np.uint8)      # RGBA, transparent where no tile
    n_real = 0
    for _, x, y in tiles:
        p = store.png_path(z, x, y)
        if not p.exists():
            continue                                  # confirmed-absent tile -> stays transparent
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGBA"))
        except Exception:
            with contextlib.suppress(OSError):        # corrupt despite a valid trailer: drop it
                p.unlink()
            raise RuntimeError(f"cached tile {z}/{x}/{y} could not be decoded and was deleted - re-run")
        if arr.shape[0] != TILE_PX or arr.shape[1] != TILE_PX:
            raise RuntimeError(f"tile {z}/{x}/{y} is {arr.shape[1]}x{arr.shape[0]}px, expected {TILE_PX}")
        oy, ox = (y - y0) * TILE_PX, (x - x0) * TILE_PX
        src[:, oy:oy + TILE_PX, ox:ox + TILE_PX] = arr.transpose(2, 0, 1)
        n_real += 1
    if n_real == 0:
        return {"status": "empty", "reason": "no tiles exist here", "tiles": len(tiles)}

    tm = merc_tile_m(z)
    src_tf = from_origin(x0 * tm - WORLD_M / 2, WORLD_M / 2 - y0 * tm, tm / TILE_PX, tm / TILE_PX)
    dst = np.zeros((4, h, w), dtype=np.uint8)
    reproject(source=src, destination=dst, src_transform=src_tf, src_crs=MERC,
              dst_transform=dst_tf, dst_crs=BNG,
              resampling=RESAMPLING[dcfg.get("resampling", "bilinear")],
              num_threads=min(4, os.cpu_count() or 1))
    del src

    mask = dst[3] >= 128
    if polygon is not None:
        mask &= rasterize([(mapping(polygon), 1)], out_shape=(h, w), transform=dst_tf,
                          fill=0, dtype="uint8").astype(bool)
    coverage = float(mask.mean())
    if coverage < float(dcfg.get("min_coverage", 0.005)):
        return {"status": "empty", "reason": f"only {100 * coverage:.2f}% of the block has map",
                "tiles": len(tiles), "coverage": round(coverage, 4)}

    rgb = dst[:3]
    rgb[:, ~mask] = 255                                # white outside the map (= patchify pad_value)
    del dst

    raw_dir = ROOT / paths["raw"] / name
    mask_dir = ROOT / paths["masks"] / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(raw_dir.glob("*.part")) + list(mask_dir.glob("*.part")):
        with contextlib.suppress(OSError):
            stale.unlink()

    def write_tif(path: Path, data: np.ndarray, compress: str, **extra) -> None:
        tmp = path.with_name(path.name + ".part")
        profile = dict(driver="GTiff", height=h, width=w, count=data.shape[0], dtype="uint8",
                       crs=BNG, transform=dst_tf, tiled=True, blockxsize=512, blockysize=512,
                       compress=compress, BIGTIFF="IF_SAFER", **extra)
        try:
            with rasterio.open(tmp, "w", **profile) as out:
                out.write(data)
            os.replace(tmp, path)
        except OSError as e:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise DiskError(f"could not write {path}: {e}") from e

    # Order matters for crash safety: mask, then provenance, then the raw .tif LAST.
    write_tif(mask_dir / f"{name}.tif", (mask.astype(np.uint8) * 255)[None], "deflate")
    atomic_write_json(raw_dir / "source.json", {
        "sheet": name, "area": request["area"], "series": series_id, "series_label": series.get("label"),
        "edition": series.get("edition"), "attribution": series.get("attribution"),
        "tile_url": request["tile_url"], "zoom": z, "resolution_m": res, "crs": BNG,
        "core_bng": blk["core"], "extent_bng": blk["extent"],
        "block": {k: blk[k] for k in ("row", "col", "rows", "cols")},
        "coverage": round(coverage, 4), "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    write_tif(raw_dir / f"{name}.tif", rgb, "lzw", predictor=2)
    return {"status": "done", "coverage": round(coverage, 4), "tiles": len(tiles),
            "size_px": [w, h]}


# -- Key images ----------------------------------------------------------------

def download_key(source: str, cfg: dict, args) -> None:
    """Stitch an NLS key tileset (a non-georeferenced pixel image) into one PNG."""
    series = (cfg.get("series") or {}).get(source)
    base = (series or {}).get("key_tiles") if series else source
    name = source if series else re.sub(r"[^A-Za-z0-9_.-]", "_", source.rstrip("/").split("/")[-1])
    if not base:
        sys.exit(f"Series '{source}' has no key_tiles URL in config.yaml. Pass a key URL directly, e.g.\n"
                 f"  --key https://mapseries-tilesets.s3.amazonaws.com/os/10k_key")
    base = base.rstrip("/")
    html = requests.get(f"{base}/openlayers.html", timeout=30, headers={"User-Agent": USER_AGENT})
    if html.status_code != 200:
        sys.exit(f"Could not read {base}/openlayers.html (HTTP {html.status_code})")

    def grab(var, cast):
        m = re.search(rf"var\s+{var}\s*=\s*\[?([^\];]+)\]?\s*;", html.text)
        if not m:
            sys.exit(f"Not a pixel-tile key set ('{var}' not found in {base}/openlayers.html)")
        vals = [cast(v) for v in m.group(1).split(",")]
        return vals if len(vals) > 1 else vals[0]

    ext, zmax, res = grab("tileExtent", float), int(grab("mapMaxZoom", float)), float(grab("mapMaxResolution", float))
    width, height = round((ext[2] - ext[0]) / res), round((ext[3] - ext[1]) / res)
    cols, rows = math.ceil(width / TILE_PX), math.ceil(height / TILE_PX)
    print(f"Key '{name}': {width} x {height}px  ({cols} x {rows} tiles at zoom {zmax})")

    dcfg = cfg.get("download", {})
    store = TileStore(ROOT / cfg["paths"]["tiles"] / "_keys" / name, f"{base}/{{z}}/{{x}}/{{y}}.png", "xyz",
                      int(args.workers or dcfg.get("workers", 4)), float(dcfg.get("timeout", 30)),
                      int(dcfg.get("retries", 5)))
    tiles = [(zmax, x, y) for x in range(cols) for y in range(rows)]
    _, failed, _ = store.ensure(tiles, "key tiles")
    if failed:
        sys.exit(f"{len(failed)} key tiles could not be fetched - re-run to retry.")
    canvas = Image.new("RGBA", (cols * TILE_PX, rows * TILE_PX), (255, 255, 255, 255))
    for _, x, y in tiles:
        p = store.png_path(zmax, x, y)
        if p.exists():
            with Image.open(p) as im:
                canvas.paste(im.convert("RGBA"), (x * TILE_PX, y * TILE_PX))
    out = ROOT / cfg["paths"].get("keys", "data/keys/") / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    canvas.crop((0, 0, width, height)).convert("RGB").save(tmp, format="PNG")
    os.replace(tmp, out)
    print(f"Saved -> {rel_to_root(out)}")


# -- CLI -----------------------------------------------------------------------

def list_series(cfg: dict) -> None:
    series = cfg.get("series") or {}
    zdef = int((cfg.get("download") or {}).get("zoom", 16))
    print(f"\n  {'id':<18} {'zoom':<7} {'m/px @52N':<10} label")
    for sid, s in series.items():
        zmax = int(s.get("max_zoom", 18))
        z = min(zdef, zmax)
        print(f"  {sid:<18} {s.get('min_zoom', 1)}-{zmax:<5} {native_ground_res(z, 52):<10.2f} {s.get('label', '')}"
              f"{'' if z == zdef else f'   (default zoom {zdef} capped to {z})'}")
    print(f"\n  Default zoom {zdef} (download.zoom). m/px is the layer's real ground resolution at that\n"
          f"  zoom at ~52N (central England/Wales); a little finer further south. Higher zoom = finer\n"
          f"  detail, 4x tiles per step.\n")


def build_request(args, cfg: dict, prior: dict | None) -> dict:
    series_map = cfg.get("series") or {}
    dcfg = cfg.get("download", {})
    have_aoi = any([args.bbox, args.centre, args.latlon_bbox, args.polygon])

    if not have_aoi:
        if not prior:
            sys.exit("No area given. Use one of --bbox, --centre/--size-km, --latlon-bbox or --polygon\n"
                     "(or re-run an existing area with just --area NAME).")
        req = dict(prior["request"])
        for k in ("series", "zoom", "resolution"):          # a resume must not change what was asked for
            given = getattr(args, k)
            if given is not None and given != req[k] and not args.rebuild:
                sys.exit(f"--{k} {given} differs from this area's saved request ({req[k]}). "
                         f"Use a new --area name, or --rebuild to redo it.")
        return req

    if not args.series:
        sys.exit("--series is required for a new area. See:  python run.py download --series help")
    if args.series not in series_map:
        sys.exit(f"Unknown series '{args.series}'. Defined in config.yaml: {', '.join(series_map) or '(none)'}")
    s = series_map[args.series]
    url = args.tile_url or s.get("tile_url")
    if not url:
        sys.exit(f"Series '{args.series}' has no tile_url in config.yaml")

    polygon = None
    if args.polygon:
        polygon = read_polygon(Path(args.polygon))
        bbox = list(polygon.bounds)
    elif args.centre:
        half = float(args.size_km) * 500.0
        bbox = [args.centre[0] - half, args.centre[1] - half, args.centre[0] + half, args.centre[1] + half]
    elif args.latlon_bbox:
        lo0, la0, lo1, la1 = args.latlon_bbox
        bbox = list(transform_bounds(WGS84, BNG, lo0, la0, lo1, la1, densify_pts=21))
    else:
        bbox = list(args.bbox)
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        sys.exit("Area is empty: need minE < maxE and minN < maxN.")
    if not (-100_000 < bbox[0] < 800_000 and -100_000 < bbox[1] < 1_400_000 and
            bbox[2] < 800_000 and bbox[3] < 1_400_000):
        sys.exit(f"Area {bbox} is outside British National Grid range (eastings 0-700000, northings 0-1300000).")

    zmin, zmax = int(s.get("min_zoom", 1)), int(s.get("max_zoom", 18))
    z = int(args.zoom if args.zoom is not None else dcfg.get("zoom", 16))
    if z > zmax:
        print(f"  Zoom {z} is above this series' maximum ({zmax}); using {zmax}.")
        z = zmax
    if z < zmin:
        sys.exit(f"Zoom {z} is below this series' minimum ({zmin}).")

    lat = centre_latitude(bbox)
    native = native_ground_res(z, lat)
    res_arg = args.resolution if args.resolution is not None else dcfg.get("resolution", "native")
    res = round(native, 2) if str(res_arg) == "native" else float(res_arg)

    sb = s.get("bounds")
    if sb:
        wb = transform_bounds(BNG, WGS84, *bbox, densify_pts=21)
        if wb[2] < sb[0] or wb[0] > sb[2] or wb[3] < sb[1] or wb[1] > sb[3]:
            sys.exit(f"The area lies outside the bounding box of series '{args.series}' "
                     f"(lon/lat {sb}). Check the series and coordinates.")

    return {
        "area": args.area, "series": args.series, "tile_url": url, "scheme": s.get("scheme", "xyz"),
        "zoom": z, "resolution": res, "native_resolution": round(native, 3), "centre_lat": round(lat, 4),
        "bbox_bng": [round(v, 3) for v in bbox],
        "polygon_geojson": mapping(polygon) if polygon is not None else None,
        "block_px": int(args.block_px or dcfg.get("block_px", 8000)),
        "overlap_m": float(args.overlap_m if args.overlap_m is not None else dcfg.get("overlap_m", 100)),
    }


def same_request(a: dict, b: dict) -> bool:
    keys = ("series", "zoom", "resolution", "bbox_bng", "polygon_geojson", "block_px", "overlap_m", "tile_url")
    return all(a.get(k) == b.get(k) for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download an area of an NLS historic OS series as pipeline-ready sheets.")
    ap.add_argument("--area", help="Name for this download (letters, digits, _ and -). Becomes the sheet name.")
    ap.add_argument("--series", help="Series id from config.yaml. Use  --series help  to list them.")
    g = ap.add_argument_group("area (choose one)")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("MINE", "MINN", "MAXE", "MAXN"),
                   help="British National Grid metres.")
    g.add_argument("--centre", nargs=2, type=float, metavar=("E", "N"), help="BNG centre; use with --size-km.")
    g.add_argument("--size-km", type=float, default=5.0, help="Side of the square around --centre (default 5).")
    g.add_argument("--latlon-bbox", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    g.add_argument("--polygon", help="Vector file (GeoJSON/GPKG/SHP) outlining the area; maps outside it are masked out.")
    ap.add_argument("--zoom", type=int, default=None, help="Tile zoom (default: download.zoom = 16).")
    ap.add_argument("--resolution", type=float, default=None,
                    help="Output metres per pixel (default: the tiles' native ground resolution).")
    ap.add_argument("--block-px", type=int, default=None, help="Max block side in pixels (default 8000).")
    ap.add_argument("--overlap-m", type=float, default=None, help="Overlap between blocks in metres (default 100).")
    ap.add_argument("--tile-url", default=None, help="Override the series' tile URL ({z}/{x}/{y}).")
    ap.add_argument("--workers", type=int, default=None, help="Parallel tile downloads (default 4).")
    ap.add_argument("--retries", type=int, default=None, help="Retries per tile with backoff (default 5).")
    ap.add_argument("--dry-run", action="store_true", help="Show the plan (blocks, tiles, size) and stop.")
    ap.add_argument("--yes", action="store_true", help="Skip the large-download confirmation.")
    ap.add_argument("--rebuild", action="store_true", help="Rebuild sheets that already exist (tiles stay cached).")
    ap.add_argument("--recheck-missing", action="store_true",
                    help="Forget tiles previously recorded as absent and ask the server again.")
    ap.add_argument("--list-series", action="store_true", help="List the series defined in config.yaml.")
    ap.add_argument("--key", metavar="SERIES_OR_URL",
                    help="Download the key image for a series (or a key tileset URL) to data/keys/.")
    args = ap.parse_args()

    cfg = load_config()
    cfg["paths"].setdefault("tiles", "data/tiles/")
    cfg["paths"].setdefault("areas", "data/areas/")
    cfg["paths"].setdefault("keys", "data/keys/")
    dcfg = cfg.get("download", {})

    if args.list_series or (args.series and args.series.lower() in ("help", "list")):
        list_series(cfg)
        return
    if args.key:
        download_key(args.key, cfg, args)
        return
    if not args.area:
        ap.error("--area is required (or use --list-series / --key)")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]*", args.area):
        sys.exit("--area may only contain letters, digits, _ and - (it becomes a folder and sheet name).")

    area_dir = ROOT / cfg["paths"]["areas"] / args.area
    manifest_path = area_dir / "aoi.json"
    prior = read_json(manifest_path)
    request = build_request(args, cfg, prior)
    if prior and not same_request(prior["request"], request) and not args.rebuild:
        sys.exit(f"Area '{args.area}' was already planned with different settings "
                 f"({rel_to_root(manifest_path)}).\nUse a new --area name, or add --rebuild to redo it.")

    series = cfg["series"][request["series"]]
    z, res = request["zoom"], request["resolution"]
    polygon = shape(request["polygon_geojson"]) if request.get("polygon_geojson") else None
    block_m = request["block_px"] * min(res, request["native_resolution"])
    blocks = make_blocks(args.area, request["bbox_bng"], polygon, block_m, request["overlap_m"], res)
    if not blocks:
        sys.exit("The polygon does not intersect any block - nothing to download.")

    all_tiles = sorted({t for b in blocks for t in block_tiles(b, z)})
    # Cache folder follows the URL actually used (saved in the request), never the command line:
    # a non-default tile URL gets its own folder so different sources can never mix.
    store_dir = ROOT / cfg["paths"]["tiles"] / (
        request["series"] if request["tile_url"] == series.get("tile_url")
        else "custom_" + hashlib.sha1(request["tile_url"].encode()).hexdigest()[:8])
    store = TileStore(store_dir, request["tile_url"], request["scheme"],
                      int(args.workers or dcfg.get("workers", 4)), float(dcfg.get("timeout", 30)),
                      int(args.retries if args.retries is not None else dcfg.get("retries", 5)))
    if args.recheck_missing:
        print(f"Forgot {store.forget_missing(all_tiles)} tiles recorded as absent.")
    cached = sum(1 for t in all_tiles if store.state(*t) is not None)
    to_fetch = len(all_tiles) - cached

    # -- plan summary
    bx = request["bbox_bng"]
    print(f"\nArea       : {args.area}   ({(bx[2] - bx[0]) / 1000:.2f} x {(bx[3] - bx[1]) / 1000:.2f} km)")
    print(f"Series     : {request['series']}  -  {series.get('label', '')}")
    print(f"Zoom       : {z}   tile pixel = {request['native_resolution']:.2f} m on the ground at lat "
          f"{request['centre_lat']:.1f}  ->  output {res} m/px")
    print(f"Sheets     : {len(blocks)}  ({'single sheet' if len(blocks) == 1 else 'blocks of <= %.1f km, %.0f m overlap' % (block_m / 1000, request['overlap_m'])})"
          f"  ->  {blocks[0]['name']}" + (f" ... {blocks[-1]['name']}" if len(blocks) > 1 else ""))
    print(f"Tiles      : {len(all_tiles):,} total, {cached:,} already cached, {to_fetch:,} to download "
          f"(~{to_fetch * AVG_TILE_KB / 1024:.0f} MB)")
    if series.get("attribution"):
        print(f"Attribution: {series['attribution']}")
    if args.dry_run:
        print("\n--dry-run: nothing downloaded.")
        return
    limit = int(dcfg.get("confirm_over_tiles", 20000))
    if to_fetch > limit and not args.yes:
        sys.exit(f"\n{to_fetch:,} tiles is a large download (limit {limit:,}). Add --yes to proceed.")

    # -- run
    manifest = prior if (prior and same_request(prior["request"], request)) else {"blocks": {}}
    manifest.update({"area": args.area, "request": request,
                     "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    manifest.setdefault("created_utc", manifest["updated_utc"])
    manifest.setdefault("blocks", {})

    summary = {"built": 0, "existing": 0, "empty": 0, "incomplete": 0}
    try:
        with area_lock(area_dir / ".lock"):
            atomic_write_json(manifest_path, manifest)
            for i, blk in enumerate(blocks, 1):
                name = blk["name"]
                tif = ROOT / cfg["paths"]["raw"] / name / f"{name}.tif"
                old = manifest["blocks"].get(name, {})
                if not args.rebuild:
                    if tif.exists():
                        summary["existing"] += 1
                        continue
                    if old.get("status") == "empty":
                        summary["empty"] += 1
                        continue
                print(f"\n[{i}/{len(blocks)}] {name}  ({blk['width']} x {blk['height']} px)")
                _, failed, aborted = store.ensure(block_tiles(blk, z), "  tiles")
                if failed:
                    print(f"  {len(failed)} tile(s) could not be fetched - block NOT built (no silent holes). "
                          f"Re-run to retry.")
                    manifest["blocks"][name] = {"status": "incomplete", "failed_tiles": len(failed),
                                                "core": blk["core"], "extent": blk["extent"]}
                    summary["incomplete"] += 1
                else:
                    print("  mosaic + warp to BNG ...")
                    entry = build_block(store, blk, z, res, polygon, cfg, request["series"], series, request)
                    entry.update({"core": blk["core"], "extent": blk["extent"]})
                    manifest["blocks"][name] = entry
                    if entry["status"] == "done":
                        summary["built"] += 1
                        print(f"  -> {rel_to_root(tif)}  ({100 * entry['coverage']:.0f}% map)")
                    else:
                        summary["empty"] += 1
                        print(f"  skipped: {entry['reason']}")
                manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                atomic_write_json(manifest_path, manifest)
                if aborted:                    # network looks dead: do not grind through the other blocks
                    break
    except KeyboardInterrupt:
        print("\n\nInterrupted. Everything finished so far is saved; re-run the same command to continue:\n"
              f"  python run.py download --area {args.area}")
        sys.exit(130)
    except DiskError as e:
        sys.exit(f"\nDisk problem: {e}\nFree some space or fix permissions, then re-run to continue.")

    print(f"\nDone: {summary['built']} built, {summary['existing']} already present, "
          f"{summary['empty']} empty, {summary['incomplete']} incomplete.")
    done = [b["name"] for b in blocks if (ROOT / cfg["paths"]["raw"] / b["name"] / f"{b['name']}.tif").exists()]
    if summary["incomplete"]:
        print(f"Some tiles failed. Re-run to finish:  python run.py download --area {args.area}")
    if done:
        first = done[0]
        print(f"\nNext step (per sheet):\n  python run.py patchify --sheet {first} --mask\n"
              f"  (the map-area mask was made for you from the tile coverage"
              f"{' and your polygon' if polygon is not None else ''})")
        if len(done) > 1:
            print(f"\n{len(done)} sheets: " + ", ".join(done[:4]) + (" ..." if len(done) > 4 else ""))
            print(f"Annotate one or two, train, then run the later steps on each sheet in turn.")
    sys.exit(1 if summary["incomplete"] else 0)


if __name__ == "__main__":
    main()
