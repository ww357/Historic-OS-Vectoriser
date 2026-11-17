"""
NLS county-layer index and sheet-outline (feature service) client for step 00.

Why this exists
---------------
Some NLS series are ONE seamless tileset (e.g. six-inch 2nd edition), but others are held as
one tileset PER COUNTY (six-inch 1st edition for England and Wales, the 25-inch series, ...).
To download an area that crosses a county border we need to know

  1. which sheets cover the area  -> NLS's feature service returns every sheet's outline,
     with its county, survey and publication years;
  2. which tileset holds each county -> a lookup table, kept in nls_tilesets.yaml.

nls_tilesets.yaml is a snapshot generated from NLS's own map viewer (and the feature service)
by `python run.py download --refresh-index`. It is committed with the toolkit so downloads do
not depend on the viewer being reachable; refresh it if NLS adds or renames layers.

This module needs only requests + PyYAML, plus the standard library.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

DEFAULT_WFS = "https://geoserver.nls.uk/geoserver/wfs"
VIEWER_URL = "https://maps.nls.uk/geo/explore/"
S3 = "https://mapseries-tilesets.s3.amazonaws.com/"
INDEX_PATH = Path(__file__).with_name("nls_tilesets.yaml")
USER_AGENT = "tile-downloader/1.0"          # deliberately generic

# One entry per family of county tilesets we can currently download.
#   wfs_typename  the feature-service layer holding that series' sheet outlines
#   wfs_filter    keep only sheets whose property is in the list (e.g. England and Wales, not Scotland)
#   tileset_re    which tileset URLs in NLS's viewer belong to the family; group 1 = the county slug
#   skip_slug     tileset slugs to ignore (the 'V' versions are lighter-toned copies of the same map)
FAMILIES: dict[str, dict] = {
    "six_inch_1st_ew": {
        "label": "OS six-inch to the mile, England and Wales, 1st edition (1840s-1880s), county layers",
        "wfs_typename": "nls:Six_Inch_Eng_Scot_Wal_1st",
        "wfs_filter": {"COUNTRY": ["ENG", "WAL"]},
        "tileset_re": r"^os/six-inch-([A-Za-z-]+)$",
        "skip_slug": r"V$",
        # A county's sheets only contain map INSIDE that county (the neighbour's land is left blank
        # paper even where the sheet rectangle extends over it), so each layer is limited to its
        # county polygon. Historic county boundaries c.1860, from NLS's feature service:
        "boundary_typename": "nls:Counties1860_HCT",
    },
}

# Sheet properties worth keeping (everything else the service returns is dropped).
KEEP_PROPS = ("IMAGE", "COUNTY", "COUNTRY", "SHEET", "SHEET_NO", "EDITION", "SUR_STA", "SUR_END",
              "PUB_STA", "PUB_END", "YEAR", "IMAGEURL")


def norm(s: str | None) -> str:
    """Lower-case letters only - the key used to match county names between NLS sources."""
    return re.sub(r"[^a-z]", "", (s or "").lower())


# -- Feature service (sheet outlines) --------------------------------------------

def _get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 120) -> requests.Response:
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"NLS request failed after {retries + 1} tries ({last}): {url}")


def fetch_sheets(typename: str, bbox_lonlat, wfs_filter: dict | None = None,
                 wfs_url: str = DEFAULT_WFS, page: int = 1000) -> list[dict]:
    """GeoJSON features (outline in lon/lat + trimmed properties) for every sheet of a series
    whose outline touches the lon/lat bounding box. Pages through large results."""
    out, start = [], 0
    while True:
        params = dict(service="WFS", version="2.0.0", request="GetFeature", typeNames=typename,
                      outputFormat="application/json", srsName="EPSG:4326",
                      bbox="%.6f,%.6f,%.6f,%.6f,EPSG:4326" % tuple(bbox_lonlat),
                      count=page, startIndex=start, sortBy="IMAGE")
        r = _get(wfs_url, params)
        try:
            feats = r.json().get("features", [])
        except ValueError:
            raise RuntimeError(f"NLS feature service returned something unexpected: {r.text[:200]!r}")
        out += feats
        if len(feats) < page:
            break
        start += page
    keep = []
    for f in out:
        p = f.get("properties") or {}
        if wfs_filter and any(p.get(k) not in v for k, v in wfs_filter.items()):
            continue
        if not f.get("geometry"):
            continue
        keep.append({"type": "Feature", "geometry": f["geometry"],
                     "properties": {k: p.get(k) for k in KEEP_PROPS if k in p}})
    return keep


def fetch_county_boundaries(typename: str, bbox_lonlat, wfs_url: str = DEFAULT_WFS, pad: float = 0.1) -> list[dict]:
    """County polygons (lon/lat) touching the bounding box, clipped to the box plus `pad` degrees
    so the saved file stays small. Properties kept: NAME / COUNTY."""
    from shapely.geometry import box, mapping, shape
    bb = (bbox_lonlat[0] - pad, bbox_lonlat[1] - pad, bbox_lonlat[2] + pad, bbox_lonlat[3] + pad)
    r = _get(wfs_url, dict(service="WFS", version="2.0.0", request="GetFeature", typeNames=typename,
                           outputFormat="application/json", srsName="EPSG:4326",
                           bbox="%.6f,%.6f,%.6f,%.6f,EPSG:4326" % bb, count=500), timeout=300)
    clip = box(*bb)
    out = []
    for f in r.json().get("features", []):
        if not f.get("geometry"):
            continue
        g = shape(f["geometry"]).intersection(clip)
        if g.is_empty:
            continue
        p = f.get("properties") or {}
        out.append({"type": "Feature", "geometry": mapping(g),
                    "properties": {k: p.get(k) for k in ("NAME", "COUNTY") if k in p}})
    return out


# -- Index file -------------------------------------------------------------------

def load_index() -> dict:
    if not INDEX_PATH.exists():
        sys.exit(f"County-layer index not found: {INDEX_PATH}\n"
                 f"Create it with:  python run.py download --refresh-index")
    return yaml.safe_load(INDEX_PATH.read_text(encoding="utf-8"))


def get_family(name: str) -> dict:
    fams = load_index().get("families", {})
    if name not in fams:
        sys.exit(f"County-layer family '{name}' is not in {INDEX_PATH.name} (known: {', '.join(fams) or 'none'}).\n"
                 f"Try:  python run.py download --refresh-index")
    fam = fams[name]
    for key in ("boundary_typename",):          # older snapshots may lack newer fields
        fam.setdefault(key, FAMILIES.get(name, {}).get(key))
    return fam


# -- Building the index (refresh) -------------------------------------------------

def _viewer_layer_script() -> tuple[str, str]:
    """URL and text of NLS's viewer layer-definition script (its folder name is dated, so it is
    discovered from the viewer page rather than hard-coded)."""
    html = _get(VIEWER_URL).text
    m = re.search(r'src="(/geo/scripts/[^"]*explore-layers[^"]*\.js)"', html)
    if not m:
        raise RuntimeError("Could not find the layer script in NLS's viewer page - its layout may have changed.")
    url = "https://maps.nls.uk" + m.group(1)
    return url, _get(url, timeout=300).text


def _viewer_tilesets(js: str) -> dict[str, dict]:
    """{tileset path (e.g. 'os/six-inch-somerset'): {'url', 'bounds'}} for every S3 tile layer in the viewer."""
    found: dict[str, dict] = {}
    for block in re.split(r"\n\s*var\s+", js):
        if not re.match(r"\w+\s*=\s*new ol\.layer\.Tile", block):
            continue
        u = re.search(r'url:\s*"(https://mapseries-tilesets\.s3\.amazonaws\.com/([^"]*?)/\{z\}/\{x\}/\{y\}\.png)"', block)
        if not u:
            continue
        ext = re.search(r"transformExtent\(\[([^\]]*)\]", block)
        found.setdefault(u.group(2), {
            "url": u.group(1),
            "bounds": [float(v) for v in ext.group(1).split(",")] if ext else None})
    return found


def _tileset_details(path: str) -> dict:
    """max zoom and lon/lat bounds from the tileset's own index page."""
    try:
        html = _get(f"{S3}{path}/index.html", retries=2, timeout=30).text
    except Exception:
        return {}
    d = {}
    mz = re.search(r'"maxzoom"\s*:\s*"?(\d+)', html)
    bb = re.search(r'"bounds"\s*:\s*"([^"]+)"', html)
    if mz:
        d["max_zoom"] = int(mz.group(1))
    if bb:
        d["bounds"] = [round(float(v), 6) for v in bb.group(1).split(",")]
    return d


def _wfs_counties(typename: str, wfs_filter: dict | None, wfs_url: str) -> list[str]:
    r = _get(wfs_url, dict(service="WFS", version="2.0.0", request="GetFeature", typeNames=typename,
                           outputFormat="application/json", propertyName="COUNTY,COUNTRY", count=200000),
             timeout=300)
    names = set()
    for f in r.json().get("features", []):
        p = f.get("properties") or {}
        if wfs_filter and any(p.get(k) not in v for k, v in wfs_filter.items()):
            continue
        if p.get("COUNTY"):
            names.add(p["COUNTY"])
    return sorted(names)


def refresh_index(wfs_url: str = DEFAULT_WFS) -> None:
    """Regenerate nls_tilesets.yaml from NLS's viewer and feature service."""
    print("Reading NLS's viewer layer list ...")
    script_url, js = _viewer_layer_script()
    viewer = _viewer_tilesets(js)
    print(f"  {len(viewer):,} tilesets listed in {script_url.rsplit('/', 2)[-2]}/{script_url.rsplit('/', 1)[-1]}")

    families: dict[str, dict] = {}
    for fam, spec in FAMILIES.items():
        print(f"\nFamily '{fam}': {spec['label']}")
        counties = _wfs_counties(spec["wfs_typename"], spec.get("wfs_filter"), wfs_url)
        print(f"  {len(counties)} counties in the feature service")
        slugs = {}
        for path in viewer:
            m = re.match(spec["tileset_re"], path)
            if m and not re.search(spec.get("skip_slug", r"(?!)"), m.group(1)):
                slugs[m.group(1)] = path
        by_norm = {norm(s): s for s in slugs}
        tilesets, unmatched = {}, []
        for county in counties:
            slug = by_norm.get(norm(county))
            if slug is None:   # tolerate small spelling differences, e.g. 'Hampshire and Isle of Wight'
                cands = [s for n, s in by_norm.items() if n.startswith(norm(county)) or norm(county).startswith(n)]
                slug = cands[0] if len(cands) == 1 else None
            if slug is None:
                unmatched.append(county)
                continue
            path = slugs[slug]
            ts = tilesets.setdefault(slug, {"url": viewer[path]["url"], "counties": [],
                                            "bounds": viewer[path]["bounds"]})
            ts["counties"].append(county)
        print(f"  matched {sum(len(t['counties']) for t in tilesets.values())} counties to {len(tilesets)} tilesets"
              + (f"; NOT matched: {unmatched}" if unmatched else ""))
        print("  reading each tileset's own details (max zoom, bounds) ...")
        for slug, ts in sorted(tilesets.items()):
            ts.update({k: v for k, v in _tileset_details(slugs[slug]).items() if v})
            ts.setdefault("max_zoom", 16)
        families[fam] = {"label": spec["label"], "wfs_typename": spec["wfs_typename"],
                         "wfs_filter": spec.get("wfs_filter"), "boundary_typename": spec.get("boundary_typename"),
                         "unmatched_counties": unmatched,
                         "tilesets": dict(sorted(tilesets.items()))}

    doc = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "generated_from": script_url, "families": families}
    header = ("# County-layer index for step 00 (python run.py download --refresh-index).\n"
              "# Generated from NLS's map viewer and feature service - do not edit by hand.\n"
              "# Each family lists the tileset that holds each county. Re-run the refresh if NLS changes layers.\n")
    tmp = INDEX_PATH.with_name(INDEX_PATH.name + ".part")
    tmp.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=140, default_flow_style=None),
                   encoding="utf-8")
    tmp.replace(INDEX_PATH)
    print(f"\nWrote {INDEX_PATH.name}")


if __name__ == "__main__":
    refresh_index()
