"""
Interactive county planner for step 00  (python run.py download ... --plan).

Opens a window over your area showing, for a county-layer series (e.g. 6in_1st):
  * the sheet outlines NLS holds, coloured by county,
  * your area (dashed),
  * a live preview of which county supplies each part of it (the same rule the download uses).

Click a county in the list on the right to change how it is used. Each click cycles
        automatic  ->  PREFERRED  ->  EXCLUDED  ->  automatic ...
  automatic  each county supplies its own territory, so seams follow the county boundaries
  PREFERRED  wins wherever it has map (its county plus a thin strip past the boundary)
  EXCLUDED   not used at all - its part of your area is left white and masked out

Enter  accept and continue with the download      Esc / close window  cancel (nothing is downloaded)
R      reset every county to automatic

Same style and backend handling as draw_mask.py / pick_patches.py (TkAgg works under WSLg).
"""

from __future__ import annotations

import dataclasses
import math
import sys

import numpy as np
from shapely.geometry import shape

import counties

PALETTE = ["#1f77b4", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d",
           "#17becf", "#8c564b"]
STATE_NEXT = {"auto": "prefer", "prefer": "exclude", "exclude": "auto"}


class PlanState:
    """Which county is automatic / preferred / excluded, and the winner preview that follows."""

    def __init__(self, layers, bbox, polygon, prefer, exclude, cap_m):
        self.layers, self.bbox, self.polygon, self.cap_m = layers, bbox, polygon, cap_m
        self.state = {ly.tileset: "auto" for ly in layers}
        self.prefer_order = []
        for name in prefer:
            for ly in layers:
                if counties.norm(name) in (counties.norm(ly.tileset), *[counties.norm(c) for c in ly.counties]):
                    self.state[ly.tileset] = "prefer"
                    self.prefer_order.append(ly.tileset)
        for name in exclude:
            for ly in layers:
                if counties.norm(name) in (counties.norm(ly.tileset), *[counties.norm(c) for c in ly.counties]):
                    self.state[ly.tileset] = "exclude"
        l, b, r, t = bbox
        self.cell = max(r - l, t - b) / 260.0
        self.colour = {ly.tileset: PALETTE[i % len(PALETTE)] for i, ly in enumerate(layers)}

    def cycle(self, i: int) -> None:
        ts = self.layers[i].tileset
        new = STATE_NEXT[self.state[ts]]
        if self.state[ts] == "prefer":
            self.prefer_order.remove(ts)
        if new == "prefer":
            self.prefer_order.append(ts)
        self.state[ts] = new

    def reset(self) -> None:
        for k in self.state:
            self.state[k] = "auto"
        self.prefer_order = []

    def active_layers(self):
        out = []
        for ly in self.layers:
            if self.state[ly.tileset] == "exclude":
                continue
            rank = len(self.prefer_order) - self.prefer_order.index(ly.tileset) if ly.tileset in self.prefer_order else 0
            out.append(dataclasses.replace(ly, rank=rank))
        return out

    def preview(self):
        """RGB image of who supplies each cell, plus the shares by tileset."""
        act = self.active_layers()
        h = np.full((1, 1, 3), 0.92)
        if not act:
            return h, {}
        w, _ = counties.preview_winners(act, self.bbox, self.cell, self.cap_m)
        img = np.full(w.shape + (3,), 0.92, dtype=float)
        from matplotlib.colors import to_rgb
        for i, ly in enumerate(act):
            img[w == i] = to_rgb(self.colour[ly.tileset])
        tot = max(1, w.size)
        return img, {ly.tileset: float((w == i).sum()) / tot for i, ly in enumerate(act)} | {"none": float((w == -1).sum()) / tot}

    def result(self):
        exclude = [ly.tileset for ly in self.layers if self.state[ly.tileset] == "exclude"]
        return list(self.prefer_order), exclude


def choose_layers(fam: dict, features: list, boundaries: list, bbox, polygon, prefer, exclude, cap_m: float,
                  tol_m: float = counties.DEFAULT_TOL_M, _script=None, _save_png: str | None = None):
    """Show the planner and return (prefer, exclude) as lists of tileset ids.

    `_script` / `_save_png` are for automated tests: a list of ('click', row) / ('enter',) steps applied
    without opening a window, and a path to save the final picture to.
    """
    import matplotlib
    headless = _script is not None
    try:
        matplotlib.use("Agg" if headless else "TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    layers, _ = counties.build_layers(fam, features, boundaries=boundaries, tol_m=tol_m)
    if len(layers) < 2:
        print("  Only one county layer covers this area, so there is nothing to choose - continuing.")
        return list(prefer), list(exclude)
    ps = PlanState(layers, bbox, polygon, prefer, exclude, cap_m)

    fig = plt.figure(figsize=(12.5, 7.2))
    ax = fig.add_axes([0.05, 0.15, 0.62, 0.77])
    lg = fig.add_axes([0.69, 0.15, 0.30, 0.77])
    lg.set_axis_off()
    l, b, r, t = bbox
    img, _ = ps.preview()
    im = ax.imshow(img, extent=[l, r, b, t], origin="upper", interpolation="nearest", alpha=0.55, zorder=1)
    for ly in layers:                       # sheet outlines (thin) and each county's own boundary (thick)
        for shp in ly.sheet_shapes:
            g = counties.to_bng(shp)
            for poly in (g.geoms if hasattr(g, "geoms") else [g]):
                x, y = poly.exterior.xy
                ax.plot(x, y, color=ps.colour[ly.tileset], lw=0.6, alpha=0.6, zorder=2)
        if ly.territory_bng is not None:
            terr = ly.territory_bng.intersection(counties.box(l - 6000, b - 6000, r + 6000, t + 6000))
            for poly in (terr.geoms if hasattr(terr, "geoms") else [terr]):
                if poly.geom_type == "Polygon":
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=ps.colour[ly.tileset], lw=2.2, alpha=0.95, zorder=2.5)
    if polygon is not None:
        for poly in (polygon.geoms if hasattr(polygon, "geoms") else [polygon]):
            x, y = poly.exterior.xy
            ax.plot(x, y, "k--", lw=1.6, zorder=3)
    else:
        ax.plot([l, r, r, l, l], [b, b, t, t, b], "k--", lw=1.6, zorder=3)
    pad = 0.08 * max(r - l, t - b)
    ax.set_xlim(l - pad, r + pad)
    ax.set_ylim(b - pad, t + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, British National Grid)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Who supplies each part of your area (dashed outline)   thick = county boundary, thin = sheet outlines", fontsize=11)
    fig.suptitle("County planner  -  click a county on the right to change how it is used", fontsize=12)

    rows = []
    foot = fig.text(0.05, 0.012, "", fontsize=9)

    def redraw():
        img2, share = ps.preview()
        im.set_data(img2)
        im.set_extent([l, r, b, t])
        for artist in rows:
            artist.remove()
        rows.clear()
        n = len(layers)
        lg.set_xlim(0, 1)
        lg.set_ylim(n + 0.5, -1.2)
        for x, s in ((0.0, "county"), (0.58, "sheets"), (0.76, "supplies")):
            rows.append(lg.text(x, -0.8, s, fontsize=9, weight="bold", va="center"))
        for i, ly in enumerate(layers):
            st = ps.state[ly.tileset]
            colour = ps.colour[ly.tileset]
            rows.append(lg.add_patch(plt.Rectangle((0.0, i - 0.30), 0.05, 0.66, color=colour,
                                                   alpha=0.35 if st == "exclude" else 0.9)))
            sh = share.get(ly.tileset, 0.0)
            rows.append(lg.text(0.07, i - 0.18, ly.label, fontsize=10, va="center",
                                color="#888888" if st == "exclude" else "black"))
            rows.append(lg.text(0.58, i - 0.18, f"{len(ly.sheets)}", fontsize=9, va="center"))
            rows.append(lg.text(0.76, i - 0.18, f"{100 * sh:.0f}%" if st != "exclude" else "-", fontsize=9,
                                va="center"))
            label = ({"auto": "automatic", "exclude": "EXCLUDED"}.get(st)
                     or f"PREFERRED #{ps.prefer_order.index(ly.tileset) + 1}")
            rows.append(lg.text(0.07, i + 0.07, label, fontsize=8.5, va="center",
                                weight="bold" if st != "auto" else "normal",
                                color={"auto": "#555555", "prefer": "#1b7f3b", "exclude": "#b00020"}[st]))
            rows.append(lg.text(0.07, i + 0.30, ly.years(), fontsize=7.5, va="center", color="#666666"))
        none = share.get("none", 0.0)
        foot.set_text(("Seams: " + ("the preferred county wins wherever it has map; others fill the rest"
                                    if ps.prefer_order else "automatic - each county supplies its own territory (seams on the county boundaries)"))
                      + (f"      {100 * none:.0f}% of the area has no map (will be white)" if none >= 0.005 else "")
                      + "\nClick = cycle automatic / preferred / excluded     R = reset     Enter = accept     Esc = cancel")
        fig.canvas.draw_idle()

    state = {"accepted": False}

    def on_click(ev):
        if ev.inaxes is not lg or ev.ydata is None:
            return
        i = int(math.floor(ev.ydata + 0.5))
        if 0 <= i < len(layers):
            ps.cycle(i)
            redraw()

    def on_key(ev):
        if ev.key in ("enter", "return"):
            state["accepted"] = True
            plt.close(fig)
        elif ev.key in ("r", "R"):
            ps.reset()
            redraw()
        elif ev.key == "escape":
            plt.close(fig)

    redraw()
    if headless:
        for step in _script:
            if step[0] == "click":
                ps.cycle(step[1])
            elif step[0] == "enter":
                state["accepted"] = True
        redraw()
        if _save_png:
            fig.savefig(_save_png, dpi=90)
        plt.close(fig)
    else:
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()                              # blocks until the window is closed
    if not state["accepted"]:
        sys.exit("Plan cancelled - nothing was downloaded.")
    prefer_ids, exclude_ids = ps.result()
    print(f"  Plan accepted: preferred = {prefer_ids or 'none (automatic seams)'}, excluded = {exclude_ids or 'none'}")
    return prefer_ids, exclude_ids
