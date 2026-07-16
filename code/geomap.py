"""Lightweight country-border drawing for the case-study maps.

No GIS stack required: reads a cached Natural Earth 50m admin-0 GeoJSON
(fetched once from the documented URL if absent) and draws country polygons
directly with matplotlib Paths. Enough for regional schematic maps that carry
the theory's overlay (grain-boundary fault lines, crisis stress-concentrators).

Source: Natural Earth via nvkelso/natural-earth-vector (public domain).
"""

import json
import os
import urllib.request

from matplotlib.patches import PathPatch
from matplotlib.path import Path

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "lattice_theory")
GEOJSON = os.path.join(CACHE, "ne_50m_countries.geojson")
URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
       "master/geojson/ne_50m_admin_0_countries.geojson")

_CACHE = {}


def _load():
    if "gj" in _CACHE:
        return _CACHE["gj"]
    if not os.path.exists(GEOJSON):
        os.makedirs(CACHE, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            open(GEOJSON, "wb").write(r.read())
    with open(GEOJSON) as f:
        gj = json.load(f)
    _CACHE["gj"] = gj
    return gj


def _polys(geom):
    """Yield lists of (lon, lat) rings for Polygon / MultiPolygon geometries."""
    t = geom["type"]
    if t == "Polygon":
        for ring in geom["coordinates"]:
            yield ring
    elif t == "MultiPolygon":
        for poly in geom["coordinates"]:
            for ring in poly:
                yield ring


def _name(props):
    for k in ("ADMIN", "NAME_LONG", "NAME"):
        if props.get(k):
            return props[k]
    return ""


def features():
    return _load()["features"]


def _match(nm, fills):
    """Case-insensitive substring match of a country name against fill keys."""
    low = nm.lower()
    for key, col in fills.items():
        if key.lower() in low:
            return col
    return None


def draw(ax, bbox, base="#eeede7", edge="#c3c2b7", lw=0.6,
         fills=None, land_label_color=None, ocean="#f4f5f7"):
    """Draw all countries intersecting bbox=(lon_min, lon_max, lat_min, lat_max).

    fills: dict {name_substring: facecolor} paints matching countries (first
    matching key wins); everything else gets `base`.
    """
    fills = fills or {}
    lon0, lon1, lat0, lat1 = bbox
    ax.set_facecolor(ocean)
    for feat in features():
        nm = _name(feat["properties"])
        face = _match(nm, fills) or base
        for ring in _polys(feat["geometry"]):
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            if max(xs) < lon0 or min(xs) > lon1 or max(ys) < lat0 or min(ys) > lat1:
                continue
            path = Path(ring, closed=True)
            ax.add_patch(PathPatch(path, facecolor=face, edgecolor=edge,
                                   lw=lw, zorder=1))
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / max(0.1, __import__("math").cos(
        __import__("math").radians((lat0 + lat1) / 2))))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
