"""Shared plotting style for the political-metallurgy case studies.

Palette follows the validated reference dataviz palette (light mode):
categorical slots in fixed order; low-contrast slots (aqua, yellow) are
always paired with direct labels per the relief rule.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# Global font scale for PDF-legible output. Set LT_FONT_SCALE (e.g. 1.7) in the
# environment to enlarge every label, title, and annotation at once; default 1.0
# leaves the original on-screen sizing untouched.
FONT_SCALE = float(os.environ.get("LT_FONT_SCALE", "1.0"))

# Journal figures carry no in-image headline: the LaTeX caption does that job.
# Set LT_TITLES=0 to suppress every figure-level title/subtitle (the bold banner
# and its subtitle line); small per-panel identifiers are kept, since captions
# refer to panels by those IDs (P1..P5, F2..F7, channel names).
SHOW_TITLES = os.environ.get("LT_TITLES", "1") != "0"

# Output resolution. Springer/JCSS asks for >=300 dpi on halftone/combination
# art, so submission figures are generated with LT_DPI=300; default stays 200
# for the on-screen theory-document figures.
DPI = int(os.environ.get("LT_DPI", "200"))

# surfaces and ink
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# categorical slots (fixed order)
BLUE = "#2a78d6"    # slot 1: observed data
AQUA = "#1baf7a"    # slot 2: theoretical fit   (relief rule: direct-label it)
YELLOW = "#eda100"  # slot 3: null model        (relief rule: direct-label it)
RED = "#e34948"     # slot 6: fracture / falsification marker


def _install_font_scaling(scale):
    """Multiply every explicit ``fontsize``/``size`` keyword by ``scale``.

    The case-study scripts hard-code font sizes in ``set_title``/``text``/
    ``annotate`` calls, which rcParams cannot override. Wrapping the relevant
    Matplotlib methods once (idempotently) lets a single env var enlarge all of
    them for print, without editing every call site.
    """
    if scale == 1.0 or getattr(mpl, "_lt_font_scaled", False):
        return
    import matplotlib.axes as _axes
    import matplotlib.figure as _figure

    def _scaled(kw):
        # Enlarge small in-panel text (labels, annotations, fit statistics) for
        # print, but cap the result so already-large titles/suptitles do not
        # blow past the figure width. Never shrink text below its original size.
        for key in ("fontsize", "size"):
            v = kw.get(key)
            if isinstance(v, (int, float)):
                kw[key] = min(v * scale, max(v, 13.5))
        return kw

    targets = [
        (_axes.Axes, "text"), (_axes.Axes, "annotate"), (_axes.Axes, "set_title"),
        (_axes.Axes, "set_xlabel"), (_axes.Axes, "set_ylabel"), (_axes.Axes, "legend"),
        (_figure.Figure, "text"), (_figure.Figure, "suptitle"), (_figure.Figure, "legend"),
    ]
    for cls, name in targets:
        orig = getattr(cls, name)

        def wrap(orig):
            def wrapped(self, *args, **kwargs):
                return orig(self, *args, **_scaled(kwargs))
            return wrapped

        setattr(cls, name, wrap(orig))
    mpl._lt_font_scaled = True


def apply_style():
    s = FONT_SCALE
    mpl.rcParams.update({
        "figure.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.size": 10 * s,
        "axes.labelsize": 11 * s,
        "axes.titlesize": 12 * s,
        "xtick.labelsize": 9.5 * s,
        "ytick.labelsize": 9.5 * s,
        "legend.fontsize": 9.5 * s,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "savefig.dpi": DPI,
        "savefig.facecolor": PAGE,
        "savefig.bbox": "tight",
    })
    _install_font_scaling(s)


def titles(ax, title, subtitle=None):
    if not SHOW_TITLES:
        return
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=14)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=9, color=INK2, va="bottom")


def direct_label(ax, x, y, text, color, dx=4, dy=0, fontsize=9, ha="left", va="center"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=fontsize, fontweight="bold", ha=ha, va=va)
