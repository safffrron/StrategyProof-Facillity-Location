"""Shared paths, plotting style and table helpers for the experiment scripts."""

from __future__ import annotations
import os
import sys
from typing import Dict, List, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Validated categorical order (blue, orange, aqua, yellow); aqua/yellow are low-contrast on
# white, so every series also gets a distinct marker and a direct label or table.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
# Ordinal blue ramp for ordered levels (e.g. blocked density), light -> dark.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def theory_line(ax, x, y, label: str, label_at: int = -1, dy: float = 0.0, text_xy=None, ha: str = "left",
                **kw) -> None:
    """Thin ink line for a proved bound, labelled directly (at one end, or at ``text_xy``)."""
    ax.plot(x, y, color=INK_2, linewidth=1.0, zorder=1, **kw)
    if text_xy is not None:
        ax.text(text_xy[0], text_xy[1], label, fontsize=7.5, color=INK_2, ha=ha, va="center")
        return
    ax.annotate(label, (x[label_at], y[label_at]), xytext=(4, 2 + dy), textcoords="offset points",
                fontsize=7.5, color=INK_2, va="bottom", ha="left")


def save(fig, name: str) -> str:
    path_png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path_png)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    plt.close(fig)
    return path_png


def write_markdown_table(rows: List[Dict], columns: Sequence[str], path: str, header: str = "") -> None:
    lines = []
    if header:
        lines += [header, ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
