"""E7 - Diagram of the tight instance behind Theorem 2 (why blocked zones double the error).

Agents at 0 and 2, a blocked zone (1, 1 + 2*eta - eps), prediction 1 + eta (error eta: the
optimum is 1, with optimal max cost 1). MinMaxP keeps the prediction (cost 1 + eta); the right end
of the zone is eps closer than the left end, so snapping moves the facility to 1 + 2*eta - eps
(cost -> 1 + 2*eta as eps -> 0).

Output: figures/e7_doubling_diagram.{png,pdf}
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from common import AXIS, GRID, INK, INK_2, MUTED, SERIES, apply_style, save

ETA = 0.35
EPS = 0.03


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    ax.axis("off")
    lo, hi = -0.2, 2.25
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.95, 1.0)
    ax.plot([lo + 0.05, hi - 0.05], [0, 0], color=AXIS, linewidth=1.2, zorder=1)
    zone = (1.0, 1.0 + 2 * ETA - EPS)
    ax.add_patch(Rectangle((zone[0], -0.08), zone[1] - zone[0], 0.16, facecolor=GRID, edgecolor=MUTED,
                           linewidth=0.6, hatch="////", zorder=2))
    ax.text(sum(zone) / 2, -0.2, "blocked zone (1, 1 + 2η − ε)", ha="center", va="top", fontsize=7.5, color=INK_2)
    for x in (0.0, 2.0):
        ax.plot([x], [0], marker="o", markersize=8, color=INK, zorder=4)
        ax.text(x, -0.2, f"agent at {x:g}", ha="center", va="top", fontsize=8, color=INK)
    ax.plot([1.0], [0], marker="^", markersize=9, color=SERIES[2], markeredgecolor="white", zorder=5)
    ax.annotate("optimum o = 1\n(max cost 1)", (1.0, 0.0), xytext=(0.55, 0.42), ha="center", va="bottom",
                fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    p = 1.0 + ETA
    ax.plot([p], [0], marker="D", markersize=7, color=SERIES[0], markeredgecolor="white", zorder=5)
    ax.annotate("prediction 1 + η = MinMaxP output\n(cost 1 + η)", (p, 0.0), xytext=(p, 0.42), ha="center",
                va="bottom", fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    f = zone[1]
    ax.plot([f], [0], marker="s", markersize=8, color=SERIES[1], markeredgecolor="white", zorder=5)
    ax.annotate("MinMaxP* output\n(cost 1 + 2η − ε)", (f, 0.0), xytext=(2.0, 0.42), ha="center", va="bottom",
                fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    ax.add_patch(FancyArrowPatch((p + 0.02, -0.55), (f - 0.01, -0.55), arrowstyle="-|>", mutation_scale=9,
                                 color=SERIES[1], linewidth=1.2))
    ax.text((p + f) / 2, -0.63, "snap to the nearest allowed point\n(the right end is ε closer)", ha="center",
            va="top", fontsize=7.5, color=INK_2)
    ax.text(lo + 0.02, 0.98, f"Tight instance for Theorem 2 (η = {ETA}): snapping reflects the prediction's error "
                             "across the blocked zone, doubling it", fontsize=8.5, color=INK, va="top")
    print("E7 saved", save(fig, "e7_doubling_diagram"))


if __name__ == "__main__":
    main()
