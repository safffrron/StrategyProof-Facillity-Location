"""E12 - Randomised mechanisms with candidate locations cannot reach ratio 2 (Theorem 17).

Tang et al. (2020) prove that no truthful-in-expectation (TIE) mechanism for the maximum cost with candidate
locations is better than 2-approximate, and ask whether 2 is attainable; Fang et al. (2025) list randomised
mechanisms as open. For two agents, the best ratio of any TIE mechanism whose reports are restricted to a
finite grid lower-bounds every TIE mechanism on the line. We compute it by linear programming
(src/candidate_lower_bound.py) and certify every value in exact rational arithmetic.

Default run: re-solve the quick configurations and re-verify every stored certificate (seconds).
``--full``: also re-solve the large grids and overwrite their certificates (about 30 minutes).

Outputs: results/e12_candidate_lb.md, results/e12_certificates/*.json, figures/e12_candidate_lb.{png,pdf}
"""

from __future__ import annotations
import argparse
import glob
import os
import time

import matplotlib.pyplot as plt

from common import AXIS, INK, INK_2, MUTED, SERIES, RES_DIR, apply_style, save, theory_line, write_markdown_table
from src.candidate_lower_bound import (CandidateLP, load_certificate, refined_grid, save_certificate,
                                       verify_certificate)

CERT_DIR = os.path.join(RES_DIR, "e12_certificates")
RICH = ("1/500", "1/50", "1/20", "1/10", "1/5", "3/10")
DENSE = ("1/1000", "1/500", "1/200", "1/100", "1/50", "3/100", "1/20", "7/100", "1/10", "3/20", "1/5", "1/4",
         "3/10", "2/5")

# name, candidates, grid keyword arguments, impose every misreport (constraint generation)?, quick?
CONFIGS = [
    ("C2_basic", [0, 2], dict(), True, True),
    ("C3_basic", [0, 2, 4], dict(), True, True),
    ("C3_rich", [0, 2, 4], dict(offsets=RICH, step="1/4", lo=-1, hi=5), False, False),
    ("C3_mirror", [0, 2, 4], dict(offsets=RICH, step="1/4", lo=-1, hi=5, mirror=True), False, False),
    ("C3_dense", [0, 2, 4], dict(offsets=DENSE, step="1/4", lo=-1, hi=5, mirror=True), False, False),
]
GRID_LABEL = {"C2_basic": "basic", "C3_basic": "basic", "C3_rich": "rich", "C3_mirror": "rich + mirrored",
              "C3_dense": "dense + mirrored"}


def run_config(name, C, kw, generate):
    lp = CandidateLP(C, refined_grid(C, **kw))
    t0 = time.time()
    lp.solve(generate=generate)
    cert = lp.certificate()
    cert["seconds"] = round(time.time() - t0, 1)
    cert["name"] = name
    save_certificate(cert, os.path.join(CERT_DIR, f"{name}.json"))
    return cert


def main(full: bool = False):
    os.makedirs(CERT_DIR, exist_ok=True)
    for name, C, kw, generate, quick in CONFIGS:
        if quick or full or not os.path.exists(os.path.join(CERT_DIR, f"{name}.json")):
            cert = run_config(name, C, kw, generate)
            print(f"  solved {name}: LP {cert['lp_value']:.6f}, certified {cert['bound_float']:.6f} "
                  f"({cert['seconds']}s)")
    columns = ["candidates", "grid", "grid points", "misreports imposed", "LP optimum", "certified lower bound",
               "multipliers (SP + ratio)", "exact match", "certificate"]
    rows, points = [], []
    certs = [(path, load_certificate(path)) for path in glob.glob(os.path.join(CERT_DIR, "*.json"))]
    for path, cert in sorted(certs, key=lambda pc: (len(pc[1]["candidates"]), pc[1]["grid_size"])):
        bound = verify_certificate(cert)
        ok = f"{bound.numerator}/{bound.denominator}" == cert["bound"]
        floor4 = int(bound * 10000) / 10000                       # rounded down: a safe claim
        rows.append(dict(zip(columns, [
            "{" + ", ".join(cert["candidates"]) + "}", GRID_LABEL.get(cert["name"], cert["name"]), cert["grid_size"],
            "all" if cert["all_misreports_imposed"] else "local", f"{cert['lp_value']:.6f}",
            f"{floor4:.4f}", f"{len(cert['sp'])} + {len(cert['ratio'])}", "yes" if ok else "NO",
            os.path.basename(path)])))
        print(f"  verified {os.path.basename(path)}: bound >= {floor4:.4f} (exact match: {ok})")
        if cert["candidates"] == ["0", "2", "4"]:
            points.append((cert["grid_size"], floor4))
    write_markdown_table(
        rows, columns, os.path.join(RES_DIR, "e12_candidate_lb.md"),
        header="E12 - Lower bounds on the max-cost ratio of truthful-in-expectation mechanisms for two agents "
               "with candidate locations (Tang et al. 2020 proved 2). Each bound is recomputed in exact "
               "rational arithmetic from the stored certificate and rounded down.")
    figure(sorted(points))



def figure(points):
    """(a) The two families of profiles that carry the certificate; (b) certified bounds against grid size."""
    apply_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax.axis("off")
    ax.set_xlim(-2.1, 5.9)
    ax.set_ylim(-1.15, 1.05)
    ax.plot([-1.7, 5.7], [0, 0], color=AXIS, linewidth=1.2, zorder=1)
    for c in (0, 2, 4):
        ax.plot([c], [0], marker="s", markersize=8, color=INK, zorder=4)
        ax.text(c, -0.16, f"{c}", ha="center", va="top", fontsize=8, color=INK)
    d = 0.1
    rows = [(0.55, (1 - d, 3 + d), SERIES[0], "$A_\\delta=(1-\\delta,\\,3+\\delta)$: optimum 2; 0 and 4 cost about 3$\\times$"),
            (-0.6, (-1 - d, 1 + d), SERIES[1], "$B_\\delta=(-1-\\delta,\\,1+\\delta)$: optimum 0; 2 costs about 3$\\times$")]
    for y, xs, col, text in rows:
        for x in xs:
            ax.plot([x, x], [0, y], color=col, linewidth=0.6, linestyle=":", zorder=2)
            ax.plot([x], [y], marker="o", markersize=7, color=col, markeredgecolor="white", zorder=5)
        ax.text(-2.0, y + (0.2 if y > 0 else -0.24), text, fontsize=7.5, color=INK, ha="left", va="center")
    ax.set_title("(a) Profiles carrying the certificate (small $\\delta$)", fontsize=9)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bx.plot(xs, ys, color=SERIES[0], marker="o", markersize=6, zorder=3)
    for x, y in points:
        bx.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5,
                    color=INK)
    lo, hi = min(xs) - 15, max(xs) + 25
    theory_line(bx, [lo, hi], [2, 2], "previous lower bound 2 (Tang et al.)", text_xy=(lo + 3, 1.93))
    theory_line(bx, [lo, hi], [3, 3], "deterministic / universally truthful: 3 (tight)", text_xy=(lo + 3, 2.93))
    bx.fill_between([lo, hi], max(ys), 3, color=SERIES[0], alpha=0.06, linewidth=0)
    bx.text(hi - 3, (max(ys) + 3) / 2, "open", ha="right", va="center", fontsize=8, color=MUTED)
    bx.set_xlim(lo, hi)
    bx.set_ylim(1.85, 3.1)
    bx.set_xlabel("grid points (candidates {0, 2, 4})")
    bx.set_ylabel("certified lower bound")
    bx.set_title("(b) Truthful-in-expectation ratio", fontsize=9)
    fig.tight_layout()
    return save(fig, "e12_candidate_lb")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="re-solve the large grids too")
    main(ap.parse_args().full)
