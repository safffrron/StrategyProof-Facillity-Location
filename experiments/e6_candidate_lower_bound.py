"""E6 - Randomised mechanisms with candidate sites cannot reach ratio 2 (report: Section 4.4, Theorem 11).

Tang et al. (2020) proved that no truthful-in-expectation (TIE) mechanism for the maximum cost with
candidate sites is better than 2-approximate, and asked whether 2 is attainable. For two agents and the
sites {0, 2, 4}, the best ratio of a TIE mechanism whose reports are restricted to a finite grid is a linear
program (src/candidate_lower_bound.py) and a lower bound for every TIE mechanism. Each value is certified:
the LP's dual multipliers are stored as rationals and the bound they imply is recomputed in exact
arithmetic, by code independent of the solver.

Default run: re-solve the two small grids and re-verify every stored certificate (seconds).
``--full``: also re-solve the large grids (about an hour; needs the highspy package and plenty of memory).

Outputs: results/e6_candidate_lower_bound.md, results/certificates/*.json,
figures/e6_candidate_lower_bound.{pdf,png}
"""

from __future__ import annotations
import argparse
import glob
import os
import time

import matplotlib.pyplot as plt

from common import INK, INK_2, MUTED, SERIES, RES_DIR, apply_style, save, write_markdown_table
from src.candidate_lower_bound import (CandidateLP, load_certificate, refined_grid, save_certificate,
                                       verify_certificate)

CERT_DIR = os.path.join(RES_DIR, "certificates")
RICH = ("1/500", "1/50", "1/20", "1/10", "1/5", "3/10")
DENSE = ("1/1000", "1/500", "1/200", "1/100", "1/50", "3/100", "1/20", "7/100", "1/10", "3/20", "1/5", "1/4",
         "3/10", "2/5")

GEOMETRIC = ("1/2000", "1/1000", "1/500", "3/1000", "1/200", "7/1000", "1/100", "3/200", "1/50", "1/40", "3/100",
             "1/25", "1/20", "7/100", "1/10", "3/20", "1/5", "1/4", "3/10", "2/5")

# name, candidates, grid keyword arguments, solver keyword arguments, quick?
# ("ipm-polish" needs the highspy package: interior point, then simplex on the tight misreport rows)
CONFIGS = [
    ("C2_basic", [0, 2], dict(), dict(generate=True), True),
    ("C3_basic", [0, 2, 4], dict(), dict(generate=True), True),
    ("C3_rich", [0, 2, 4], dict(offsets=RICH, step="1/4", lo=-1, hi=5), dict(generate=False), False),
    ("C3_mirror", [0, 2, 4], dict(offsets=RICH, step="1/4", lo=-1, hi=5, mirror=True), dict(generate=False), False),
    ("C3_dense", [0, 2, 4], dict(offsets=DENSE, step="1/4", lo=-1, hi=5, mirror=True), dict(generate=False), False),
    ("C3_geometric", [0, 2, 4], dict(offsets=GEOMETRIC, step="1/4", lo=-2, hi=6, mirror=True),
     dict(generate=False, method="ipm-polish"), False),
]
GRID_LABEL = {"C2_basic": "basic", "C3_basic": "basic", "C3_rich": "rich", "C3_mirror": "rich + mirrored",
              "C3_dense": "dense + mirrored", "C3_geometric": "geometric + mirrored"}


def run_config(name, C, kw, solve_kw):
    lp = CandidateLP(C, refined_grid(C, **kw))
    t0 = time.time()
    lp.solve(**solve_kw)
    cert = lp.certificate()
    cert["name"] = name
    save_certificate(cert, os.path.join(CERT_DIR, f"{name}.json"))
    cert["seconds"] = round(time.time() - t0, 1)                 # reported, not stored: keeps files reproducible
    return cert


def main(full: bool = False):
    os.makedirs(CERT_DIR, exist_ok=True)
    for name, C, kw, solve_kw, quick in CONFIGS:
        if quick or full or not os.path.exists(os.path.join(CERT_DIR, f"{name}.json")):
            cert = run_config(name, C, kw, solve_kw)
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
        rows, columns, os.path.join(RES_DIR, "e6_candidate_lower_bound.md"),
        header="E6 - Lower bounds on the max-cost ratio of truthful-in-expectation mechanisms for two agents "
               "with candidate sites (Tang et al. 2020 proved 2). Each bound is recomputed in exact "
               "rational arithmetic from the stored certificate and rounded down.")
    print("  figure:", figure(sorted(points), [b for c, _, b in rows_two(rows)]))
    assert all(r["exact match"] == "yes" for r in rows), "a certificate does not reproduce its bound"


def rows_two(rows):
    """(candidates, grid points, certified bound) of the two-site rows."""
    return [(r["candidates"], r["grid points"], float(r["certified lower bound"])) for r in rows
            if r["candidates"] == "{0, 2}"]



def figure(points, two_sites):
    """Certified lower bounds for the sites {0, 2, 4} against the number of grid points (Figure 3)."""
    apply_style()
    blue, orange = SERIES[0], SERIES[1]
    fig, ax = plt.subplots(figsize=(6.4, 2.55))
    best = max(lb for _, lb in points)
    ax.axhspan(best, 3, color=blue, alpha=0.07, lw=0, zorder=0)
    for level, text in [(3, "best known mechanisms: ratio 3"), (2, "previous lower bound: 2 (Tang et al.)")]:
        ax.axhline(level, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.text(318, level + 0.025, text, ha="right", va="bottom", fontsize=7.4, color=INK_2)
    ax.plot([p for p, _ in points], [lb for _, lb in points], "-o", color=blue, ms=6, mec="white", zorder=3,
            label="candidate sites {0, 2, 4}")
    for lb in two_sites:
        ax.plot([25], [lb], "s", color=orange, ms=6.5, mec="white", zorder=3,
                label=f"two sites {{0, 2}}: {lb:.3f}, close to Tang et al.'s 2")
    for pts, lb in points:                                       # rounded down: every value is a lower bound
        ax.text(pts, lb + 0.045, f"{int(lb * 1000) / 1000:.3f}", ha="center", va="bottom", fontsize=7.2, color=INK)
    ax.text(250, 2.64, f"still open: [{int(best * 1000) / 1000:.3f}, 3]", ha="center", va="center", fontsize=7.6,
            color=INK_2)
    ax.set_xlim(0, 320)
    ax.set_ylim(1.9, 3.1)
    ax.set_yticks([2, 2.25, 2.5, 2.75, 3])
    ax.set_xlabel("number of grid points for each agent's report")
    ax.set_ylabel("certified lower bound")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
    return save(fig, "e6_candidate_lower_bound")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="re-solve the large grids too")
    main(ap.parse_args().full)
