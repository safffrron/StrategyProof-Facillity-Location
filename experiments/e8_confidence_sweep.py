"""E8 - The project brief's confidence sweep on the line (report: Section 3.3, Appendix C).

The confidence family PhantomMedian[k] places the facility at the median of the reports and k copies of the
prediction: k = 0 is the median (no trust) and k = n - 1 is MinMaxP (full trust). For n = 7 agents on a line
without blocked zones we sweep k in {0, 2, 4, 6}, for both objectives:

(a) consistency (correct prediction) and robustness (arbitrary prediction), found by adversarial search, against
    the theory: for the maximum cost, consistency 2 for every k < n - 1 and 1 for MinMaxP, robustness 2
    (Agrawal et al. 2022); for the social cost, consistency 1 (the median is optimal) and robustness
    max{1, (n + k - 1)/(n - k + 1)};
(b) the ratio against the prediction error eta: the mean over random instances for each k, and for MinMaxP the
    worst case found by search against its proved bound 1 + min{eta, 1} (Agrawal et al.; Chan, Lin and Wang).

Outputs: results/e8_confidence_sweep.md, figures/e8_confidence_sweep.{pdf,png}
"""

from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt

from common import INK, INK_2, RES_DIR, SERIES, apply_style, save, write_markdown_table
from src.experiment_utils import Scenario, error_scale, expected_objective, hill_climb, optimum, point_predictions, ratio
from src.learning_augmented import PhantomMedian

N = 7
KS = [0, 2, 4, 6]
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5]
OBJECTIVES = {"mc": "max cost", "sc": "social cost"}


def theory(k: int, objective: str):
    """(consistency, robustness) of PhantomMedian[k] on the line, n = N."""
    if objective == "mc":
        return (1.0 if k == N - 1 else 2.0), 2.0
    return 1.0, max(1.0, (N + k - 1) / (N - k + 1))


def search(mech, objective: str, mode: str, eta: float = 0.0, seed: int = 0) -> float:
    return hill_climb(Scenario(mech, objective, N, 0, mode, eta=eta), restarts=512, steps=600, seed=seed).best_ratio


def typical_mean(k: int, objective: str, eta: float, rng, profiles: int = 20_000) -> float:
    """Mean ratio over random uniform profiles with predictions at error exactly eta."""
    X = rng.uniform(0, 1, (profiles, N))
    o, opt = optimum(X, None, objective)
    pred = point_predictions(o, error_scale(opt, N, objective), eta, rng)
    return float(ratio(expected_objective(PhantomMedian(k, snap=False), X, None, pred, objective), opt).mean())


def figure(frontier, curves, worst):
    apply_style()
    blue, orange = SERIES[0], SERIES[1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for objective, color, marker in (("mc", blue, "o"), ("sc", orange, "s")):
        rows = [r for r in frontier if r["objective"] == OBJECTIVES[objective]]
        ax.plot([r["robustness (search)"] for r in rows], [r["consistency (search)"] for r in rows], marker=marker,
                color=color, ms=6, mec="white", lw=1.0, label=f"{OBJECTIVES[objective]}: found by search")
        ax.plot([r["robustness (theory)"] for r in rows], [r["consistency (theory)"] for r in rows], marker=marker,
                color=color, ms=10, mfc="none", lw=0, label=f"{OBJECTIVES[objective]}: theory")
        groups = {}                                              # several k can share one point
        for r in rows:
            groups.setdefault((r["robustness (search)"], r["consistency (search)"]), []).append(str(r["k"]))
        for xy, ks in groups.items():                          # max cost labelled below, social cost above
            ax.annotate("k = " + ", ".join(ks), xy, xytext=(6, -12) if objective == "mc" else (5, 4),
                        textcoords="offset points", fontsize=7.2, color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 6])
    ax.set_xticklabels(["1", "2", "3", "6"])
    ax.minorticks_off()
    ax.set_xlabel("robustness")
    ax.set_ylabel("consistency")
    ax.set_ylim(0.85, 2.2)
    ax.set_title(f"(a) Confidence sweep, n = {N}")
    ax.legend(loc="center right", fontsize=6.8)
    e = np.linspace(0, max(ETAS), 200)
    bx.plot(e, 1 + np.minimum(e, 1), color=INK, lw=1.0, label="bound 1 + min{η, 1}")
    bx.plot(ETAS, worst, "o", color=blue, ms=5.5, mec="white", label="worst found (MinMaxP)")
    for k, color in zip(KS, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]):
        label = f"mean, k = {k}" + (" = MinMaxP" if k == N - 1 else "")
        bx.plot(ETAS, curves[k], color=color, lw=1.4, ls="--" if k == N - 1 else "-", label=label)
    bx.set_xlabel("prediction error η")
    bx.set_ylabel("max-cost ratio")
    bx.set_title("(b) Ratio against prediction error")
    bx.set_ylim(0.95, 2.3)
    bx.legend(loc="upper left", fontsize=6.8)
    fig.tight_layout()
    return save(fig, "e8_confidence_sweep")


def main() -> None:
    frontier = []
    for objective, label in OBJECTIVES.items():
        for k in KS:
            mech = PhantomMedian(k, snap=False)
            cons, rob = search(mech, objective, "error"), search(mech, objective, "robust")
            t_cons, t_rob = theory(k, objective)
            frontier.append({"objective": label, "k": k, "consistency (search)": round(cons, 4),
                             "consistency (theory)": round(t_cons, 4), "robustness (search)": round(rob, 4),
                             "robustness (theory)": round(t_rob, 4)})
            print("  ", frontier[-1])
    rng = np.random.default_rng(0)
    curves = {k: [typical_mean(k, "mc", e, rng) for e in ETAS] for k in KS}
    worst = [search(PhantomMedian(N - 1, snap=False), "mc", "error", eta=e, seed=i) for i, e in enumerate(ETAS)]
    errors = [{"eta": e, **{f"mean, k = {k}": round(curves[k][i], 4) for k in KS},
               "MinMaxP worst (search)": round(worst[i], 4), "MinMaxP bound": round(1 + min(e, 1), 4)}
              for i, e in enumerate(ETAS)]
    path = os.path.join(RES_DIR, "e8_confidence_sweep.md")
    write_markdown_table(frontier, list(frontier[0]), path,
                         header=f"## E8a - Consistency and robustness of PhantomMedian[k], n = {N}, no blocked zones "
                                "(search values are lower bounds on the worst case)")
    tmp = path + ".b"
    write_markdown_table(errors, list(errors[0]), tmp,
                         header="## E8b - Max-cost ratio against prediction error: mean over 20,000 random profiles "
                                "for each k, and MinMaxP's worst case found by search")
    with open(path, "a") as fh, open(tmp) as fb:
        fh.write("\n" + fb.read())
    os.remove(tmp)
    print("  figure:", figure(frontier, curves, worst))
    for r in frontier:
        assert abs(r["consistency (search)"] - r["consistency (theory)"]) < 1e-2, r
        assert abs(r["robustness (search)"] - r["robustness (theory)"]) < 1e-2, r
    for e, w in zip(ETAS, worst):
        assert abs(w - (1 + min(e, 1))) < 1e-2, (e, w)


if __name__ == "__main__":
    main()
