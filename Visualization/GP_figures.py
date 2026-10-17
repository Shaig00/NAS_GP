"""
The three Section 5.3 figures, from the stored diagnostics.  Nothing is
recomputed -- every number here was already saved by GP_uncertainty_diagnostics
and GP_results_aggregate.

    fig_pit.pdf             PIT under the deployed Student-t and under a Gaussian
    fig_nu_by_market.pdf    fitted tail index by market, log scale
    fig_sparsification.pdf  model, oracle and random orderings

    python make_section53_figures.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME, SEQ_LEN, RUN = "ensemble", 192, 1
NU_CEILING = 200.0
OUT_DIR = Path("gp_figures")
HORIZONS = (48, 96, 192)
STYLE = {48: ("o", "tab:blue"), 96: ("s", "tab:orange"), 192: ("^", "tab:green")}

# same sizes and weights as the Section 4.1 figures
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / name)
    plt.close(fig)
    print(f"saved -> {OUT_DIR / name}")


# ---------------------------------------------------------------------------
def fig_pit(diag):
    """Uniform means the predictive fits; a U shape means it is too narrow."""
    n_bins = len(diag["pit_t"])
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, key, title in zip(axes,
                              ("pit_t", "pit_gauss"),
                              ("Student-$t$ (deployed)", "Gaussian")):
        ax.bar(centres, diag[key], width=0.92 / n_bins,
               color="tab:blue", edgecolor="white", lw=0.4)
        ax.axhline(1 / n_bins, color="black", lw=0.8, ls="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel("PIT value")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2, lw=0.4, axis="y")

    axes[0].set_ylabel("Share of observations")
    fig.tight_layout()
    save(fig, f"fig_pit_{SEQ_LEN}.pdf")


# ---------------------------------------------------------------------------
def fig_nu(nu_by_horizon):
    """One marker per horizon: tight clusters mean the ordering reproduces,
    wide ones mean the index is not identified."""
    order = (pd.DataFrame({h: d["nu_final"] for h, d in nu_by_horizon.items()})
             .median(axis=1).sort_values().index)
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.hlines(y, 1, NU_CEILING, color="0.93", lw=0.7, zorder=0)

    for h, d in nu_by_horizon.items():
        marker, colour = STYLE[h]
        ax.scatter(d.loc[order, "nu_final"], y, s=22, marker=marker, zorder=3,
                   facecolors="none", edgecolors=colour, linewidths=1.0,
                   label=f"{h}-hour horizon")

    ax.axvline(2, color="0.4", lw=0.8, ls=":",
               label=r"$\nu = 2$: no finite variance below")
    if any((d["nu_final"] >= 0.99 * NU_CEILING).any()
           for d in nu_by_horizon.values()):
        ax.axvline(NU_CEILING, color="tab:red", lw=0.8, ls="--",
                   label="upper constraint, not identified")

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel(r"Fitted tail index $\nu$, logarithmic scale")
    ax.grid(alpha=0.2, lw=0.4, axis="x")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "fig_nu_by_market.pdf")


# ---------------------------------------------------------------------------
def fig_sparsification(diag):
    """An informative variance tracks the oracle; an uninformative one, random."""
    f, curves = diag["fractions"], diag["curves"]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.fill_between(f, curves["oracle"], curves["model"],
                    color="tab:blue", alpha=0.07, lw=0)
    for key, colour, style, label in (
            ("random", "0.5", "--", "random ordering"),
            ("model", "tab:blue", "-", "ordered by predicted uncertainty"),
            ("oracle", "tab:green", "-", "ordered by realised error")):
        ax.plot(f, curves[key], color=colour, ls=style, lw=1.6, label=label)

    ax.set_xlabel("Fraction of most uncertain observations removed")
    ax.set_ylabel("RMSE on the remainder")
    ax.set_xlim(f.min(), f.max())
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2, lw=0.4)
    ax.legend(loc="center left", frameon=False)
    ax.text(0.98, 0.05, f"AUSE = {diag['ause']:.3f}", transform=ax.transAxes,
            ha="right", fontsize=8, color="0.3")
    fig.tight_layout()
    save(fig, f"fig_sparsification_{SEQ_LEN}.pdf")


# ---------------------------------------------------------------------------
def main():
    stem = f"{NAME}_{SEQ_LEN}_{RUN}"
    with open(f"gp_uncertainty_diagnostics_{stem}.pkl", "rb") as f:
        diag = pickle.load(f)
    nu = pd.read_csv(f"gp_aggregates/nu_by_channel_{stem}.csv", index_col=0)

    fig_pit(diag)
    fig_sparsification(diag)

    nu_by_horizon = {h: pd.read_csv(
        f"gp_aggregates/nu_by_channel_{NAME}_{h}_{RUN}.csv", index_col=0)
        for h in HORIZONS}
    
    fig_nu(nu_by_horizon)


if __name__ == "__main__":
    main()