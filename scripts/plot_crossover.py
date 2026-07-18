"""Learning curve figure: test MAE against number of training labels.

Colour encodes the arm (composition or structure), because that contrast is the
claim the figure exists to support. Line style separates models within an arm.
The mean predictor is a neutral reference line, not a series.

Palette validated for colour vision deficiency: #0072B2 / #D55E00 give
dE 91.9 under protanopia against a light surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "savefig.dpi": 300,
    }
)

RESULTS = Path("results/crossover")
OUT = Path("../When_Composition_Beats_Structure__Machine_Learning_Discovery_of_Solid_State_Electrolytes/figures")

COMPOSITION = "#0072B2"
STRUCTURE = "#D55E00"
NEUTRAL = "#666666"

STYLE = {
    "random_forest": (COMPOSITION, "-", "o", "Random forest (composition)"),
    "xgboost": (COMPOSITION, "--", "s", "XGBoost (composition)"),
    "cgcnn": (STRUCTURE, "-", "o", "CGCNN (structure)"),
    "alignn": (STRUCTURE, "--", "s", "ALIGNN (structure)"),
    "alignn_small": (STRUCTURE, ":", "^", "ALIGNN small (structure)"),
}

# Experimental reproducibility floor reported with OBELiX, in log10 S/cm.
EXPERIMENTAL_FLOOR = 0.41


def load() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in RESULTS.glob("raw_*.csv")]
    if not frames:
        sys.exit("no results found; run scripts/run_crossover.py first")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Reference band: below the experimental floor there is nothing left to learn.
    ax.axhspan(0, EXPERIMENTAL_FLOOR, color=NEUTRAL, alpha=0.10, lw=0)
    ax.text(
        df.n.max(), EXPERIMENTAL_FLOOR - 0.07, "experimental scatter (0.41)",
        fontsize=8, color=NEUTRAL, va="top", ha="right",
    )

    if "mean_baseline" in set(df.model):
        b = df[df.model == "mean_baseline"].groupby("n")["test_mae"].mean()
        ax.plot(b.index, b.values, color=NEUTRAL, ls=(0, (4, 3)), lw=1.4, zorder=1)
        ax.text(
            b.index[0] * 1.03, b.values[0] + 0.05, "predict the mean",
            fontsize=8, color=NEUTRAL, ha="left", va="bottom",
        )

    for model, (color, ls, marker, label) in STYLE.items():
        sub = df[df.model == model]
        if sub.empty:
            continue
        g = sub.groupby("n")["test_mae"].agg(["mean", "std", "count"])
        ax.plot(
            g.index, g["mean"], color=color, ls=ls, marker=marker,
            ms=5, lw=2.0, label=label, zorder=3,
            markeredgecolor="white", markeredgewidth=0.6,
        )
        if (g["count"] > 1).any():
            ax.fill_between(
                g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                color=color, alpha=0.12, lw=0, zorder=2,
            )

    ax.set_xscale("log")
    ax.set_xticks(sorted(df.n.unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    # Suppress the log minor ticks (3x10^1 etc); the label positions are the grid.
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("Training labels $n$")
    ax.set_ylabel(r"Test MAE on $\log_{10}\sigma$  (S cm$^{-1}$)")
    ax.set_ylim(0, max(2.4, df.test_mae.max() * 1.05))
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"crossover.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/crossover.pdf")

    summary = (
        df.groupby(["arm", "model", "n"])["test_mae"].agg(["mean", "std", "count"]).round(3)
    )
    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()
