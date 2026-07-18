"""Feature analysis for the composition arm: SHAP attribution and descriptor redundancy.

Answers two questions the model card should not leave open. Which descriptors
carry the prediction, and how much of the 132 dimensional Magpie vector is
actually independent information.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

plt.rcParams.update(
    {"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
     "axes.edgecolor": "#333333", "savefig.dpi": 300}
)

CACHE = Path("data/obelix_cache")
OUT = Path("../When_Composition_Beats_Structure__Machine_Learning_Discovery_of_Solid_State_Electrolytes/figures")
RESULTS = Path("results/crossover")
COMPOSITION = "#0072B2"


def main() -> None:
    import shap

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(CACHE / "features.parquet")
    cols = [c for c in feats.columns if c.startswith("MagpieData")]
    X_all = feats[cols].to_numpy(dtype=float)
    y = feats["y"].to_numpy(dtype=float)
    groups = feats["reduced"].to_numpy()

    # Same grouped split as the learning curve, so the model explained here is
    # the model reported there.
    pool_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X_all, y, groups)
    )
    imp = SimpleImputer(strategy="median").fit(X_all[pool_idx])
    X_tr, X_te = imp.transform(X_all[pool_idx]), imp.transform(X_all[test_idx])

    rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(X_tr, y[pool_idx])
    print(f"RF trained on {len(pool_idx)} entries; test MAE "
          f"{np.mean(np.abs(rf.predict(X_te) - y[test_idx])):.3f}")

    # ---- SHAP -------------------------------------------------------
    shap_values = shap.TreeExplainer(rf).shap_values(X_te)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]

    short = [cols[i].replace("MagpieData ", "") for i in order]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, mean_abs[order], color=COMPOSITION, height=0.68)
    ax.set_yticks(ypos)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel(r"mean $|$SHAP$|$  (log$_{10}\sigma$ per descriptor)")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "shap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "shap.png", bbox_inches="tight")
    print(f"wrote {OUT}/shap.pdf")

    pd.DataFrame({"feature": [cols[i] for i in order],
                  "mean_abs_shap": mean_abs[order]}).to_csv(
        RESULTS / "shap_top15.csv", index=False)

    # ---- Redundancy -------------------------------------------------
    # How many descriptors are effectively independent? Count principal
    # components needed to reach 95% of the variance, and count |r| > 0.95 pairs.
    Z = (X_tr - X_tr.mean(0)) / (X_tr.std(0) + 1e-12)
    evals = np.linalg.svd(np.cov(Z, rowvar=False), compute_uv=False)
    cum = np.cumsum(evals) / evals.sum()
    n95 = int(np.searchsorted(cum, 0.95) + 1)

    corr = np.corrcoef(Z, rowvar=False)
    iu = np.triu_indices_from(corr, k=1)
    n_pairs = int((np.abs(corr[iu]) > 0.95).sum())

    print()
    print(f"Magpie descriptors                 : {len(cols)}")
    print(f"PCs to explain 95% of variance     : {n95}")
    print(f"descriptor pairs with |r| > 0.95   : {n_pairs}")
    print(f"top 15 carry {100*mean_abs[order].sum()/mean_abs.sum():.0f}% of total |SHAP|")

    pd.Series({"n_features": len(cols), "n_pcs_95pct": n95,
               "n_pairs_abs_r_gt_0.95": n_pairs}).to_csv(
        RESULTS / "redundancy.csv")


if __name__ == "__main__":
    main()
