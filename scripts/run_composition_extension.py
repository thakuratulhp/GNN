"""Composition arm beyond the structure ceiling.

The matched comparison stops at n=200 because only 281 OBELiX entries carry a
structure. The composition arm has no such limit: it can use all 558 usable
measurements. Running it out to the full set answers a question the matched
comparison cannot, namely whether the composition curve is still descending at
the point where the structural arm runs out of data, and therefore whether the
structural arm's deficit could plausibly be closed by labels alone.

This is reported as a separate curve, never overlaid on the matched comparison
as though it were part of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_crossover as R  # noqa: E402

N_GRID = [25, 50, 100, 150, 200, 300, 400]
SEEDS = [0, 1, 2, 3, 4]
OUT = Path("results/crossover")
CENSOR_FLOOR = -14.0


def main() -> None:
    from obelix import OBELiX
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition

    OUT.mkdir(parents=True, exist_ok=True)
    cache = Path("data/obelix_cache/features_all.parquet")

    if cache.exists():
        feats = pd.read_parquet(cache)
    else:
        ob = OBELiX(data_path="data/obelix")
        df = ob.dataframe.copy()
        df["sigma"] = pd.to_numeric(df["Ionic conductivity (S cm-1)"], errors="coerce")
        df["y"] = np.log10(df["sigma"].where(df["sigma"] > 0))
        df = df[df["y"].notna() & (df["y"] > CENSOR_FLOOR)]
        df = df.reset_index()[["True Composition", "Reduced Composition", "y"]]
        df.columns = ["composition", "reduced", "y"]
        df = StrToComposition(target_col_id="obj").featurize_dataframe(
            df, col_id="composition", ignore_errors=True
        )
        df = ElementProperty.from_preset("magpie").featurize_dataframe(
            df, col_id="obj", ignore_errors=True
        )
        feats = df.drop(columns=["obj"])
        feats.to_parquet(cache, index=False)

    cols = [c for c in feats.columns if c.startswith("MagpieData")]
    feats = feats.dropna(subset=["y"])
    X = feats[cols].to_numpy(dtype=float)
    y = feats["y"].to_numpy(dtype=float)
    groups = feats["reduced"].astype(str).to_numpy()
    print(f"composition-only set: {len(y)} entries "
          f"({len(np.unique(groups))} unique compositions)")

    pool_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=R.TEST_FRACTION, random_state=42).split(X, y, groups)
    )
    X_te, y_te = X[test_idx], y[test_idx]
    pool_groups = groups[pool_idx]
    print(f"pool={len(pool_idx)}  test={len(test_idx)}  "
          f"shared compositions={len(set(pool_groups) & set(groups[test_idx]))}")

    rows = []
    for n in N_GRID:
        if n > len(pool_idx):
            continue
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            uniq = rng.permutation(np.unique(pool_groups))
            picked: list[int] = []
            for g in uniq:
                if len(picked) >= n:
                    break
                members = pool_idx[pool_groups == g]
                picked.extend(members[: n - len(picked)].tolist())
            sub = np.array(picked)
            for name in ["mean_baseline", "random_forest", "xgboost"]:
                mae = R.run_composition(name, X[sub], y[sub], X_te, y_te, seed)
                rows.append(dict(n=n, seed=seed, model=name, arm="composition_all",
                                 test_mae=mae))
        done = pd.DataFrame(rows)
        print(f"n={n:4d} " + "  ".join(
            f"{m}={done[(done.n==n)&(done.model==m)].test_mae.mean():.3f}"
            for m in ["mean_baseline", "random_forest", "xgboost"]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "composition_extension.csv", index=False)
    print()
    print(df.groupby(["model", "n"])["test_mae"].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
