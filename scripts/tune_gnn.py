"""Learning rate search for the structural arm.

The central claim of this study is that structural models trail composition
models at every label count we can reach. That claim is only worth anything if
the structural models were given a fair chance, so we search the one
hyperparameter they are most sensitive to and report the search rather than
asserting a default was fine.

Selection is on the inner validation split, which is carved out of the training
subsample. The test set is never consulted during selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_crossover as R  # noqa: E402

LRS = [3e-4, 1e-3, 3e-3]
MODELS = ["cgcnn", "alignn_small"]
SEEDS = [0, 1, 2]
N_TRAIN = 200
OUT = Path("results/crossover")


def main() -> None:
    R.DEVICE = "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(R.CACHE / "features.parquet")
    graphs = torch.load(R.CACHE / "graphs.pt", weights_only=False)
    y = feats["y"].to_numpy(dtype=float)
    groups = feats["reduced"].to_numpy()

    pool_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=R.TEST_FRACTION, random_state=42).split(
            np.zeros(len(y)), y, groups
        )
    )
    graphs_te = [graphs[i] for i in test_idx]

    rows = []
    for model in MODELS:
        for lr in LRS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                pool_groups = groups[pool_idx]
                uniq = rng.permutation(np.unique(pool_groups))
                picked: list[int] = []
                for g in uniq:
                    if len(picked) >= N_TRAIN:
                        break
                    members = pool_idx[pool_groups == g]
                    picked.extend(members[: N_TRAIN - len(picked)].tolist())
                graphs_tr = [graphs[i] for i in picked]

                mae, n_params, val = R.run_gnn(
                    model, graphs_tr, graphs_te, seed, lr=lr, return_val=True,
                    train_groups=groups[np.array(picked)],
                )
                rows.append(
                    dict(model=model, lr=lr, seed=seed, val_mae=val, test_mae=mae,
                         n_params=n_params)
                )
                print(f"{model:13s} lr={lr:<7g} seed={seed}  val={val:.3f}  test={mae:.3f}",
                      flush=True)
                pd.DataFrame(rows).to_csv(OUT / "lr_search.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== selection on VALIDATION (test shown only for reporting) ===")
    summary = df.groupby(["model", "lr"])[["val_mae", "test_mae"]].mean().round(3)
    print(summary.to_string())
    print()
    for model in MODELS:
        sub = df[df.model == model].groupby("lr")[["val_mae", "test_mae"]].mean()
        best = sub["val_mae"].idxmin()
        print(f"{model}: best lr by validation = {best:g}  "
              f"-> test MAE {sub.loc[best, 'test_mae']:.3f}")


if __name__ == "__main__":
    main()
