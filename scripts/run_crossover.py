"""Learning curve study: where, if anywhere, do structural models overtake composition?

Trains composition and structure models on identical nested subsamples of the
OBELiX modelling set and evaluates every one on the same held out test set, so
that the only variable is the number of labels.

Splits are grouped by reduced composition: no composition may appear in both the
training pool and the test set. Without this, near duplicate entries leak and
every curve is optimistic.

Usage:  python scripts/run_crossover.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnsse.models.alignn import ALIGNN  # noqa: E402
from gnnsse.models.cgcnn import CGCNN  # noqa: E402

CACHE = Path("data/obelix_cache")
RESULTS = Path("results/crossover")
TEST_FRACTION = 0.2
N_GRID = [25, 50, 100, 150, 200]
SEEDS = [0, 1, 2]
# CPU measured faster than MPS for these scatter heavy line graph updates
# (6.3s vs 9.0s per epoch over 32 graphs for the reference ALIGNN), so CPU is
# the default; override with --device.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# Composition arm
# ----------------------------------------------------------------------


def _composition_model(name: str, seed: int):
    if name == "mean_baseline":
        return DummyRegressor(strategy="mean")
    if name == "random_forest":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("rf", RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)),
            ]
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(name)


def run_composition(name, X_tr, y_tr, X_te, y_te, seed) -> float:
    model = _composition_model(name, seed)
    model.fit(X_tr, y_tr)
    return float(np.mean(np.abs(model.predict(X_te) - y_te)))


# ----------------------------------------------------------------------
# Structure arm
# ----------------------------------------------------------------------


def _gnn(name: str) -> nn.Module:
    if name == "cgcnn":
        return CGCNN(atom_fea_len=64, n_conv=3, h_fea_len=128, n_tasks=1)
    if name == "alignn":
        # Reference configuration from Choudhary & DeCost (2021). Checkpointed:
        # a batch of 32 can carry up to 1.19M line graph edges, which exhausts
        # memory without it. Checkpointing is exact (verified to 2e-7) so these
        # runs stay comparable to the rest of the grid.
        return ALIGNN(hidden_size=256, n_layers=4, n_gcn_layers=4, n_tasks=1,
                      checkpoint=True)
    if name == "alignn_small":
        # Deliberately low capacity: tests whether the structural arm's behaviour
        # is a property of ALIGNN's inductive bias or merely of its size.
        return ALIGNN(hidden_size=64, n_layers=2, n_gcn_layers=2, n_tasks=1)
    raise ValueError(name)


def _grouped_val_split(
    n_items: int, groups: np.ndarray | None, seed: int, frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into (fit, val) so that no group spans both sides.

    The inner validation split must be grouped, not random. The training
    subsample is drawn by composition, so it routinely holds several entries of
    the same formula; a random inner split would scatter those across fit and
    validation, and the model would early stop against near duplicates of what
    it just trained on. Validation error then collapses while test error does
    not, and any hyperparameter chosen on that signal is chosen on noise.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.permutation(n_items)
        n_val = max(4, int(frac * n_items))
        return idx[n_val:], idx[:n_val]

    uniq = rng.permutation(np.unique(groups))
    target = max(4, int(frac * n_items))
    val_groups: set = set()
    n_val = 0
    for g in uniq:
        if n_val >= target:
            break
        val_groups.add(g)
        n_val += int((groups == g).sum())
    val_idx = np.array([i for i in range(n_items) if groups[i] in val_groups])
    fit_idx = np.array([i for i in range(n_items) if groups[i] not in val_groups])
    if len(fit_idx) == 0 or len(val_idx) == 0:  # degenerate; fall back
        idx = rng.permutation(n_items)
        n_v = max(1, int(frac * n_items))
        return idx[n_v:], idx[:n_v]
    return fit_idx, val_idx


def run_gnn(
    name: str,
    train_graphs: list,
    test_graphs: list,
    seed: int,
    max_epochs: int = 150,
    patience: int = 20,
    lr: float = 1e-3,
    return_val: bool = False,
    train_groups: np.ndarray | None = None,
    micro_batch: int | None = None,
):
    """Train one GNN and return (test MAE in log10 S/cm, trainable parameter count).

    With ``return_val`` the inner validation MAE is returned as a third element,
    so that a hyperparameter can be selected without consulting the test set.
    ``train_groups`` carries the composition of each training graph so that the
    inner validation split can be grouped; without it the split leaks.
    """
    torch.manual_seed(seed)

    fit_idx, val_idx = _grouped_val_split(len(train_graphs), train_groups, seed)
    fit = [train_graphs[i] for i in fit_idx]
    val = [train_graphs[i] for i in val_idx]

    # Standardise the target on the fitting split only.
    y_fit = np.array([float(g.y) for g in fit])
    mu, sigma = float(y_fit.mean()), float(y_fit.std() or 1.0)

    model = _gnn(name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Batch size is fixed at 32 for every model. Do not micro batch to save
    # memory here: both architectures use batch normalisation, so splitting the
    # step changes the batch statistics and the accumulated gradient is not the
    # full batch gradient (we measured an 8.5% discrepancy). Memory is handled
    # by activation checkpointing inside the model instead, which is exact.
    fit_loader = DataLoader(fit, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=16)
    test_loader = DataLoader(test_graphs, batch_size=16)

    def evaluate(loader) -> float:
        model.eval()
        errs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                pred = model(batch) * sigma + mu
                errs.append((pred - batch.y.view(-1)).abs().cpu())
        return float(torch.cat(errs).mean())

    best_val, best_state, waited = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for batch in fit_loader:
            batch = batch.to(DEVICE)
            target = (batch.y.view(-1) - mu) / sigma
            loss = nn.functional.mse_loss(model(batch), target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_mae = evaluate(val_loader)
        if val_mae < best_val - 1e-4:
            best_val, waited = val_mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if return_val:
        return evaluate(test_loader), n_params, best_val
    return evaluate(test_loader), n_params


# ----------------------------------------------------------------------


def main() -> None:
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, small grid, smoke test")
    ap.add_argument("--device", default=DEVICE, help="cpu | mps | cuda")
    ap.add_argument("--models", default="", help="comma separated subset to run")
    ap.add_argument("--tag", default="all", help="output file suffix")
    ap.add_argument("--seeds", default="", help="comma separated seeds, overrides default")
    ap.add_argument("--ngrid", default="", help="comma separated n values, overrides default")
    args = ap.parse_args()
    DEVICE = args.device

    n_grid, seeds = N_GRID, SEEDS
    if args.quick:
        n_grid, seeds = [25, 50], [0]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    if args.ngrid:
        n_grid = [int(v) for v in args.ngrid.split(",")]

    comp_models = ["mean_baseline", "random_forest", "xgboost"]
    gnn_models = ["cgcnn", "alignn_small", "alignn"]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        comp_models = [m for m in comp_models if m in wanted]
        gnn_models = [m for m in gnn_models if m in wanted]

    RESULTS.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(CACHE / "features.parquet")
    graphs = torch.load(CACHE / "graphs.pt", weights_only=False)
    assert len(feats) == len(graphs), "feature/graph cache out of sync"

    feat_cols = [c for c in feats.columns if c.startswith("MagpieData")]
    X = feats[feat_cols].to_numpy(dtype=float)
    y = feats["y"].to_numpy(dtype=float)
    groups = feats["reduced"].to_numpy()

    # Fixed grouped test set, identical for every model and every N.
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=42)
    pool_idx, test_idx = next(splitter.split(X, y, groups))
    print(f"device={DEVICE}  pool={len(pool_idx)}  test={len(test_idx)}  "
          f"({len(set(groups[pool_idx]) & set(groups[test_idx]))} shared compositions)")

    X_te, y_te = X[test_idx], y[test_idx]
    graphs_te = [graphs[i] for i in test_idx]
    pool_groups = groups[pool_idx]

    rows = []
    for n in n_grid:
        if n > len(pool_idx):
            continue
        for seed in seeds:
            # Subsample the pool by GROUP so smaller N stays leakage free too.
            rng = np.random.default_rng(seed)
            uniq = rng.permutation(np.unique(pool_groups))
            picked, chosen = [], set()
            for g in uniq:
                if len(picked) >= n:
                    break
                members = pool_idx[pool_groups == g]
                take = members[: n - len(picked)]
                picked.extend(take.tolist())
                chosen.add(g)
            sub = np.array(picked)

            X_tr, y_tr = X[sub], y[sub]
            graphs_tr = [graphs[i] for i in sub]

            for name in comp_models:
                t0 = time.time()
                mae = run_composition(name, X_tr, y_tr, X_te, y_te, seed)
                rows.append(dict(n=n, seed=seed, model=name, arm="composition",
                                 test_mae=mae, n_params=np.nan, secs=time.time() - t0))
                print(f"n={n:4d} seed={seed} {name:14s} MAE={mae:.3f}", flush=True)

            for name in gnn_models:
                t0 = time.time()
                mae, n_params = run_gnn(
                    name, graphs_tr, graphs_te, seed, train_groups=groups[sub]
                )
                rows.append(dict(n=n, seed=seed, model=name, arm="structure",
                                 test_mae=mae, n_params=n_params, secs=time.time() - t0))
                print(f"n={n:4d} seed={seed} {name:14s} MAE={mae:.3f} "
                      f"({n_params:,} params, {time.time()-t0:.0f}s)", flush=True)

            pd.DataFrame(rows).to_csv(RESULTS / f"raw_{args.tag}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"raw_{args.tag}.csv", index=False)
    summary = (
        df.groupby(["arm", "model", "n"])["test_mae"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(RESULTS / f"summary_{args.tag}.csv", index=False)
    (RESULTS / f"config_{args.tag}.json").write_text(
        json.dumps(
            {
                "n_grid": n_grid,
                "seeds": seeds,
                "test_fraction": TEST_FRACTION,
                "n_pool": int(len(pool_idx)),
                "n_test": int(len(test_idx)),
                "grouping": "reduced composition",
                "target": "log10 sigma (S/cm)",
                "metric": "test MAE",
                "device": DEVICE,
            },
            indent=2,
        )
    )
    print("\n=== SUMMARY (test MAE) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
