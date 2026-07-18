"""Screen Materials Project lithium entries with a composition model trained on OBELiX.

The model is a regressor on log10 sigma, so candidates are ranked by a predicted
conductivity directly. Labels are the 558 usable OBELiX measurements. The scope
is lithium, because OBELiX is lithium only and a model trained on lithium has no
standing to rank sodium or magnesium compounds.

What this ranks is ionic conductivity. It is not a deployability screen: it says
nothing about electrochemical stability window, reactivity against a cathode, or
interfacial behaviour, all of which decide whether a fast conductor is usable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SNAPSHOT = Path("data/mp_snapshots/mp_snapshot_latest.parquet")
FEATS_ALL = Path("data/obelix_cache/features_all.parquet")
MP_FEATS = Path("data/obelix_cache/mp_li_features.parquet")
OUT = Path("results/screen")

EHULL_MAX = 0.025      # eV/atom
BANDGAP_MIN = 1.0      # eV
TOXIC = {"Pb", "Cd", "Hg", "As"}
TOP_K = 50


def reduced(formula: str) -> str | None:
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def featurize_mp() -> pd.DataFrame:
    if MP_FEATS.exists():
        return pd.read_parquet(MP_FEATS)

    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition

    df = pd.read_parquet(SNAPSHOT, columns=[
        "material_id", "formula", "band_gap", "energy_above_hull",
        "is_stable", "spacegroup", "working_ion",
    ])
    # Lithium only: the training labels are lithium only.
    df = df[df["formula"].str.contains("Li", na=False)]
    df = df[df["formula"].apply(lambda f: "Li" in (Composition(f).as_dict() if reduced(f) else {}))]
    print(f"MP lithium entries: {len(df)}")

    df = StrToComposition(target_col_id="obj").featurize_dataframe(
        df, col_id="formula", ignore_errors=True)
    df = ElementProperty.from_preset("magpie").featurize_dataframe(
        df, col_id="obj", ignore_errors=True)
    df = df.drop(columns=["obj"])
    MP_FEATS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MP_FEATS, index=False)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(FEATS_ALL).dropna(subset=["y"])
    cols = [c for c in train.columns if c.startswith("MagpieData")]
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)),
    ])
    model.fit(train[cols].to_numpy(dtype=float), train["y"].to_numpy(dtype=float))
    print(f"trained on {len(train)} OBELiX lithium measurements")

    mp = featurize_mp()
    gates: dict[str, int] = {"MP lithium entries": len(mp)}

    mp["reduced"] = mp["formula"].map(reduced)
    mp = mp[mp["reduced"].notna()]

    def has_toxic(f: str) -> bool:
        try:
            return any(e.symbol in TOXIC for e in Composition(f).elements)
        except Exception:
            return True

    mp = mp[~mp["formula"].map(has_toxic)]
    gates["no toxic heavy elements"] = len(mp)

    mp = mp[mp["energy_above_hull"].fillna(999) <= EHULL_MAX]
    gates[f"E_hull <= {EHULL_MAX*1000:.0f} meV/atom"] = len(mp)

    mp = mp[mp["band_gap"].fillna(0) >= BANDGAP_MIN]
    gates[f"band gap >= {BANDGAP_MIN} eV"] = len(mp)

    # The mechanical gate is omitted deliberately: the Materials Project returned
    # no elastic moduli for any entry in this snapshot, so the gate could only
    # ever pass everything, and reporting it as a filter would be misleading.

    X = mp[cols].to_numpy(dtype=float)
    mp["pred_log10_sigma"] = model.predict(X)

    # Per-tree spread as a crude dispersion estimate, not a calibrated interval.
    trees = np.stack([t.predict(model.named_steps["impute"].transform(X))
                      for t in model.named_steps["rf"].estimators_])
    mp["pred_spread"] = trees.std(axis=0)

    known = set(train["reduced"].astype(str).map(lambda f: reduced(f) or f))
    mp["in_training"] = mp["reduced"].isin(known)
    gates["scored by the model"] = len(mp)

    ranked = mp.sort_values("pred_log10_sigma", ascending=False)
    novel = ranked[~ranked["in_training"]]
    gates[f"top {TOP_K} novel"] = min(TOP_K, len(novel))

    top = novel.head(TOP_K)[
        ["material_id", "formula", "spacegroup", "pred_log10_sigma",
         "pred_spread", "energy_above_hull", "band_gap"]
    ].reset_index(drop=True)
    top.to_csv(OUT / "top50_li.csv", index=False)
    ranked[["material_id", "formula", "pred_log10_sigma", "pred_spread",
            "in_training"]].to_csv(OUT / "ranked_all.csv", index=False)
    (OUT / "gates.json").write_text(json.dumps(gates, indent=2))

    print()
    for k, v in gates.items():
        print(f"  {k:35s} {v:>7,}")
    print()
    print(top.head(12).to_string(index=False))
    print()
    print(f"predicted log10 sigma range on shortlist: "
          f"{top.pred_log10_sigma.min():.2f} to {top.pred_log10_sigma.max():.2f}")
    print(f"mean per-tree spread on shortlist       : {top.pred_spread.mean():.2f}")


if __name__ == "__main__":
    main()
