"""Cache the OBELiX modelling set: composition features, crystal graphs, targets.

Writes ``data/obelix_cache/{features.parquet, graphs.pt, meta.json}``.

Entries are kept only when they carry BOTH a usable measured conductivity and a
crystal structure, because the composition and structure arms of the learning
curve must be trained on identical samples for the comparison to mean anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnsse.features.obelix_graph import structure_to_graph  # noqa: E402

# OBELiX encodes "reported as low, value unspecified" as 1e-15. Those are
# censored placeholders, not measurements, so they are excluded.
CENSOR_FLOOR = -14.0

OUT_DIR = Path("data/obelix_cache")
CUTOFF = 8.0
MAX_NEIGHBORS = 12


def main() -> None:
    from obelix import OBELiX

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ob = OBELiX(data_path="data/obelix")
    df = ob.dataframe.copy()
    df["sigma"] = pd.to_numeric(df["Ionic conductivity (S cm-1)"], errors="coerce")
    df["y"] = np.log10(df["sigma"].where(df["sigma"] > 0))

    n_total = len(df)
    df = df[df["y"].notna() & (df["y"] > CENSOR_FLOOR)]
    n_real = len(df)
    df = df[df["structure"].notna()]
    n_struct = len(df)

    print(f"OBELiX total            : {n_total}")
    print(f"  usable measurement    : {n_real}  (dropped {n_total - n_real} censored)")
    print(f"  and has structure     : {n_struct}")

    records = []
    graphs: list = []
    failures: list[tuple[str, str]] = []

    for idx, row in df.iterrows():
        try:
            graph = structure_to_graph(
                row["structure"], cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS
            )
        except Exception as exc:  # keep going; report honestly at the end
            failures.append((str(idx), f"{type(exc).__name__}: {exc}"))
            continue
        graph.y = torch.tensor([float(row["y"])], dtype=torch.float32)
        graphs.append(graph)
        records.append(
            {
                "entry_id": str(idx),
                "composition": row["True Composition"],
                "reduced": row["Reduced Composition"],
                "family": row["Family"],
                "space_group": row["Space group #"],
                "doi": row["DOI"],
                "y": float(row["y"]),
            }
        )

    meta_df = pd.DataFrame(records)
    print(f"  graphs built          : {len(graphs)}  (failed {len(failures)})")
    for eid, err in failures[:5]:
        print(f"    ! {eid}: {err}")

    # Composition descriptors (Magpie) on the same rows, in the same order.
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition

    feat_df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
        meta_df, col_id="composition", ignore_errors=True
    )
    magpie = ElementProperty.from_preset("magpie")
    feat_df = magpie.featurize_dataframe(
        feat_df, col_id="composition_obj", ignore_errors=True
    )
    feat_cols = [c for c in feat_df.columns if c.startswith("MagpieData")]
    feat_df = feat_df.drop(columns=["composition_obj"])

    print(f"  Magpie descriptors    : {len(feat_cols)}")
    bad = feat_df[feat_cols].isna().any(axis=1).sum()
    print(f"  rows with any NaN feat: {bad}")

    feat_df.to_parquet(OUT_DIR / "features.parquet", index=False)
    torch.save(graphs, OUT_DIR / "graphs.pt")
    (OUT_DIR / "meta.json").write_text(
        json.dumps(
            {
                "n_obelix_total": n_total,
                "n_usable_measurement": n_real,
                "n_with_structure": n_struct,
                "n_graphs": len(graphs),
                "n_failures": len(failures),
                "failures": failures,
                "magpie_n_features": len(feat_cols),
                "graph_cutoff_A": CUTOFF,
                "graph_max_neighbors": MAX_NEIGHBORS,
                "censor_floor_log10": CENSOR_FLOOR,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT_DIR}/  (features.parquet, graphs.pt, meta.json)")


if __name__ == "__main__":
    main()
