# GNN-SSE-Discovery

Machine learning for lithium solid electrolyte ionic conductivity.

This code supports **"The missing crossover: learning curves for composition and
structural machine learning on lithium solid electrolytes."** It measures how the
gap between composition-based and crystal-graph models behaves as the number of
training labels grows. Over 25 to 200 labels the gap does not close, it roughly
doubles, and three architectures spanning a 40x range of capacity (98k to 4.04M
parameters) all land in the same band. Labels come from OBELiX (Therrien et al.),
a published, expert-curated dataset, with splits grouped by composition at every
level so that near-duplicate entries cannot leak between train and test.

## Layout

    scripts/prepare_obelix.py          build the modelling set from OBELiX
    scripts/run_crossover.py           the learning curve
    scripts/tune_gnn.py                learning-rate search
    scripts/run_feature_analysis.py    SHAP + descriptor redundancy
    scripts/run_composition_extension.py   composition arm out to n=400
    scripts/run_screen_obelix.py       Materials Project screen (needs MP_API_KEY)
    scripts/fetch_snapshot.py          fetch the MP snapshot
    scripts/plot_crossover.py          the figure

    src/gnnsse/features/obelix_graph.py   image-aware, occupancy-aware graphs
    src/gnnsse/models/{alignn,cgcnn}.py   the structural models
    src/gnnsse/data/mp_fetch.py           MP retrieval

    results/                           every number the paper reports, as CSV
    tests/                             tests for the graph builder and models

## Reproducing the paper

```bash
conda env create -f environment.yml && conda activate gnn-sse
# or: pip install -e .

python scripts/prepare_obelix.py      # 599 -> 558 usable -> 281 with structures
python scripts/run_crossover.py --device cpu --tag fast --seeds 0,1,2,3,4 \
    --models mean_baseline,random_forest,xgboost,cgcnn,alignn_small
python scripts/run_crossover.py --device cpu --tag ref --models alignn
python scripts/tune_gnn.py
python scripts/run_feature_analysis.py
python scripts/run_composition_extension.py
python scripts/plot_crossover.py

export MP_API_KEY="..."               # only for the screen
python scripts/fetch_snapshot.py
python scripts/run_screen_obelix.py
```

Results land in `results/` and are committed, so the tables can be checked
without rerunning anything. Roughly 19 CPU-hours total on an Apple M1, 16 GB;
the reference ALIGNN alone is about 13 of those.

## Tests

`pytest` covers the two properties of the graph builder that fail silently on
experimental structures (bond angles under periodic images, partial occupancy),
that ALIGNN activations stay bounded through the stack, and that activation
checkpointing does not change the gradient.

## Note

The graph models are re-implementations rather than the reference code bases.
Validating against `usnistgov/alignn` is a worthwhile independent check.
