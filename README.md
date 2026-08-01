# bipartite-layout

A layout-optimization method for bipartite graphs (evaluated on MovieLens
user–movie data and DBLP author–paper data), built around stress
majorization with a continuous knob between two competing notions of
"closeness":

- **Same-type structure** — users near other similar users, movies near
  other similar movies (via Jaccard similarity over shared neighbors).
- **Cross-type structure** — users near the movies they actually rated /
  authors near the papers they actually wrote (the real bipartite edges,
  weighted by one of three interchangeable schemes: degree-based,
  uniform, or a "commonality of neighboring nodes" scheme).

A continuous parameter `alpha` mixes the two: `alpha=0` recovers a
standard real-edges-only force layout, `alpha=1` produces a fully
same-type-separated layout, and everything in between is a genuine blend
rather than a discrete switch. On top of that, an optional **direction
alignment** term (method C) nudges edges within the same structural
cluster to point the same way, and **method D** achieves a similar effect
more cheaply via sequential bundling: converge the alpha-mixing objective
first, then anchor that layout and optimize direction alignment as a
lightweight post-process — trading a small amount of alignment quality for
much better convergence stability than jointly optimizing both objectives
at once (method C).

## Package layout

```
src/bipartite_layout/
├── config.py              # frozen dataclasses: paths, sampling sizes, graph-build/cluster/layout params
├── data/
│   ├── movielens.py        # MovieLens ratings -> bipartite graph
│   └── dblp.py              # DBLP XML (streaming parse) -> bipartite graph, with pickle caching
├── sampling.py              # sample a small subgraph from the full dataset
├── matrices.py              # the three same-type/cross-type weight-matrix builders
├── caching.py                # memoizes sampling/matrices/direction-precompute within one run
├── direction.py              # edge-similarity clustering + the direction-alignment stress term
├── layout_core.py            # the core stress-majorization objective + methods A/B/C/D
├── layout_workers.py         # multi-seed layout runs (ProcessPoolExecutor-parallelizable)
├── metrics.py                 # separation/cluster/NMI/Cluster-Quality metrics
├── plotting.py                 # layout visualization (alpha grids, ablation grids)
├── diagnostics.py              # one-off exploratory reports (Jaccard distributions, null models, ...)
├── experiment_context.py       # shared helpers (ExperimentContext, save_figure, default_alpha_grid, ...)
└── experiments/
    ├── workers.py               # ProcessPoolExecutor targets for the B-vs-C / B-vs-C-vs-D sweeps
    ├── alpha_sweeps.py           # the core alpha-sweep experiments
    ├── ablations.py               # repulsion/init ablations, weight-mode image comparison
    └── batteries.py               # the full per-dataset experiment battery + entry point

scripts/run_experiments.py         # thin entry point: runs the full MovieLens + DBLP battery
```

Both `numba` (JIT-compiled hot loops) and a numpy-only fallback are
supported transparently — `layout_core.py` and `direction.py` each detect
`numba` availability independently and dispatch accordingly.

## Running it

```bash
pip install -e .          # add the `fast` extra (pip install -e ".[fast]") for numba acceleration
python scripts/run_experiments.py
```

This loads MovieLens and DBLP, and for each runs the full experiment
battery (`experiments.batteries.run_all_experiments_for_dataset`):
sampling-parameter sweeps, the alpha-vs-method (B/C/D) sweeps across all
three weight schemes, NMI/Cluster-Quality comparisons across sampling
scale, the repulsion/init ablation study, and alpha-grid visualizations
— writing PNGs and printed tables to the current directory.

`config.py`'s `PathsConfig` points at the raw MovieLens/DBLP files under
`~/Downloads/`; adjust `DEFAULT_CONFIG.paths` if your data lives elsewhere.

## Design notes

- **Weight design follows Gansner–Koren–North (2004)**: higher edge
  weight maps to a *shorter* target distance in the stress objective,
  matching how weighted stress majorization is defined in the graph
  drawing literature — the three weighting schemes differ only in how
  that weight is computed.
- **Repulsion is applied to all node pairs**, not just connected ones.
  Combined with sparse same-type attraction (top-k nearest neighbors
  only), most node pairs feel pure repulsion — which is why layouts at
  extreme alpha values can look lattice-like; this is an artifact of
  repulsion-dominated regions having no attractive counterforce, not a
  bug (see `experiments/ablations.py:main_repulsion_and_init_ablation`,
  which demonstrates this directly by disabling same-type repulsion).
- **Cluster Quality (CQ)**, in `metrics.py`, measures whether a layout's
  geometric clusters (k-means on coordinates) actually correspond to the
  graph's structural clusters (Louvain communities on the same-type
  similarity graph) — a best-Jaccard-match, size-weighted score. This
  catches failure modes that separation/cluster-count metrics alone miss:
  a layout can separate users from movies cleanly while still scrambling
  which users cluster with which.
