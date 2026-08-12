import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bipartite_layout.config import DEFAULT_CONFIG
from bipartite_layout.data.movielens import load_movielens_graph
from bipartite_layout.experiment_context import prepare_experiment_context
from bipartite_layout.experiments.koala_sweeps import plot_koala_real_edge_comparison

if __name__ == "__main__":
    M = load_movielens_graph(DEFAULT_CONFIG.paths.data_path)
    ctx = prepare_experiment_context(M)
    alphas = np.round(np.arange(0.0, 1.01, 0.1), 1)
    plot_koala_real_edge_comparison(ctx, alphas)
