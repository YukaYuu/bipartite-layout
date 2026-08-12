import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bipartite_layout.config import DEFAULT_CONFIG
from bipartite_layout.data.movielens import load_movielens_graph
from bipartite_layout.experiment_context import prepare_experiment_context
from bipartite_layout.experiments.koala_sweeps import plot_koala_epsilon_floor

if __name__ == "__main__":
    # 使い方: python3 scripts/koala_epsilon_floor_fine.py [real_edge_epsilon]
    real_edge_epsilon = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    M = load_movielens_graph(DEFAULT_CONFIG.paths.data_path)
    ctx = prepare_experiment_context(M)
    alphas = np.round(np.arange(0.0, 1.01, 0.1), 1)
    eps_str = str(real_edge_epsilon).replace(".", "")
    plot_koala_epsilon_floor(
        ctx, alphas, filename=f"koala_epsilon_floor_fine_eps{eps_str}.png",
        real_edge_epsilon=real_edge_epsilon,
    )
