import networkx as nx
import numpy as np
import pytest

from bipartite_layout.caching import get_matrices_cached, get_small_subgraph_cached


@pytest.fixture(scope="session")
def full_graph():
    """user-movie二部グラフ全体(build_small_subgraphのサンプリング元)。決定的に構築する。"""
    rng = np.random.default_rng(0)
    M = nx.Graph()
    users = [f"u_{i}" for i in range(120)]
    movies = [f"m_{i}" for i in range(60)]
    for u in users:
        chosen = rng.choice(movies, size=6, replace=False)
        for m in chosen:
            M.add_edge(u, m)
    return M


@pytest.fixture(scope="session")
def small_graph(full_graph):
    """full_graphから抽出した小さいサブグラフ(実際の実験と同じ抽出ロジック)。"""
    return get_small_subgraph_cached(full_graph, n_seed_movies=5, n_users_per_movie=10)


@pytest.fixture(scope="session")
def matrices(small_graph):
    """degree重み・top_k=3・threshold=0.0での標準的な行列一式。"""
    return get_matrices_cached(small_graph, "degree", 0.0, 3)
