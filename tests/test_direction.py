import numpy as np

from bipartite_layout import direction
from bipartite_layout.caching import get_edge_direction_cached


def test_numba_and_numpy_direction_alignment_agree(small_graph, matrices):
    """layout_coreのstress_and_gradと同様、方向整列項もnumba/numpyの2実装が
    数値的に一致することを確認する(こちらもこの一致を守る唯一のテスト)。"""
    nodes, node_idx, common_deg, weight, is_user = matrices
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(small_graph, node_idx)

    N = len(nodes)
    rng = np.random.default_rng(0)
    coords_flat = rng.uniform(0, 1, size=N * 2)

    args = (
        coords_flat, N,
        direction_precomputed["m_idx_arr"], direction_precomputed["u_idx_arr"],
        direction_precomputed["pair_i"], direction_precomputed["pair_j"], direction_precomputed["pair_w"],
    )
    loss_loop, grad_loop = direction._direction_alignment_loop(*args)
    loss_np, grad_np = direction._direction_alignment_numpy(*args)

    assert np.isclose(loss_loop, loss_np, rtol=1e-6)
    assert np.allclose(grad_loop, grad_np, atol=1e-8)


def test_direction_alignment_score_is_normalized(small_graph, matrices):
    nodes, node_idx, common_deg, weight, is_user = matrices
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(small_graph, node_idx)
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, size=(len(nodes), 2))
    score = direction.calc_direction_alignment_score(coords, nodes, node_idx, direction_precomputed)
    assert not np.isnan(score)
    assert 0.0 <= score <= 1.0
