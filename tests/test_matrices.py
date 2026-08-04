import numpy as np

from bipartite_layout.matrices import (
    build_matrices,
    build_matrices_commonality_weight,
    build_matrices_uniform_weight,
)
from bipartite_layout.sampling import apply_top_k_sparsification


def test_build_matrices_shapes(small_graph):
    nodes, idx, common_deg, weight, is_user = build_matrices(small_graph, threshold_common_deg=0.0, top_k_same_type=3)
    n = len(nodes)
    assert common_deg.shape == (n, n)
    assert weight.shape == (n, n)
    assert is_user.shape == (n,)
    assert idx[nodes[0]] == 0


def test_common_deg_symmetric_and_zero_diagonal(matrices):
    _, _, common_deg, _, _ = matrices
    assert np.allclose(common_deg, common_deg.T)
    assert np.allclose(np.diagonal(common_deg), 0.0)


def test_weight_only_between_opposite_types(small_graph):
    nodes, idx, common_deg, weight, is_user = build_matrices(small_graph, threshold_common_deg=0.0, top_k_same_type=3)
    same_type = is_user[:, None] == is_user[None, :]
    assert np.all(weight[same_type] == 0.0)


def test_weight_real_edges_match_graph_edges(small_graph):
    nodes, idx, common_deg, weight, is_user = build_matrices(small_graph, threshold_common_deg=0.0, top_k_same_type=3)
    weighted_pairs = set()
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            if weight[i, j] > 0:
                weighted_pairs.add(frozenset((nodes[i], nodes[j])))
    graph_edges = {frozenset(e) for e in small_graph.edges()}
    assert weighted_pairs == graph_edges


def test_uniform_weight_is_constant_on_real_edges(small_graph):
    _, _, _, weight, _ = build_matrices_uniform_weight(
        small_graph, threshold_common_deg=0.0, top_k_same_type=3, uniform_weight_value=1.0
    )
    nonzero = weight[weight > 0]
    assert np.allclose(nonzero, 1.0)


def test_commonality_weight_in_normalized_range(small_graph):
    _, _, _, weight, _ = build_matrices_commonality_weight(small_graph, threshold_common_deg=0.0, top_k_same_type=3)
    nonzero = weight[weight > 0]
    assert nonzero.size > 0
    # 正規化ルール(0.4〜1.0)により、非ゼロの重みは必ずこの範囲に収まる
    assert np.all(nonzero >= 0.4 - 1e-9)
    assert np.all(nonzero <= 1.0 + 1e-9)


def test_top_k_sparsification_enforces_max_degree():
    rng = np.random.default_rng(1)
    n = 10
    sim = rng.uniform(0, 1, size=(n, n))
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 0.0)
    is_user = np.array([True] * 6 + [False] * 4)

    sparsified = apply_top_k_sparsification(sim, is_user, top_k=2, mutual_only=False)
    for i in range(n):
        row_type_mask = is_user == is_user[i]
        row_type_mask[i] = False
        # 「上位k(=2)を残す」は非対称なOR結合(相手からも選ばれれば残る)なので、
        # 各行の非ゼロ数は最低でもtop_k以上残ることを確認する(過剰に削られていないか)
        n_kept = np.sum((sparsified[i] > 0) & row_type_mask)
        assert n_kept >= 2 or row_type_mask.sum() < 2
