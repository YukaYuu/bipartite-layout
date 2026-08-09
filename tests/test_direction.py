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


def test_continuous_mode_is_not_restricted_to_same_community_pairs(small_graph, matrices):
    """
    先生からのご指摘への対応: mode="cluster"(既定)は、Louvainで分割した同一
    コミュニティ内のペアだけを方向整列の対象にするため、コミュニティ数が少ないと
    レイアウト全体が少数の離散的な方向に強制収束してしまう(格子状になる)。
    mode="continuous"はコミュニティ分割を経由せず、類似度が閾値を超えるすべての
    ペアを対象にするため、対象ペア数がcluster版以上になるはず(異なるコミュニティに
    属していたせいで除外されていたペアも含まれるようになるため)。
    """
    nodes, node_idx, common_deg, weight, is_user = matrices
    _, edge_labels_cluster, cluster_precomputed = get_edge_direction_cached(
        small_graph, node_idx, mode="cluster"
    )
    _, edge_labels_continuous, continuous_precomputed = get_edge_direction_cached(
        small_graph, node_idx, mode="continuous"
    )

    n_communities = len(set(edge_labels_cluster.tolist()))
    n_pairs_cluster = len(cluster_precomputed["pair_i"])
    n_pairs_continuous = len(continuous_precomputed["pair_i"])

    # edge_labelsは可視化用の色分けとして両モードで同じロジックにより計算される。
    assert np.array_equal(edge_labels_cluster, edge_labels_continuous)
    # 複数コミュニティが存在する限り、continuousの方が対象ペア数が真に多いはず
    # (異なるコミュニティ間のペアも含まれるようになるため)。
    if n_communities > 1:
        assert n_pairs_continuous > n_pairs_cluster


def test_get_edge_direction_cached_rejects_unknown_mode(small_graph, matrices):
    nodes, node_idx, common_deg, weight, is_user = matrices
    try:
        get_edge_direction_cached(small_graph, node_idx, mode="bogus")
        assert False, "expected ValueError for an unknown mode"
    except ValueError:
        pass
