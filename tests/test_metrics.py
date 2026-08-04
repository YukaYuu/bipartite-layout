import numpy as np
import pytest

from bipartite_layout.metrics import compute_cluster_quality, compute_nn_distance_cv, compute_separation_metrics


def test_nn_distance_cv_grid_is_near_zero():
    grid = np.array([[x, y] for x in range(5) for y in range(5)], dtype=float)
    cv = compute_nn_distance_cv(grid)
    assert cv < 0.2


def test_nn_distance_cv_outlier_increases_cv():
    grid = np.array([[x, y] for x in range(5) for y in range(5)], dtype=float)
    cv_grid = compute_nn_distance_cv(grid)
    outlier = grid.copy()
    outlier[0] = [100.0, 100.0]
    cv_outlier = compute_nn_distance_cv(outlier)
    assert cv_outlier > cv_grid


def test_nn_distance_cv_single_point_is_nan():
    assert np.isnan(compute_nn_distance_cv(np.array([[0.0, 0.0]])))


def test_cluster_quality_perfect_match_when_clusters_are_well_separated():
    """仮想エッジが無い(全ノードが孤立)場合、Louvainは各ノードを個別コミュニティにする。
    user側2点・movie側2点がそれぞれ十分離れた座標にあれば、k-meansで完全に正解クラスタを
    再現できるはずなので、CQは両方とも1.0になる。"""
    nodes = ["u_0", "u_1", "m_0", "m_1"]
    is_user = np.array([True, True, False, False])
    common_deg = np.zeros((4, 4))
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cq_user, cq_movie = compute_cluster_quality(nodes, is_user, common_deg, coords)
    assert cq_user == pytest.approx(1.0)
    assert cq_movie == pytest.approx(1.0)


def test_cluster_quality_nan_when_fewer_than_two_ground_truth_clusters():
    """u_0-u_1間に強い仮想エッジ(common_deg>0)を入れるとLouvainがuser側を1コミュニティに
    まとめてしまい、正解クラスタが1個未満になるためcq_userはnanを返す、という契約をロックする
    (main_split_nmi_size_comparisonが検証していたのと同じ現象のmetrics層での単体テスト版)。"""
    nodes = ["u_0", "u_1", "m_0", "m_1"]
    is_user = np.array([True, True, False, False])
    common_deg = np.zeros((4, 4))
    common_deg[0, 1] = common_deg[1, 0] = 0.9
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cq_user, cq_movie = compute_cluster_quality(nodes, is_user, common_deg, coords)
    assert np.isnan(cq_user)
    assert cq_movie == pytest.approx(1.0)


def test_separation_metrics_returns_finite_values(matrices):
    nodes, node_idx, common_deg, weight, is_user = matrices
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, size=(len(nodes), 2))
    centroid_sep, nn_ratio = compute_separation_metrics(coords, is_user)
    assert np.isfinite(centroid_sep)
    assert np.isfinite(nn_ratio)
