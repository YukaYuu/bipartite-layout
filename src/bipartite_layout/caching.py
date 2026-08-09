"""Memoization layer over sampling/matrices/direction computations.

Only ever relied upon within a single process/run — caches are plain
module-level dicts keyed by id(M)/id(G).
"""

from bipartite_layout.direction import (
    calc_edge_similarity,
    cluster_edges_by_community,
    precompute_direction_pairs,
    precompute_direction_pairs_continuous,
)
from bipartite_layout.matrices import (
    build_matrices,
    build_matrices_commonality_weight,
    build_matrices_uniform_weight,
)
from bipartite_layout.sampling import build_small_subgraph

_small_subgraph_cache = {}
_matrices_cache = {}
_edge_direction_cache = {}


def get_small_subgraph_cached(M, **kwargs):
    """build_small_subgraph(M, **kwargs)をメモ化する(同じM・同じkwargsなら再計算しない)。"""
    key = (id(M), tuple(sorted(kwargs.items())))
    if key not in _small_subgraph_cache:
        _small_subgraph_cache[key] = build_small_subgraph(M, **kwargs)
    return _small_subgraph_cache[key]


def get_matrices_cached(G, weight_mode, threshold_common_deg, top_k_same_type,
                         mutual_top_k_only=False, uniform_weight_value=1.0):
    """
    build_matrices/build_matrices_uniform_weight/build_matrices_commonality_weightを
    メモ化する(同じG・同じweight_mode・同じパラメータなら再計算しない)。
    """
    key = (id(G), weight_mode, threshold_common_deg, top_k_same_type,
           mutual_top_k_only, uniform_weight_value)
    if key not in _matrices_cache:
        if weight_mode == "uniform":
            _matrices_cache[key] = build_matrices_uniform_weight(
                G, threshold_common_deg=threshold_common_deg, top_k_same_type=top_k_same_type,
                mutual_top_k_only=mutual_top_k_only, uniform_weight_value=uniform_weight_value
            )
        elif weight_mode == "degree":
            _matrices_cache[key] = build_matrices(
                G, threshold_common_deg=threshold_common_deg, top_k_same_type=top_k_same_type,
                mutual_top_k_only=mutual_top_k_only
            )
        elif weight_mode == "commonality":
            _matrices_cache[key] = build_matrices_commonality_weight(
                G, threshold_common_deg=threshold_common_deg, top_k_same_type=top_k_same_type,
                mutual_top_k_only=mutual_top_k_only
            )
        else:
            raise ValueError(
                f"unknown weight_mode: {weight_mode!r} (must be 'uniform', 'degree', or 'commonality')"
            )
    return _matrices_cache[key]


def get_edge_direction_cached(G, node_idx, mode="cluster"):
    """
    calc_edge_similarity + cluster_edges_by_community + precompute_direction_pairs(系)の
    結果をG・modeごとにメモ化する。戻り値: (edges, edge_labels, direction_precomputed)。

    mode="cluster"(既定、既存挙動): 方向整列の対象を、Louvainで分割した同一コミュニティ内の
    ペアだけに限定する。コミュニティ数が少ないと、レイアウト全体が少数の離散的な方向に
    強制的に収束してしまう(格子状になる)ことがある(先生からのご指摘)。

    mode="continuous": コミュニティ分割を経由せず、類似度が閾値を超える「すべての」
    エッジペアを、類似度をそのまま重みとして方向整列の対象にする。edge_labelsは
    (可視化での色分け用に)cluster同様に計算されるが、方向整列の力の計算には使われない。
    """
    key = (id(G), mode)
    if key not in _edge_direction_cache:
        edges, similarity = calc_edge_similarity(G)
        edge_labels = cluster_edges_by_community(edges, similarity, sim_threshold=0.0, resolution=1.0)
        if mode == "cluster":
            direction_precomputed = precompute_direction_pairs(edges, similarity, edge_labels, node_idx)
        elif mode == "continuous":
            direction_precomputed = precompute_direction_pairs_continuous(edges, similarity, node_idx)
        else:
            raise ValueError(f"unknown mode: {mode!r} (must be 'cluster' or 'continuous')")
        _edge_direction_cache[key] = (edges, edge_labels, direction_precomputed)
    return _edge_direction_cache[key]
