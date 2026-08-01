"""ProcessPoolExecutor workers for the B-vs-C and B/C/D alpha sweeps."""

from bipartite_layout.direction import calc_direction_alignment_score
from bipartite_layout.layout_core import compute_layout_method
from bipartite_layout.metrics import compute_cluster_quality, compute_separation_metrics


def _bc_seed_worker(args):
    """run_b_vs_c_alpha_sweepの1(method, seed)分の計算(ProcessPoolExecutorに渡すため関数化)。"""
    (method, common_deg, weight, effective_alpha, nodes, node_idx, is_user,
     direction_precomputed, seed, gamma) = args
    coords, _, converged, _, _ = compute_layout_method(
        method, common_deg, weight, effective_alpha, nodes, node_idx, is_user,
        direction_precomputed, seed=seed, gamma=gamma
    )
    if converged:
        dir_score = calc_direction_alignment_score(coords, nodes, node_idx, direction_precomputed)
        nn_ratio = compute_separation_metrics(coords, is_user)[1]
        cq_user, cq_movie = compute_cluster_quality(nodes, is_user, common_deg, coords)
    else:
        dir_score, nn_ratio, cq_user, cq_movie = (float("nan"),) * 4
    return converged, dir_score, nn_ratio, cq_user, cq_movie


def _bcd_seed_worker(args):
    """run_b_c_d_alpha_sweepの1(method, seed)分の計算(ProcessPoolExecutorに渡すため関数化)。"""
    (method, common_deg, weight, alpha, nodes, node_idx, is_user,
     direction_precomputed, seed, gamma, anchor_weight) = args
    coords, _, converged, _, _ = compute_layout_method(
        method, common_deg, weight, alpha, nodes, node_idx, is_user,
        direction_precomputed, seed=seed, gamma=gamma, anchor_weight=anchor_weight
    )
    if converged:
        dir_score = calc_direction_alignment_score(coords, nodes, node_idx, direction_precomputed)
        nn_ratio = compute_separation_metrics(coords, is_user)[1]
        cq_user, cq_movie = compute_cluster_quality(nodes, is_user, common_deg, coords)
    else:
        dir_score, nn_ratio, cq_user, cq_movie = (float("nan"),) * 4
    return converged, dir_score, nn_ratio, cq_user, cq_movie
