"""Multi-seed layout computation, including the ProcessPoolExecutor worker.

Kept separate from layout_core.py so the pickle target for spawned worker
processes (bipartite_layout.layout_workers._multi_seed_worker) stays a
lightweight module with no plotting imports.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bipartite_layout.layout_core import compute_layout_method
from bipartite_layout.metrics import compute_cluster_metrics, compute_separation_metrics


def _multi_seed_worker(args):
    """compute_layout_multi_seedの1seed分の計算(ProcessPoolExecutorに渡すため関数化)。"""
    common_deg, weight, alpha, nodes, is_user, seed = args
    coords, final_stress, converged, n_iter, grad_norm = compute_layout_method(
        "B", common_deg, weight, alpha, nodes, None, is_user, None, seed=seed, gamma=0.0
    )
    centroid_sep, nn_ratio = compute_separation_metrics(coords, is_user)
    n_clusters_user, noise_user = compute_cluster_metrics(coords, is_user)
    n_clusters_movie, noise_movie = compute_cluster_metrics(coords, ~is_user)
    return (coords, final_stress, centroid_sep, nn_ratio, n_clusters_user, n_clusters_movie,
            noise_user, noise_movie, converged, n_iter, grad_norm)


def compute_layout_multi_seed(common_deg, weight, alpha, nodes, is_user, n_seeds=8, n_workers=None):
    """
    複数のseedでレイアウトを計算し、局所解による結果のばらつきを確認する
    (alpha混合のみ = compute_layout_method("B", gamma=0.0)を複数seedで実行)。

    n_workersを指定すると(例: n_workers=4)、seedごとの計算をProcessPoolExecutorで
    並列実行する。デフォルトNoneは従来通り逐次実行(結果は完全に同一、速度のみ異なる)。
    """
    tasks = [(common_deg, weight, alpha, nodes, is_user, seed) for seed in range(n_seeds)]
    if n_workers:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_multi_seed_worker, tasks))
    else:
        results = [_multi_seed_worker(t) for t in tasks]

    best_coords, best_stress = None, np.inf
    centroid_seps, nn_ratios = [], []
    n_clusters_user_list, n_clusters_movie_list = [], []
    noise_ratio_user_list, noise_ratio_movie_list = [], []
    converged_list, n_iter_list, grad_norm_list = [], [], []

    for (coords, final_stress, centroid_sep, nn_ratio, n_clusters_user, n_clusters_movie,
         noise_user, noise_movie, converged, n_iter, grad_norm) in results:
        centroid_seps.append(centroid_sep)
        nn_ratios.append(nn_ratio)
        n_clusters_user_list.append(n_clusters_user)
        n_clusters_movie_list.append(n_clusters_movie)
        noise_ratio_user_list.append(noise_user)
        noise_ratio_movie_list.append(noise_movie)
        converged_list.append(converged)
        n_iter_list.append(n_iter)
        grad_norm_list.append(grad_norm)

        if final_stress < best_stress:
            best_stress = final_stress
            best_coords = coords

    return (best_coords,
            np.array(centroid_seps), np.array(nn_ratios),
            np.array(n_clusters_user_list, dtype=float),
            np.array(n_clusters_movie_list, dtype=float),
            np.array(noise_ratio_user_list, dtype=float),
            np.array(noise_ratio_movie_list, dtype=float),
            np.array(converged_list, dtype=bool),
            np.array(n_iter_list, dtype=float),
            np.array(grad_norm_list, dtype=float))
