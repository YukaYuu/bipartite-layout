"""Alpha-sweep experiment drivers (B vs C, B vs C vs D, gamma sweep, etc.)."""

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from bipartite_layout.caching import get_edge_direction_cached, get_matrices_cached, get_small_subgraph_cached
from bipartite_layout.config import DEFAULT_CONFIG
from bipartite_layout.diagnostics import (
    analyze_common_neighbor_popularity_bias,
    analyze_degree_and_null_model,
    analyze_virtual_edge_graph_structure,
    analyze_virtual_edge_graph_structure_quiet,
    compare_threshold_candidates,
    compute_jaccard_distribution_stats,
    inspect_common_deg_distribution,
    inspect_common_deg_distribution_by_type,
)
from bipartite_layout.direction import (
    calc_direction_alignment_score,
    calc_edge_similarity,
    cluster_edges_by_community,
    precompute_direction_pairs,
)
from bipartite_layout.experiment_context import save_figure
from bipartite_layout.experiments.workers import _bc_seed_worker, _bcd_seed_worker
from bipartite_layout.layout_core import compute_layout_method
from bipartite_layout.layout_workers import compute_layout_multi_seed
from bipartite_layout.matrices import build_matrices
from bipartite_layout.metrics import compute_cluster_metrics, compute_separation_metrics
from bipartite_layout.plotting import plot_and_save


def run_full_experiment(M, build_kwargs, label, n_seeds=20,
                         threshold=DEFAULT_CONFIG.graph_build.threshold_common_deg, run_heavy_diagnostics=True):
    G = get_small_subgraph_cached(M, **build_kwargs)
    n_user_nodes = sum(1 for n in G.nodes() if n.startswith("u_"))
    n_movie_nodes = sum(1 for n in G.nodes() if n.startswith("m_"))
    print(f"\n{'=' * 20} [{label}] サブグラフ実験開始 {'=' * 20}")
    print(f"サブグラフ(固定): {G.number_of_nodes()} ノード "
          f"(user={n_user_nodes}, movie={n_movie_nodes}), {G.number_of_edges()} エッジ")
    print(f"サンプリングパラメータ: {build_kwargs}")

    if run_heavy_diagnostics:
        analyze_common_neighbor_popularity_bias(
            G, M, save_path=f"popularity_bias_{label}.png"
        )
        analyze_degree_and_null_model(G, save_path=f"degree_null_model_{label}.png")

        cv_stats = compute_jaccard_distribution_stats(G)
        print(f"\n[user-user] Jaccard分布: 平均={cv_stats['user_mean']:.3f}, "
              f"標準偏差={cv_stats['user_std']:.3f}, CV={cv_stats['user_cv']:.3f}")
        print(f"[movie-movie] Jaccard分布: 平均={cv_stats['movie_mean']:.3f}, "
              f"標準偏差={cv_stats['movie_std']:.3f}, CV={cv_stats['movie_cv']:.3f}")

        threshold_candidates = [0.0, 0.1, 0.167, 0.25, 0.375]
        print(f"\n{'threshold':>10} | {'user modularity':>16} | {'movie modularity':>17}")
        print("-" * 50)
        for th in threshold_candidates:
            nodes, idx, common_deg, weight, is_user = build_matrices(G, threshold_common_deg=th)
            mod_user, mod_movie = analyze_virtual_edge_graph_structure_quiet(
                nodes, common_deg, is_user, th
            )
            print(f"{th:>10.3f} | {mod_user:>16.3f} | {mod_movie:>17.3f}")

        values = inspect_common_deg_distribution(
            G, save_path=f"common_deg_distribution_{label}.png"
        )
        if len(values):
            candidates = sorted(set(
                list(np.round(np.percentile(values, [0, 25, 50, 75, 90, 100]), 3))
            ))
            print("\n閾値候補ごとの比較 (分位点ベース、user/movieプール):")
            compare_threshold_candidates(G, candidates)

        inspect_common_deg_distribution_by_type(
            G, save_path=f"common_deg_distribution_by_type_{label}.png"
        )

    if threshold is None:
        print("\nthreshold(閾値)が未設定です。閾値を決めてから再実行してください。")
        return

    nodes, idx, common_deg, weight, is_user = get_matrices_cached(
        G, "degree", threshold, DEFAULT_CONFIG.graph_build.top_k_same_type, DEFAULT_CONFIG.graph_build.mutual_top_k_only)

    if run_heavy_diagnostics:
        analyze_virtual_edge_graph_structure(nodes, common_deg, is_user, threshold)

    fine_range = np.round(np.arange(0.25, 0.76, 0.1), 2)
    alphas = sorted(set([0.0, 1.0] + list(fine_range)))

    fig, axes = plt.subplots(1, len(alphas), figsize=(4.0 * len(alphas), 4.0))

    print(f"\n[{label}] 各alphaにつき {n_seeds} 個のseedで実行し、ばらつきを確認します。")
    header = (f"{'alpha':>6} | {'centroid_sep':>16} | {'nn_ratio':>16} | "
              f"{'n_cluster(user)':>16} | {'n_cluster(movie)':>17}")
    print(f"\n{header}")
    print("-" * len(header))

    sep_means, sep_stds = [], []
    nn_means, nn_stds = [], []
    ncl_u_means, ncl_u_stds = [], []
    ncl_m_means, ncl_m_stds = [], []

    for ax, alpha in zip(axes, alphas):
        (best_coords, centroid_seps, nn_ratios,
         n_clusters_user, n_clusters_movie,
         noise_ratio_user, noise_ratio_movie,
         converged_arr, n_iter_arr, grad_norm_arr) = compute_layout_multi_seed(
            common_deg, weight, alpha, nodes, is_user, n_seeds=n_seeds
        )

        sep_mean, sep_std = centroid_seps.mean(), centroid_seps.std()
        nn_mean, nn_std = nn_ratios.mean(), nn_ratios.std()
        ncl_u_mean, ncl_u_std = np.nanmean(n_clusters_user), np.nanstd(n_clusters_user)
        ncl_m_mean, ncl_m_std = np.nanmean(n_clusters_movie), np.nanstd(n_clusters_movie)

        sep_means.append(sep_mean); sep_stds.append(sep_std)
        nn_means.append(nn_mean); nn_stds.append(nn_std)
        ncl_u_means.append(ncl_u_mean); ncl_u_stds.append(ncl_u_std)
        ncl_m_means.append(ncl_m_mean); ncl_m_stds.append(ncl_m_std)

        print(f"{alpha:>6.2f} | {sep_mean:>7.3f}±{sep_std:<7.3f} | "
              f"{nn_mean:>7.3f}±{nn_std:<7.3f} | "
              f"{ncl_u_mean:>7.2f}±{ncl_u_std:<7.2f} | "
              f"{ncl_m_mean:>7.2f}±{ncl_m_std:<7.2f}")

        n_converged = int(converged_arr.sum())
        mean_iter = n_iter_arr.mean()
        mean_grad_norm = grad_norm_arr.mean()
        print(f"        └ 収束: {n_converged}/{n_seeds}回成功, "
              f"平均反復回数={mean_iter:.1f}, 平均勾配ノルム={mean_grad_norm:.4f}")

        ax.scatter(best_coords[is_user, 0], best_coords[is_user, 1], c="tab:blue", label="user", s=60)
        ax.scatter(best_coords[~is_user, 0], best_coords[~is_user, 1], c="tab:red", label="movie", s=60)
        for i, j in G.edges():
            xi, xj = best_coords[idx[i]], best_coords[idx[j]]
            ax.plot([xi[0], xj[0]], [xi[1], xj[1]], color="gray", alpha=0.3, linewidth=0.7)
        ax.set_title(f"alpha={alpha}\nsep={sep_mean:.2f} ncl(u,m)=({ncl_u_mean:.1f},{ncl_m_mean:.1f})",
                     fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(f"[{label}] threshold = {threshold}, "
                 f"n_user={n_user_nodes}, n_movie={n_movie_nodes}")
    out_path = f"alpha_layout_th{threshold}_{label}.png"
    save_figure(fig, out_path)

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    ax1, ax2, ax3 = axes2

    ax1.errorbar(alphas, sep_means, yerr=sep_stds, marker="o", capsize=4)
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("centroid separation (normalized)")
    ax1.set_title(f"[{label}] Centroid separation vs alpha ({n_seeds} seeds)")

    ax2.errorbar(alphas, nn_means, yerr=nn_stds, marker="o", color="tab:orange", capsize=4)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("alpha")
    ax2.set_ylabel("nn_ratio (opposite / same)")
    ax2.set_title(f"[{label}] Nearest-neighbor ratio vs alpha ({n_seeds} seeds)")

    ax3.errorbar(alphas, ncl_u_means, yerr=ncl_u_stds, marker="o", color="tab:blue",
                 capsize=4, label="user")
    ax3.errorbar(alphas, ncl_m_means, yerr=ncl_m_stds, marker="s", color="tab:red",
                 capsize=4, label="movie")
    ax3.set_xlabel("alpha")
    ax3.set_ylabel("DBSCAN cluster count")
    ax3.set_title(f"[{label}] Cluster count vs alpha ({n_seeds} seeds)")
    ax3.legend()

    trend_path = f"separation_and_cluster_trend_th{threshold}_{label}.png"
    save_figure(fig2, trend_path)

    return {
        "label": label, "n_user": n_user_nodes, "n_movie": n_movie_nodes,
        "alphas": alphas, "sep_means": sep_means, "nn_means": nn_means,
        "ncl_u_means": ncl_u_means, "ncl_m_means": ncl_m_means,
    }


def gamma_sweep(common_deg, weight, nodes, node_idx, is_user, direction_precomputed,
                alpha=0.5, gammas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0), n_seeds=10):
    """STEP1: gammaを振って、収束が安定する値を探す(手法C固定)。"""
    print(f"\n=== gammaスイープ(手法C、alpha={alpha}固定) ===")
    print(f"{'gamma':>8} | {'dir_score':>10} | {'nn_ratio':>10} | {'converged':>10}")
    print("-" * 50)
    for gamma in gammas:
        dir_scores, nn_ratios, n_converged = [], [], 0
        for seed in range(n_seeds):
            coords, _, converged, _, _ = compute_layout_method(
                "C", common_deg, weight, alpha, nodes, node_idx, is_user,
                direction_precomputed, seed=seed, gamma=gamma
            )
            dir_scores.append(calc_direction_alignment_score(coords, nodes, node_idx, direction_precomputed))
            nn_ratios.append(compute_separation_metrics(coords, is_user)[1])
            n_converged += int(converged)
        print(f"{gamma:>8.2f} | {np.nanmean(dir_scores):>10.3f} | {np.nanmean(nn_ratios):>10.3f} | "
              f"{n_converged:>7d}/{n_seeds}")
    print("\n見方: 収束率が高く保たれつつ、dir_scoreがgamma=0(手法B相当)より明確に高い、"
          "一番小さいgammaを選ぶのが良い候補です。")


def run_three_method_comparison(G, common_deg, weight, nodes, node_idx, is_user,
                                 alphas, n_seeds=10, gamma=0.1):
    edges, similarity = calc_edge_similarity(G)
    edge_labels = cluster_edges_by_community(edges, similarity, sim_threshold=0.0, resolution=1.0)
    direction_precomputed = precompute_direction_pairs(edges, similarity, edge_labels, node_idx)
    print(f"\nエッジクラスタ数: {edge_labels.max() + 1}, 方向整列の対象ペア数: {len(direction_precomputed['pair_i'])}")

    def run_multi_seed(method, alpha):
        nn_ratios, ncl_u_list, ncl_m_list, dir_scores, n_converged = [], [], [], [], 0
        for seed in range(n_seeds):
            coords, _, converged, _, _ = compute_layout_method(
                method, common_deg, weight, alpha, nodes, node_idx, is_user,
                direction_precomputed, seed=seed, gamma=gamma
            )
            nn_ratios.append(compute_separation_metrics(coords, is_user)[1])
            ncl_u_list.append(compute_cluster_metrics(coords, is_user)[0])
            ncl_m_list.append(compute_cluster_metrics(coords, ~is_user)[0])
            dir_scores.append(calc_direction_alignment_score(coords, nodes, node_idx, direction_precomputed))
            n_converged += int(converged)
        return {"nn_ratio": np.nanmean(nn_ratios), "ncl_u": np.nanmean(ncl_u_list),
                "ncl_m": np.nanmean(ncl_m_list), "dir_score": np.nanmean(dir_scores), "converged": n_converged}

    print(f"\n{'method':>8} | {'alpha':>6} | {'nn_ratio':>10} | {'ncl(u)':>8} | {'ncl(m)':>8} | "
          f"{'dir_score':>10} | {'converged':>10}")
    print("-" * 80)

    res_a = run_multi_seed("A", alpha=0.5)
    print(f"{'A':>8} | {'-':>6} | {res_a['nn_ratio']:>10.3f} | {res_a['ncl_u']:>8.2f} | "
          f"{res_a['ncl_m']:>8.2f} | {res_a['dir_score']:>10.3f} | {res_a['converged']:>7d}/{n_seeds}")

    for alpha in alphas:
        res_b = run_multi_seed("B", alpha)
        res_c = run_multi_seed("C", alpha)
        print(f"{'B':>8} | {alpha:>6.2f} | {res_b['nn_ratio']:>10.3f} | {res_b['ncl_u']:>8.2f} | "
              f"{res_b['ncl_m']:>8.2f} | {res_b['dir_score']:>10.3f} | {res_b['converged']:>7d}/{n_seeds}")
        print(f"{'C':>8} | {alpha:>6.2f} | {res_c['nn_ratio']:>10.3f} | {res_c['ncl_u']:>8.2f} | "
              f"{res_c['ncl_m']:>8.2f} | {res_c['dir_score']:>10.3f} | {res_c['converged']:>7d}/{n_seeds}")


def run_three_method_comparison_with_plots(G, common_deg, weight, nodes, node_idx, is_user,
                                            alphas, gamma=0.1, plot_seed=0, out_dir="."):
    edges, similarity = calc_edge_similarity(G)
    edge_labels = cluster_edges_by_community(edges, similarity, sim_threshold=0.0, resolution=1.0)
    direction_precomputed = precompute_direction_pairs(edges, similarity, edge_labels, node_idx)

    # 手法A(alpha非依存、1回だけ)
    coords_a, _, conv_a, _, _ = compute_layout_method(
        "A", common_deg, weight, 0.5, nodes, node_idx, is_user,
        direction_precomputed, seed=plot_seed, gamma=gamma
    )
    dir_score_a = calc_direction_alignment_score(coords_a, nodes, node_idx, direction_precomputed)
    plot_and_save(coords_a, is_user, G, node_idx,
                  f"Method A (direction alignment only)\ndir_score={dir_score_a:.3f}, 収束={conv_a}",
                  f"{out_dir}/layout_A.png", edge_labels=edge_labels)

    for alpha in alphas:
        coords_b, _, conv_b, _, _ = compute_layout_method(
            "B", common_deg, weight, alpha, nodes, node_idx, is_user,
            direction_precomputed, seed=plot_seed, gamma=gamma
        )
        dir_score_b = calc_direction_alignment_score(coords_b, nodes, node_idx, direction_precomputed)
        nn_ratio_b = compute_separation_metrics(coords_b, is_user)[1]
        plot_and_save(coords_b, is_user, G, node_idx,
                      f"重み付けのみ, alpha={alpha}\n方向整列スコア={dir_score_b:.3f}, 異種/同種 最近傍距離比={nn_ratio_b:.3f}, 収束={conv_b}",
                      f"{out_dir}/layout_B_alpha{alpha}.png", edge_labels=edge_labels)

        coords_c, _, conv_c, _, _ = compute_layout_method(
            "C", common_deg, weight, alpha, nodes, node_idx, is_user,
            direction_precomputed, seed=plot_seed, gamma=gamma
        )
        dir_score_c = calc_direction_alignment_score(coords_c, nodes, node_idx, direction_precomputed)
        nn_ratio_c = compute_separation_metrics(coords_c, is_user)[1]
        plot_and_save(coords_c, is_user, G, node_idx,
                      f"方向整列との組み合わせ, alpha={alpha}\n方向整列スコア={dir_score_c:.3f}, 異種/同種 最近傍距離比={nn_ratio_c:.3f}, 収束={conv_c}",
                      f"{out_dir}/layout_C_alpha{alpha}.png", edge_labels=edge_labels)


def beta_transform(alpha):
    """
    β変換: α<0.5ではβ=0.5α、α>=0.5ではβ=0.75+0.5(α-0.5)というピースワイズ線形変換。
    α=0.5の前後でβが0.25→0.75へ不連続にジャンプする(β≈0.5付近の値を意図的に避ける)。
    「同種・異種の重みがβ≈0.5付近で拮抗すること自体が、中間alphaで観測される収束不安定性の
    主因かどうか」を検証するための変換(その領域を素通りさせても不安定性が残るなら、
    主因ではないと言える)。
    """
    if alpha < 0.5:
        return 0.5 * alpha
    else:
        return 0.75 + 0.5 * (alpha - 0.5)


def run_b_vs_c_alpha_sweep(G, dataset_label, weight_mode="uniform", alphas=None, n_seeds=20, gamma=0.1,
                           alpha_transform=None, n_workers=None):
    """
    手法B(alpha混合のみ)とC(alpha混合+方向整列)を、同じサブグラフGに対してalphaを
    振りながら比較する。weight_modeで異種ペア(実エッジ)の重みづけ方式を切り替えられる:
      - "uniform"    : build_matrices_uniform_weight (先生のご提案の検証用、全ての実エッジを同じ重みにする)
      - "degree"     : build_matrices (次数ベースのresource allocation指数で重みづけ、0.4〜1.0に正規化)
      - "commonality": build_matrices_commonality_weight (隣接ノードの共通度=sparsify前の
                       同種Jaccard類似度から、実エッジの自然長を設計する)
    同じ関数・同じG・同じ絞り込みロジックで重み方式だけを変えられるため、
    「実エッジの重み設計がα=1.0付近の劣化や中間alphaでの収束不安定性の主因かどうか」を
    直接比較できる。

    alpha_transformを指定すると、alphasでスイープする値(表示・比較の基準となる「見かけの
    alpha」)はそのままに、実際にcompute_layout_methodへ渡す値だけをalpha_transform(alpha)に
    差し替える(例: beta_transform)。同じ見かけ上のalphaグリッドのまま、内部で使う
    混合係数だけを変えた場合の効果を比較できる。

    n_workersを指定すると(例: n_workers=4)、各alphaにつきB/C×n_seeds回のレイアウト計算を
    ProcessPoolExecutorで並列実行する。デフォルトNoneは従来通り逐次実行で、結果(収束数・
    各種平均値)は並列実行でも完全に同一になる(seedで決まる計算内容自体は変わらないため)。
    """
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, weight_mode, DEFAULT_CONFIG.graph_build.threshold_common_deg, DEFAULT_CONFIG.graph_build.top_k_same_type, DEFAULT_CONFIG.graph_build.mutual_top_k_only
    )
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    if alphas is None:
        alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)

    print(f"\n{'-' * 15} [{dataset_label}] B vs C alphaスイープ (weight_mode={weight_mode}) {'-' * 15}")
    print(f"{'alpha':>6} | {'B conv':>8} | {'C conv':>8} | {'B dir(conv)':>12} | {'C dir(conv)':>12} | {'B nn(conv)':>11} | {'C nn(conv)':>11}")

    rows = []
    executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers else None
    try:
        for alpha in alphas:
            effective_alpha = alpha_transform(alpha) if alpha_transform is not None else alpha

            tasks = [
                (method, common_deg, weight, effective_alpha, nodes, node_idx, is_user,
                 direction_precomputed, seed, gamma)
                for method in ("B", "C") for seed in range(n_seeds)
            ]
            if executor is not None:
                results = list(executor.map(_bc_seed_worker, tasks))
            else:
                results = [_bc_seed_worker(t) for t in tasks]

            b_results, c_results = results[:n_seeds], results[n_seeds:]
            n_conv_b = sum(1 for conv, _, _, _, _ in b_results if conv)
            n_conv_c = sum(1 for conv, _, _, _, _ in c_results if conv)
            dir_b = [d for conv, d, _, _, _ in b_results if conv]
            dir_c = [d for conv, d, _, _, _ in c_results if conv]
            nn_b = [n for conv, _, n, _, _ in b_results if conv]
            nn_c = [n for conv, _, n, _, _ in c_results if conv]
            # cq_*はそのtypeの正解クラスタが2個未満だとnanを含みうるため、
            # 平均を取る前に非nan値だけへ絞る(np.nanmeanの"Mean of empty slice"警告を避けるため)。
            cq_user_b = [v for conv, _, _, v, _ in b_results if conv and not np.isnan(v)]
            cq_user_c = [v for conv, _, _, v, _ in c_results if conv and not np.isnan(v)]
            cq_movie_b = [v for conv, _, _, _, v in b_results if conv and not np.isnan(v)]
            cq_movie_c = [v for conv, _, _, _, v in c_results if conv and not np.isnan(v)]

            row = {
                "alpha": alpha, "n_conv_b": n_conv_b, "n_conv_c": n_conv_c,
                "dir_b": np.nanmean(dir_b) if dir_b else float("nan"),
                "dir_c": np.nanmean(dir_c) if dir_c else float("nan"),
                "nn_b": np.nanmean(nn_b) if nn_b else float("nan"),
                "nn_c": np.nanmean(nn_c) if nn_c else float("nan"),
                "cq_user_b": np.mean(cq_user_b) if cq_user_b else float("nan"),
                "cq_user_c": np.mean(cq_user_c) if cq_user_c else float("nan"),
                "cq_movie_b": np.mean(cq_movie_b) if cq_movie_b else float("nan"),
                "cq_movie_c": np.mean(cq_movie_c) if cq_movie_c else float("nan"),
            }
            rows.append(row)

            db = f"{row['dir_b']:.3f}" if dir_b else "N/A"
            dc = f"{row['dir_c']:.3f}" if dir_c else "N/A"
            nb = f"{row['nn_b']:.3f}" if nn_b else "N/A"
            nc = f"{row['nn_c']:.3f}" if nn_c else "N/A"
            print(f"{alpha:>6.2f} | {n_conv_b:>4d}/{n_seeds:<3d}| {n_conv_c:>4d}/{n_seeds:<3d}| "
                  f"{db:>12} | {dc:>12} | {nb:>11} | {nc:>11}")
    finally:
        if executor is not None:
            executor.shutdown()

    print("\n[Cluster Quality (CQ): レイアウトのk-meansクラスタが真の構造クラスタとどれだけ一致するか]")
    print(f"{'alpha':>6} | {'B cq(u)':>9} | {'C cq(u)':>9} | {'B cq(m)':>9} | {'C cq(m)':>9}")
    for row in rows:
        print(f"{row['alpha']:>6.2f} | {row['cq_user_b']:>9.3f} | {row['cq_user_c']:>9.3f} | "
              f"{row['cq_movie_b']:>9.3f} | {row['cq_movie_c']:>9.3f}")

    return {
        "nodes": nodes, "node_idx": node_idx, "common_deg": common_deg, "weight": weight,
        "is_user": is_user, "edges": edges, "edge_labels": edge_labels, "rows": rows,
    }


def run_b_c_d_alpha_sweep(G, dataset_label, weight_mode="uniform", alphas=None, n_seeds=20, gamma=0.1,
                          anchor_weight=1.0, methods=("B", "C", "D"), n_workers=None):
    """
    手法B(alpha混合のみ)・C(alpha混合+方向整列を同時最適化)・
    D(alpha混合で収束させた後、そのレイアウトを弱くアンカーしつつ方向整列のみを
    追加で最適化する逐次的束ね)を比較する。

    「方向整列を同時に最適化することの不安定性コストは、逐次的に(まずレイアウトを
    決めてから)束ねることで解消できるか」を検証するための実験。Dの収束率がCより
    明確に高く、dir_score/nn_ratioの低下がわずかであれば、「同時最適化はその
    不安定化コストに見合わない」という具体的な根拠になる。逆にDのdir_scoreが
    大きく劣化するなら、同時最適化でなければ得られない効果がある、という根拠になる。

    dir_score/nn_ratioに加えて、Cluster Quality(CQ、Claudeとの相談で追加)も
    集計する。CQは「レイアウトの座標をk-meansでクラスタリングしたものが、
    common_degのLouvainコミュニティ(真の構造的クラスタ)とどれだけ一致するか」を
    Jaccard類似度のベストマッチで測る指標(compute_cluster_quality参照)。
    nn_ratio/n_clusterでは捉えられない「レイアウトが構造的クラスタを空間的に
    忠実に再現できているか」を直接測るための指標。

    anchor_weightはstage2でレイアウトをどれだけ強く固定するかを決めるパラメータ
    (大きいほどstage1のレイアウトを厳密に保つが、方向整列による調整の余地が減る)。
    run_b_vs_c_alpha_sweepとは別の独立した関数にしてあり、既存のB/C比較の
    実験結果には一切影響しない。
    """
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, weight_mode, DEFAULT_CONFIG.graph_build.threshold_common_deg, DEFAULT_CONFIG.graph_build.top_k_same_type, DEFAULT_CONFIG.graph_build.mutual_top_k_only
    )
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    if alphas is None:
        alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)

    print(f"\n{'-' * 15} [{dataset_label}] B/C/D alphaスイープ "
          f"(weight_mode={weight_mode}, anchor_weight={anchor_weight}) {'-' * 15}")
    print("[収束数・方向整列スコア・分離度]")
    header = (f"{'alpha':>6} | " + " | ".join(f"{m + ' conv':>9}" for m in methods) + " | " +
              " | ".join(f"{m + ' dir':>9}" for m in methods) + " | " +
              " | ".join(f"{m + ' nn':>9}" for m in methods))
    print(header)

    rows = []
    executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers else None
    try:
        for alpha in alphas:
            tasks = [
                (method, common_deg, weight, alpha, nodes, node_idx, is_user,
                 direction_precomputed, seed, gamma, anchor_weight)
                for method in methods for seed in range(n_seeds)
            ]
            if executor is not None:
                results = list(executor.map(_bcd_seed_worker, tasks))
            else:
                results = [_bcd_seed_worker(t) for t in tasks]

            row = {"alpha": alpha}
            for mi, method in enumerate(methods):
                m_results = results[mi * n_seeds:(mi + 1) * n_seeds]
                n_conv = sum(1 for conv, _, _, _, _ in m_results if conv)
                dir_vals = [d for conv, d, _, _, _ in m_results if conv]
                nn_vals = [n for conv, _, n, _, _ in m_results if conv]
                # cq_user_vals/cq_movie_valsは、そのtypeの正解クラスタが2個未満だった
                # 収束済みseedについてnanを含みうる(compute_cluster_quality参照)。
                # 全部nanのままnp.nanmeanに渡すと"Mean of empty slice"警告が出るため、
                # 事前に非nan値だけへ絞ってから平均する。
                cq_user_vals = [cqu for conv, _, _, cqu, _ in m_results if conv]
                cq_movie_vals = [cqm for conv, _, _, _, cqm in m_results if conv]
                cq_user_valid = [v for v in cq_user_vals if not np.isnan(v)]
                cq_movie_valid = [v for v in cq_movie_vals if not np.isnan(v)]
                row[f"n_conv_{method}"] = n_conv
                row[f"dir_{method}"] = np.nanmean(dir_vals) if dir_vals else float("nan")
                row[f"nn_{method}"] = np.nanmean(nn_vals) if nn_vals else float("nan")
                row[f"cq_user_{method}"] = np.mean(cq_user_valid) if cq_user_valid else float("nan")
                row[f"cq_movie_{method}"] = np.mean(cq_movie_valid) if cq_movie_valid else float("nan")
            rows.append(row)

            conv_cells = " | ".join(f"{row[f'n_conv_{m}']:>6d}/{n_seeds:<2d}" for m in methods)
            dir_cells = " | ".join(f"{row[f'dir_{m}']:>9.3f}" for m in methods)
            nn_cells = " | ".join(f"{row[f'nn_{m}']:>9.3f}" for m in methods)
            print(f"{alpha:>6.2f} | {conv_cells} | {dir_cells} | {nn_cells}")
    finally:
        if executor is not None:
            executor.shutdown()

    print("\n[Cluster Quality (CQ): レイアウトのk-meansクラスタが真の構造クラスタとどれだけ一致するか]")
    cq_header = (f"{'alpha':>6} | " + " | ".join(f"{m + ' cq(u)':>10}" for m in methods) + " | " +
                 " | ".join(f"{m + ' cq(m)':>10}" for m in methods))
    print(cq_header)
    for row in rows:
        cq_u_cells = " | ".join(f"{row[f'cq_user_{m}']:>10.3f}" for m in methods)
        cq_m_cells = " | ".join(f"{row[f'cq_movie_{m}']:>10.3f}" for m in methods)
        print(f"{row['alpha']:>6.2f} | {cq_u_cells} | {cq_m_cells}")

    return {
        "nodes": nodes, "node_idx": node_idx, "common_deg": common_deg, "weight": weight,
        "is_user": is_user, "edges": edges, "edge_labels": edge_labels, "rows": rows,
    }
