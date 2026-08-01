"""Ablation studies: repulsion/init ablation, weight-mode image comparison, balance experiment."""

import numpy as np

from bipartite_layout.caching import get_edge_direction_cached, get_matrices_cached, get_small_subgraph_cached
from bipartite_layout.config import LARGE_BUILD_KWARGS, MUTUAL_TOP_K_ONLY, SMALL_BUILD_KWARGS, THRESHOLD_COMMON_DEG, TOP_K_SAME_TYPE
from bipartite_layout.diagnostics import run_sampling_parameter_sweep
from bipartite_layout.experiments.alpha_sweeps import run_full_experiment
from bipartite_layout.layout_core import compute_layout_method
from bipartite_layout.metrics import compute_nn_distance_cv
from bipartite_layout.plotting import plot_ablation_grid, plot_alpha_grid


def main_repulsion_and_init_ablation(M, dataset_label, alphas=None, gamma=0.1, seed=0, method="B"):
    """
    2つのablationを実証・比較する:
    (1) 同タイプノード間の反発を切ると、格子状に見える均一配置は消えるか
        (repel_same_type=False)
    (2) 初期配置をalphaに依存させず完全ランダムにしても、二層配置はストレス関数の
        alpha混合項だけから自然に立ち上がるか(random_init=True)
    画像グリッド(plot_ablation_grid)に加え、最近傍距離CV(compute_nn_distance_cv、
    小さいほど格子的)の数表も出力し、視覚的印象だけでなく数値でも比較できるようにする。
    """
    print(f"\n{'#' * 20} [{dataset_label}] 反発・初期配置のablation実験 {'#' * 20}")

    G = get_small_subgraph_cached(M)
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, "degree", THRESHOLD_COMMON_DEG, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY
    )
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    if alphas is None:
        alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)

    repulsion_configs = [
        ("通常(全ペア反発)", {}),
        ("同タイプ反発なし", {"repel_same_type": False}),
    ]
    init_configs = [
        ("通常(alpha依存初期配置)", {}),
        ("完全ランダム初期配置", {"random_init": True}),
    ]

    plot_ablation_grid(G, common_deg, weight, nodes, node_idx, is_user, alphas,
                        repulsion_configs, method=method, gamma=gamma, seed=seed,
                        filename=f"ablation_repulsion_{dataset_label}.png")
    plot_ablation_grid(G, common_deg, weight, nodes, node_idx, is_user, alphas,
                        init_configs, method=method, gamma=gamma, seed=seed,
                        filename=f"ablation_init_{dataset_label}.png")

    all_configs = [("repel_all", repulsion_configs[0]), ("repel_cross_only", repulsion_configs[1]),
                   ("init_default", init_configs[0]), ("init_random", init_configs[1])]
    print(f"\n--------------- [{dataset_label}] 最近傍距離CV(小さいほど格子的、大きいほど疎密がある) ---------------")
    print("alpha".ljust(8) + "".join(f"{name:>20}" for name, _ in all_configs))
    for alpha in alphas:
        row_vals = []
        for _, (_, kwargs) in all_configs:
            coords, _, conv, _, _ = compute_layout_method(
                method, common_deg, weight, alpha, nodes, node_idx, is_user,
                direction_precomputed, seed=seed, gamma=gamma, **kwargs
            )
            row_vals.append(compute_nn_distance_cv(coords))
        print(f"{alpha:<8.2f}" + "".join(f"{v:>20.4f}" for v in row_vals))


def main_weight_mode_image_comparison(M, dataset_label, weight_modes=("uniform", "degree", "commonality"),
                                       alphas=None, gamma=0.1, seed=0):
    """
    次数ベース(degree)・均一(uniform)・共通隣接度(commonality)、指定した全ての実エッジ重み方式で
    alpha_gridの画像を生成し、直接見比べられるようにする。同じG(build_small_subgraphの結果)に
    対して重み方式だけを変えるため、レイアウトの違いが純粋に重み方式の違いに起因すると言える。
    ファイル名にweight_modeを含めるため、他の実験の出力(alpha_grid_{label}_part*.png)とも
    衝突しない。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_weight_mode_image_comparison {'#' * 20}")

    G = get_small_subgraph_cached(M)
    if alphas is None:
        alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)

    for weight_mode in weight_modes:
        nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
            G, weight_mode, THRESHOLD_COMMON_DEG, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY
        )
        print(f"\n--- [{dataset_label}] weight_mode={weight_mode} の画像を生成 ---")
        plot_alpha_grid(G, common_deg, weight, nodes, node_idx, is_user, alphas,
                         gamma=gamma, seed=seed, filename=f"alpha_grid_{dataset_label}_{weight_mode}.png")


def main_balance_experiment(M, dataset_label):
    """
    experiment_balance5.pyの本実験: 小規模サンプリングと、userを大幅に増やした
    大規模サンプリングを比較する。Mは既に読み込み済みのグラフ(MovieLensでもDBLPでも可)、
    dataset_labelは出力ファイル名・見出しの接頭辞("movielens"/"dblp"等)。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_balance_experiment {'#' * 20}")

    run_sampling_parameter_sweep(M, [SMALL_BUILD_KWARGS, LARGE_BUILD_KWARGS])

    run_full_experiment(M, SMALL_BUILD_KWARGS, label=f"{dataset_label}_small", n_seeds=20)
    run_full_experiment(M, LARGE_BUILD_KWARGS, label=f"{dataset_label}_large", n_seeds=10)

    print("\n" + "=" * 60)
    print(f"[{dataset_label}_small]と[{dataset_label}_large]、それぞれのalpha_layout_th*.pngと"
          "separation_and_cluster_trend_*.pngを見比べてください。特にuser側のn_cluster(user)が"
          "alphaに応じて動くようになったかどうかが、「userノード数が少なすぎたこと」がどれだけ"
          "非対称性に寄与していたかの直接的な確認になります。")
