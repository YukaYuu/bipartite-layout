"""Top-level experiment batteries composing the full per-dataset run."""

import numpy as np

from bipartite_layout.caching import get_edge_direction_cached, get_matrices_cached, get_small_subgraph_cached
from bipartite_layout.config import DEFAULT_CONFIG, LARGE_BUILD_KWARGS, SMALL_BUILD_KWARGS
from bipartite_layout.data.dblp import load_dblp_graph_cached
from bipartite_layout.data.movielens import load_movielens_graph
from bipartite_layout.experiments.ablations import (
    main_balance_experiment,
    main_repulsion_and_init_ablation,
    main_weight_mode_image_comparison,
)
from bipartite_layout.experiment_context import default_alpha_grid
from bipartite_layout.experiments.alpha_sweeps import beta_transform, run_b_vs_c_alpha_sweep
from bipartite_layout.metrics import compute_split_nmi
from bipartite_layout.plotting import plot_alpha_grid


def main_three_method_gamma_and_nmi(M, dataset_label):
    """
    three_method_experiment.pyの本実験: build_matrices_uniform_weightで
    手法B/Cのalphaスイープ(収束率・方向整列スコア・nn_ratio)を行い、
    続けてノードクラスタとエッジクラスタのNMIをuser/movie別に計算する。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_three_method_gamma_and_nmi {'#' * 20}")

    G = get_small_subgraph_cached(M)
    result = run_b_vs_c_alpha_sweep(G, dataset_label, weight_mode="uniform", n_seeds=20, gamma=0.1)

    nmi_all, nmi_user, nmi_movie = compute_split_nmi(
        result["nodes"], result["is_user"], result["common_deg"], result["edges"], result["edge_labels"]
    )
    print(f"全体のNMI: {nmi_all:.3f}")
    print(f"user側のみのNMI: {nmi_user:.3f}")
    print(f"movie側のみのNMI: {nmi_movie:.3f}")


def main_weight_mode_comparison(M, dataset_label, weight_modes=("uniform", "degree", "commonality"),
                                 n_seeds=15, gamma=0.1):
    """
    「実エッジの重みをどう設計するかが、α=1.0付近の劣化や中間alphaでの収束不安定性に
    どれだけ効くか」を、同じサブグラフGに対してrun_b_vs_c_alpha_sweepをweight_modesで
    指定した全方式(既定ではuniform/degree/commonalityの3方式)で実行し、直接比較する。
    weight_modeの略称は表中でu=uniform, d=degree, k=commonality(共通隣接度、
    method Cの"C"と紛らわしいkoetsuu/commonalityの頭文字)を使う。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_weight_mode_comparison {'#' * 20}")

    G = get_small_subgraph_cached(M)
    results = {
        wm: run_b_vs_c_alpha_sweep(G, dataset_label, weight_mode=wm, n_seeds=n_seeds, gamma=gamma)
        for wm in weight_modes
    }
    abbrev = {"uniform": "u", "degree": "d", "commonality": "k"}
    tags = [abbrev.get(wm, wm[:1]) for wm in weight_modes]

    print(f"\n{'=' * 100}\n[{dataset_label}] weight_mode比較 ({' vs '.join(weight_modes)})\n{'=' * 100}")

    header = f"{'alpha':>6} | " + " | ".join(f"{'B(' + t + ')':>8}" for t in tags) + " | " + \
             " | ".join(f"{'C(' + t + ')':>8}" for t in tags)
    print("[収束数]")
    print(header)
    all_rows = list(zip(*[results[wm]["rows"] for wm in weight_modes]))
    for row_tuple in all_rows:
        alpha = row_tuple[0]["alpha"]
        assert all(r["alpha"] == alpha for r in row_tuple)
        b_cells = " | ".join(f"{r['n_conv_b']:>5d}/{n_seeds:<2d}" for r in row_tuple)
        c_cells = " | ".join(f"{r['n_conv_c']:>5d}/{n_seeds:<2d}" for r in row_tuple)
        print(f"{alpha:>6.2f} | {b_cells} | {c_cells}")

    print("\n[手法Cの方向整列スコア(dir)・分離度(nn)]")
    header2 = f"{'alpha':>6} | " + " | ".join(f"{'dir(' + t + ')':>8}" for t in tags) + " | " + \
              " | ".join(f"{'nn(' + t + ')':>8}" for t in tags)
    print(header2)
    for row_tuple in all_rows:
        alpha = row_tuple[0]["alpha"]
        dir_cells = " | ".join(f"{r['dir_c']:>8.3f}" for r in row_tuple)
        nn_cells = " | ".join(f"{r['nn_c']:>8.3f}" for r in row_tuple)
        print(f"{alpha:>6.2f} | {dir_cells} | {nn_cells}")

    print(f"\n見方: {', '.join(f'{t}={wm}' for t, wm in zip(tags, weight_modes))}。alpha=1.0付近で"
          "収束数が低いほど、その重み方式がα=1.0付近の劣化に寄与していることの直接的な確認になる。"
          "一方、中間alpha(0.05〜0.95)の収束数がどの方式でも同程度に不安定であれば、"
          "重み設計は中間alphaの不安定性の主因ではないと言える。")

    return results


def main_beta_transform_comparison(M, dataset_label, weight_mode="uniform", n_seeds=15, gamma=0.1):
    """
    β変換(beta_transform)を混合係数に適用した場合、しない場合(通常のalpha)を
    同じサブグラフGに対して比較する。「β=0.5付近(同種・異種の重みが拮抗する領域)を
    素通りさせることで、中間alphaで観測される収束不安定性が改善するか」を検証するための実験。
    改善が見られなければ、「β≈0.5付近での拮抗」自体は不安定性の主因ではないと言える。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_beta_transform_comparison (weight_mode={weight_mode}) {'#' * 20}")

    G = get_small_subgraph_cached(M)
    result_raw = run_b_vs_c_alpha_sweep(
        G, dataset_label, weight_mode=weight_mode, n_seeds=n_seeds, gamma=gamma, alpha_transform=None
    )
    result_beta = run_b_vs_c_alpha_sweep(
        G, dataset_label, weight_mode=weight_mode, n_seeds=n_seeds, gamma=gamma, alpha_transform=beta_transform
    )

    print(f"\n{'=' * 100}\n[{dataset_label}] β変換比較 (raw alpha vs beta_transform(alpha), weight_mode={weight_mode})\n{'=' * 100}")
    print(f"{'alpha':>6} | {'beta':>6} | {'B conv(raw)':>11} | {'B conv(β)':>9} | {'C conv(raw)':>11} | {'C conv(β)':>9} | "
          f"{'C dir(raw)':>10} | {'C dir(β)':>9} | {'C nn(raw)':>9} | {'C nn(β)':>8}")
    for row_raw, row_beta in zip(result_raw["rows"], result_beta["rows"]):
        assert row_raw["alpha"] == row_beta["alpha"]
        alpha = row_raw["alpha"]
        print(f"{alpha:>6.2f} | {beta_transform(alpha):>6.2f} | "
              f"{row_raw['n_conv_b']:>8d}/{n_seeds:<2d} | {row_beta['n_conv_b']:>6d}/{n_seeds:<2d} | "
              f"{row_raw['n_conv_c']:>8d}/{n_seeds:<2d} | {row_beta['n_conv_c']:>6d}/{n_seeds:<2d} | "
              f"{row_raw['dir_c']:>10.3f} | {row_beta['dir_c']:>9.3f} | "
              f"{row_raw['nn_c']:>9.3f} | {row_beta['nn_c']:>8.3f}")

    print("\n見方: conv(raw)/conv(β)が通常のalpha/β変換適用時それぞれの収束数。"
          "β変換はalpha<0.5でβ<0.25、alpha>=0.5でβ>=0.75になるようジャンプさせ、"
          "β≈0.5付近を素通りさせる。中間alpha域の収束数がconv(β)でも改善しなければ、"
          "「β≈0.5付近での同種・異種重みの拮抗」自体は中間alphaの不安定性の主因ではない"
          "ことの直接的な確認になる。")

    return result_raw, result_beta


def compute_split_nmi_for_config(M, build_kwargs, label, threshold=DEFAULT_CONFIG.graph_build.threshold_common_deg):
    G = get_small_subgraph_cached(M, **build_kwargs)
    n_user = sum(1 for n in G.nodes() if n.startswith("u_"))
    n_movie = sum(1 for n in G.nodes() if n.startswith("m_"))
    print(f"\n{'=' * 20} [{label}] 分割NMIチェック (n_user={n_user}, n_movie={n_movie}) {'=' * 20}")

    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, "uniform", threshold, DEFAULT_CONFIG.graph_build.top_k_same_type, DEFAULT_CONFIG.graph_build.mutual_top_k_only
    )
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    nmi_all, nmi_user, nmi_movie = compute_split_nmi(nodes, is_user, common_deg, edges, edge_labels)
    return {"label": label, "n_user": n_user, "n_movie": n_movie,
            "nmi_all": nmi_all, "nmi_user": nmi_user, "nmi_movie": nmi_movie}


def main_split_nmi_size_comparison(M, dataset_label):
    """
    「userノード数が少なすぎたことが、user側の分割NMIがnanになる原因だったのか」を、
    small(既定サンプリング)とlarge(userを大幅に増やしたサンプリング)で直接比較する。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_split_nmi_size_comparison {'#' * 20}")

    result_small = compute_split_nmi_for_config(M, SMALL_BUILD_KWARGS, label=f"{dataset_label}_small")
    result_large = compute_split_nmi_for_config(M, LARGE_BUILD_KWARGS, label=f"{dataset_label}_large")

    print("\n" + "=" * 60)
    print(f"{'config':>8} | {'n_user':>6} | {'n_movie':>7} | {'NMI(all)':>9} | {'NMI(user)':>10} | {'NMI(movie)':>11}")
    for r in [result_small, result_large]:
        print(f"{r['label']:>8} | {r['n_user']:>6d} | {r['n_movie']:>7d} | "
              f"{r['nmi_all']:>9.3f} | {r['nmi_user']:>10.3f} | {r['nmi_movie']:>11.3f}")
    print("\n見方: smallでuser側NMIがnanで、largeで実数値になれば、"
          "「user側は本質的にクラスタ構造がない」のではなく「userノード数が少なすぎて"
          "Louvainがuser側を1コミュニティにまとめてしまっていた」ことが直接確認できる。")

    return result_small, result_large


def main_alpha_grid_plot(M, dataset_label):
    """
    three_experiment2.py / plot.pyの本実験: build_matrices(degree-weighted)で
    alpha_gridの可視化画像を作る。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_alpha_grid_plot {'#' * 20}")

    G = get_small_subgraph_cached(M)
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, "degree", DEFAULT_CONFIG.graph_build.threshold_common_deg, DEFAULT_CONFIG.graph_build.top_k_same_type, DEFAULT_CONFIG.graph_build.mutual_top_k_only
    )

    alphas = default_alpha_grid()  # 21点
    plot_alpha_grid(G, common_deg, weight, nodes, node_idx, is_user, alphas,
                     gamma=0.1, seed=0, filename=f"alpha_grid_{dataset_label}.png")


def run_all_experiments_for_dataset(M, dataset_label):
    """
    グラフM(MovieLensでもDBLPでも可)に対して、7つの実験一式
    (サンプリング比較・小/大規模の本実験、3手法比較+NMI、重み方式(uniform/degree)比較、
    β変換比較、規模別分割NMI比較、alpha_gridの可視化、反発・初期配置のablation)を
    全て実行する。dataset_labelは出力ファイル名・見出しの接頭辞として使われるため、
    複数データセットの結果が混ざらず、同じディレクトリに出しても後で見比べられる
    (例: alpha_layout_th0.25_movielens_small.png vs alpha_layout_th0.25_dblp_small.png)。
    """
    print(f"\n{'=' * 70}\n[{dataset_label}] 全実験を開始します\n{'=' * 70}")
    main_balance_experiment(M, dataset_label)
    main_three_method_gamma_and_nmi(M, dataset_label)
    main_weight_mode_comparison(M, dataset_label)
    main_beta_transform_comparison(M, dataset_label)
    main_split_nmi_size_comparison(M, dataset_label)
    # main_alpha_grid_plot(M, dataset_label)  # main_weight_mode_image_comparisonが
    # degree版を含む形で置き換えるため、二重に計算しないようコメントアウト
    main_weight_mode_image_comparison(M, dataset_label)
    main_repulsion_and_init_ablation(M, dataset_label)


def main_compare_movielens_and_dblp():
    """
    MovieLensとDBLPに対して全く同じ実験一式(run_all_experiments_for_dataset)を実行し、
    両者の結果を出力ファイル名(dataset_label付き)で見比べられるようにする。
    """
    M_movielens = load_movielens_graph(DEFAULT_CONFIG.paths.data_path)
    run_all_experiments_for_dataset(M_movielens, "movielens")

    M_dblp = load_dblp_graph_cached()
    run_all_experiments_for_dataset(M_dblp, "dblp")
