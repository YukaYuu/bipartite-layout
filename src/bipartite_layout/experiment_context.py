"""Shared helpers used across experiment drivers (dedup target)."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from bipartite_layout.caching import get_edge_direction_cached, get_matrices_cached, get_small_subgraph_cached
from bipartite_layout.config import DEFAULT_CONFIG


def run_worker_tasks(worker_fn, tasks, n_workers=None):
    """
    n_workersが指定されていればProcessPoolExecutorで並列実行し、Noneなら逐次実行する、
    という分岐をまとめたヘルパー。compute_layout_multi_seed(1回のバッチ呼び出し)向け。
    run_b_vs_c_alpha_sweep/run_b_c_d_alpha_sweepはalphaループ全体で1つのexecutorを
    使い回す(効率のため)ため、あえてこのヘルパーは使わず、独自のtry/finally
    executorライフサイクルのままにしてある。
    """
    if n_workers:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            return list(executor.map(worker_fn, tasks))
    return [worker_fn(t) for t in tasks]


@dataclass
class ExperimentContext:
    G: object
    nodes: list
    node_idx: dict
    common_deg: object
    weight: object
    is_user: object
    edges: object
    edge_labels: object
    direction_precomputed: object


def build_experiment_context(G, weight_mode="degree", *, graph_build=DEFAULT_CONFIG.graph_build,
                              include_direction=True):
    """
    Gが既に手元にある関数(run_b_vs_c_alpha_sweep/run_b_c_d_alpha_sweep等、Gを引数で
    直接受け取る関数)向け。get_matrices_cached + get_edge_direction_cachedという
    2段階のキャッシュ呼び出しを1回のExperimentContext構築にまとめる。
    """
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, weight_mode, graph_build.threshold_common_deg,
        graph_build.top_k_same_type, graph_build.mutual_top_k_only,
    )
    edges = edge_labels = direction_precomputed = None
    if include_direction:
        edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)
    return ExperimentContext(G, nodes, node_idx, common_deg, weight, is_user,
                              edges, edge_labels, direction_precomputed)


def prepare_experiment_context(M, weight_mode="degree", *, build_kwargs=None,
                                graph_build=DEFAULT_CONFIG.graph_build,
                                include_direction=True):
    """
    Mからサブグラフを作るところから始める関数(main_repulsion_and_init_ablation等)向け。
    get_small_subgraph_cached → get_matrices_cached → get_edge_direction_cachedという、
    複数のmain_*/run_*関数の冒頭で繰り返されていた3段階のセットアップを1回にまとめる。
    """
    G = get_small_subgraph_cached(M, **(build_kwargs or {}))
    return build_experiment_context(G, weight_mode, graph_build=graph_build, include_direction=include_direction)


def default_alpha_grid(step=0.05):
    """
    21点(既定step=0.05でalpha=0.00〜1.00)の標準alphaグリッド。複数のalphaスイープ系
    関数(run_b_vs_c_alpha_sweep, run_b_c_d_alpha_sweep, main_repulsion_and_init_ablation,
    main_weight_mode_image_comparison, main_alpha_grid_plot)で"alphas is None"の
    フォールバックとして同一のnp.round(np.arange(0.0, 1.01, 0.05), 2)が重複していたのを
    共通化したもの。
    """
    return np.round(np.arange(0.0, 1.01, step), 2)


def save_figure(fig, path, dpi=150, message=None):
    """
    tight_layout + savefig + close + "保存しました"ログ出力をまとめたヘルパー。
    plot_and_save/plot_alpha_grid/plot_ablation_grid/各diagnosticsのpng保存で
    毎回繰り返されていた4行を1回にまとめる。messageを指定すると、既定の
    "保存しました: {path}"の代わりにその文字列をそのまま出力する
    (alpha一覧や補足説明などpathだけでは表現できない追加情報がある場合用)。
    """
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(message if message is not None else f"保存しました: {path}")
