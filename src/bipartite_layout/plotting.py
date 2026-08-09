"""Layout plotting/saving."""

import numpy as np
import matplotlib.pyplot as plt

from bipartite_layout.caching import get_edge_direction_cached
from bipartite_layout.experiment_context import save_figure
from bipartite_layout.layout_core import compute_layout_method


def _chunk_alphas(alphas, max_cols=7):
    """
    alphasをmax_cols列ずつのチャンクに分割するジェネレータ(chunk_idx, chunk_alphas)。
    plot_alpha_grid/plot_ablation_gridで重複していたチャンク分割ロジックの共通化。
    """
    n_chunks = int(np.ceil(len(alphas) / max_cols))
    for chunk_idx in range(n_chunks):
        yield chunk_idx, alphas[chunk_idx * max_cols: (chunk_idx + 1) * max_cols]


def plot_and_save(coords, is_user, G, node_idx, title, filename, edge_labels=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[is_user, 0], coords[is_user, 1], c="black", marker="s", label="user", s=40, zorder=3)
    ax.scatter(coords[~is_user, 0], coords[~is_user, 1], c="black", marker="o", label="movie", s=40, zorder=3)

    edges = list(G.edges())
    if edge_labels is not None:
        n_clusters = int(edge_labels.max()) + 1
        cmap = plt.colormaps["tab10"].resampled(max(n_clusters, 1))
        edge_colors = [cmap(edge_labels[k]) for k in range(len(edges))]
    else:
        edge_colors = ["gray"] * len(edges)

    for k, (u, v) in enumerate(edges):
        i, j = node_idx[u], node_idx[v]
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color=edge_colors[k], alpha=0.7, linewidth=1.2, zorder=1)

    ax.set_title(title, fontsize=10)
    ax.legend()
    save_figure(fig, filename)


def plot_alpha_grid(G, common_deg, weight, nodes, node_idx, is_user,
                     alphas, gamma=0.1, seed=0, filename="alpha_grid.png",
                     methods=("B", "C"), anchor_weight=1.0, real_edge_epsilon=0.0):
    """
    行=methods(既定ではB, C。methods=("B","C","D")のようにDも含められる)、
    列=alphaの値、というグリッドでレイアウトを一覧表示する。

    real_edge_epsilon: 実エッジ項の係数を(1-alpha)ではなく(1-alpha+real_edge_epsilon)
    にする(先生からのご指摘への対応)。デフォルト0.0は既存挙動と同じ。
    """
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    n_rows = len(methods)

    for chunk_idx, chunk_alphas in _chunk_alphas(alphas):
        n_cols = len(chunk_alphas)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows))
        if n_cols == 1:
            axes = axes.reshape(n_rows, 1)
        if n_rows == 1:
            axes = axes.reshape(1, n_cols)

        for col, alpha in enumerate(chunk_alphas):
            method_results = [
                (label,) + compute_layout_method(
                    label, common_deg, weight, alpha, nodes, node_idx, is_user,
                    direction_precomputed, seed=seed, gamma=gamma, anchor_weight=anchor_weight,
                    real_edge_epsilon=real_edge_epsilon
                )
                for label in methods
            ]

            for row, (label, coords, _, conv, _, _) in enumerate(method_results):
                ax = axes[row, col]
                ax.scatter(coords[is_user, 0], coords[is_user, 1], c="black", marker="s", s=10, zorder=2)
                ax.scatter(coords[~is_user, 0], coords[~is_user, 1], c="black", marker="o", s=10, zorder=2)
                for k, (u, v) in enumerate(edges):
                    i, j = node_idx[u], node_idx[v]
                    ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                            color=plt.colormaps["tab10"].resampled(edge_labels.max() + 1)(edge_labels[k]),
                            linewidth=0.6, alpha=0.6, zorder=1)
                # 行(B/C/D...)ごとに、そのサブプロット自身の収束状況を明示する。
                conv_mark = "" if conv else " (未収束)"
                if row == 0:
                    title = f"α={alpha:.2f} [{label}]{conv_mark}"
                else:
                    title = f"[{label}]{conv_mark}"
                ax.set_title(title, fontsize=9, color=("black" if conv else "red"))
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(label, fontsize=12, rotation=0, labelpad=20)

        out_name = filename.replace(".png", f"_part{chunk_idx+1}.png")
        save_figure(fig, out_name, dpi=130,
                    message=f"保存しました: {out_name}（alpha: {[f'{a:.2f}' for a in chunk_alphas]}）")


def plot_ablation_grid(G, common_deg, weight, nodes, node_idx, is_user,
                        alphas, configs, method="B", gamma=0.1, seed=0,
                        filename="ablation_grid.png"):
    """
    plot_alpha_gridがmethod(B/C/D)間の比較用なのに対し、これはcompute_layout_methodへの
    追加キーワード引数(repel_same_type, random_initなど)違いによるconfig間比較用。
    行=configs(各要素は(label, kwargs)のタプル)、列=alphaの値。methodは1つに固定する。

    例:
      configs = [("通常(全ペア反発)", {}), ("同タイプ反発なし", {"repel_same_type": False})]
      configs = [("通常(alpha依存初期配置)", {}), ("完全ランダム初期配置", {"random_init": True})]
    """
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    n_rows = len(configs)

    for chunk_idx, chunk_alphas in _chunk_alphas(alphas):
        n_cols = len(chunk_alphas)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows))
        if n_cols == 1:
            axes = axes.reshape(n_rows, 1)
        if n_rows == 1:
            axes = axes.reshape(1, n_cols)

        for col, alpha in enumerate(chunk_alphas):
            for row, (label, kwargs) in enumerate(configs):
                coords, _, conv, _, _ = compute_layout_method(
                    method, common_deg, weight, alpha, nodes, node_idx, is_user,
                    direction_precomputed, seed=seed, gamma=gamma, **kwargs
                )
                ax = axes[row, col]
                ax.scatter(coords[is_user, 0], coords[is_user, 1], c="black", marker="s", s=10, zorder=2)
                ax.scatter(coords[~is_user, 0], coords[~is_user, 1], c="black", marker="o", s=10, zorder=2)
                for k, (u, v) in enumerate(edges):
                    i, j = node_idx[u], node_idx[v]
                    ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                            color=plt.colormaps["tab10"].resampled(edge_labels.max() + 1)(edge_labels[k]),
                            linewidth=0.6, alpha=0.6, zorder=1)
                conv_mark = "" if conv else " (未収束)"
                title = f"α={alpha:.2f}{conv_mark}" if row == 0 else conv_mark
                ax.set_title(title, fontsize=9, color=("black" if conv else "red"))
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=45, ha="right")

        out_name = filename.replace(".png", f"_part{chunk_idx+1}.png")
        save_figure(fig, out_name, dpi=130,
                    message=f"保存しました: {out_name}（alpha: {[f'{a:.2f}' for a in chunk_alphas]}）")
