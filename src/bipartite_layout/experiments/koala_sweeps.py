"""
先生の提案(「α=0からα=1まで常にkoala(LinLogエネルギーモデル)でレイアウトし、
αを大きくするにつれて仮想エッジの重みを加えていく」)を検証する実験一式。

3つの画像を出力する:
  koala_constant.png         実エッジの重みをalphaに関わらず固定した場合
                              (連続性は保たれるが、分離がほぼ進まない)
  koala_shrinking.png        実エッジの重みを(1-alpha)にした場合
                              (分離は滑らかに進むが、alpha=1.0で係数が文字通り0になり、
                               stress majorization版と同じ破局的なジャンプが再現される)
  koala_epsilon_floor.png    実エッジの重みを(1-alpha)+epsilonにし、attrExponent=1.0
                              (真のLinLogクラスタリング指数)にした場合。連続性・分離・
                              方向整列(束ね)のいずれも良好で、現時点でのベスト設定。

各画像のタイトルに、隣接alpha間の形状変化(shape_distance)・分離度(nn_ratio)・
方向整列スコア(align、明示的なgamma項を使わずに計算)を表示する。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from bipartite_layout.direction import calc_direction_alignment_score
from bipartite_layout.experiment_context import save_figure
from bipartite_layout.linlog import compute_koala_alpha_layout
from bipartite_layout.metrics import compute_separation_metrics


def _shape_distance(coords_a, coords_b):
    da, db = pdist(coords_a), pdist(coords_b)
    return float(np.linalg.norm(da - db) / (np.linalg.norm(da) + 1e-12))


def _draw_koala_row(axes_row, ctx, alphas, real_weight_mode, real_edge_epsilon=0.1,
                     attr_exponent=3.0, seed=0, maxiter=2000, row_label=None):
    """axes_row(1行分のAxes配列)に、alphaを動かしながらkoalaレイアウトを描画する。
    複数の設定を縦に並べて比較する図(plot_koala_real_edge_comparisonなど)と、
    1設定だけの図(_plot_koala_sweep)の両方から共通して使われる。"""
    edges = list(ctx.G.edges())
    prev_coords = None
    for ax, a in zip(axes_row, alphas):
        x0 = prev_coords.flatten() if prev_coords is not None else None
        coords, _, converged, _, _ = compute_koala_alpha_layout(
            ctx.common_deg, ctx.weight, a, seed=seed, maxiter=maxiter, x0=x0,
            real_weight_mode=real_weight_mode, real_edge_epsilon=real_edge_epsilon,
            attr_exponent=attr_exponent,
        )
        shape_dist = _shape_distance(coords, prev_coords) if prev_coords is not None else None
        prev_coords = coords

        _, nn_ratio = compute_separation_metrics(coords, ctx.is_user)
        align = calc_direction_alignment_score(coords, ctx.nodes, ctx.node_idx, ctx.direction_precomputed)

        for (u, v) in edges:
            i, j = ctx.node_idx[u], ctx.node_idx[v]
            ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                    color="gray", alpha=0.4, linewidth=0.6, zorder=1)
        ax.scatter(coords[ctx.is_user, 0], coords[ctx.is_user, 1], c="black", marker="s", s=25, zorder=2, label="user")
        ax.scatter(coords[~ctx.is_user, 0], coords[~ctx.is_user, 1], c="tab:red", marker="o", s=25, zorder=2, label="movie")

        shape_dist_str = f"{shape_dist:.3f}" if shape_dist is not None else "--"
        ax.set_title(f"alpha={a:.2f}{'' if converged else ' (未収束)'}\n"
                     f"shape_dist={shape_dist_str}  nn_ratio={nn_ratio:.1f}  align={align:.3f}",
                     fontsize=9, color=("black" if converged else "red"))
        ax.set_xticks([]); ax.set_yticks([])
        if a == alphas[0]:
            if row_label:
                ax.set_ylabel(row_label, fontsize=10)
            ax.legend(fontsize=8)


def _plot_koala_sweep(ctx, alphas, filename, title_suffix, real_weight_mode, real_edge_epsilon=0.1,
                       attr_exponent=3.0, seed=0, maxiter=2000):
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.0 * len(alphas), 4.4))
    fig.suptitle(f"koala alpha sweep -- {title_suffix}", fontsize=11)
    _draw_koala_row(axes, ctx, alphas, real_weight_mode, real_edge_epsilon, attr_exponent, seed, maxiter)
    save_figure(fig, filename, dpi=130)


def plot_koala_real_edge_comparison(ctx, alphas, filename="koala_real_edge_comparison.png",
                                     real_edge_epsilon=0.1, attr_exponent=1.0, seed=0, maxiter=2000):
    """
    先生・共同研究者からのご指摘: 「実エッジの重みを仮想エッジと逆(どんどん小さくなって
    いく)の働きで追加した場合」(=epsilon_floor)と、「実エッジの重みは追加しない場合」
    (=real_weight_mode="none", 全alphaでreal_component=0)を、同じalpha刻みで
    直接比較する図。上段がepsilon_floor、下段がnone。
    """
    fig, axes = plt.subplots(2, len(alphas), figsize=(4.0 * len(alphas), 8.8))
    fig.suptitle(
        "koala: real edge weight present (shrinking, top) vs absent entirely (bottom)", fontsize=11
    )
    _draw_koala_row(axes[0], ctx, alphas, "epsilon_floor", real_edge_epsilon, attr_exponent, seed, maxiter,
                     row_label="real edge:\n(1-a)+eps")
    _draw_koala_row(axes[1], ctx, alphas, "none", real_edge_epsilon, attr_exponent, seed, maxiter,
                     row_label="real edge:\nnone")
    save_figure(fig, filename, dpi=130)


def plot_koala_epsilon_floor(ctx, alphas, filename="koala_epsilon_floor_fine.png", real_edge_epsilon=0.1,
                              seed=0, maxiter=2000):
    """koala_epsilon_floor.pngと同じ設定(attrExp=1.0の真のLinLogクラスタリング指数)を、
    好きなalpha刻みで生成する(0.1刻みで細かく見たい場合など)。"""
    _plot_koala_sweep(
        ctx, alphas, filename,
        "real weight = (1-alpha)+eps, attrExp=1.0 (true LinLog clustering)",
        real_weight_mode="epsilon_floor", real_edge_epsilon=real_edge_epsilon, attr_exponent=1.0,
        seed=seed, maxiter=maxiter,
    )


def run_koala_comparison(ctx, alphas=None, seed=0, maxiter=2000):
    """3つの設定を比較する画像を生成する(カレントディレクトリに保存)。"""
    if alphas is None:
        alphas = [0.0, 0.5, 0.85, 0.94, 1.0]

    _plot_koala_sweep(
        ctx, alphas, "koala_constant.png",
        "real weight constant (attrExp=3.0, Fruchterman-Reingold-like)",
        real_weight_mode="constant", attr_exponent=3.0, seed=seed, maxiter=maxiter,
    )
    _plot_koala_sweep(
        ctx, alphas, "koala_shrinking.png",
        "real weight = (1-alpha) (attrExp=3.0) -- reproduces the alpha=1 collapse",
        real_weight_mode="one_minus_alpha", attr_exponent=3.0, seed=seed, maxiter=maxiter,
    )
    _plot_koala_sweep(
        ctx, alphas, "koala_epsilon_floor.png",
        "real weight = (1-alpha)+eps, attrExp=1.0 (true LinLog clustering) -- best so far",
        real_weight_mode="epsilon_floor", real_edge_epsilon=0.1, attr_exponent=1.0,
        seed=seed, maxiter=maxiter,
    )
