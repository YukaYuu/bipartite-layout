"""Shared helpers used across experiment drivers (dedup target)."""

import matplotlib.pyplot as plt
import numpy as np


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
