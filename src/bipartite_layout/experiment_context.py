"""Shared helpers used across experiment drivers (dedup target)."""

import matplotlib.pyplot as plt


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
