import numpy as np

from bipartite_layout import layout_core
from bipartite_layout.caching import get_edge_direction_cached
from bipartite_layout.layout_core import compute_layout_method, make_initial_positions_random


def test_numba_and_numpy_stress_and_grad_agree(matrices):
    """
    layout_core.HAS_NUMBAがTrueの環境では、実際に使われるstress_and_grad(numba版)と
    フォールバック用の_stress_and_grad_numpyが数値的に一致することを確認する。
    片方だけ壊れて片方だけ通るような回帰を検出するための、この2実装の間の唯一の
    自動テスト。
    """
    nodes, node_idx, common_deg, weight, is_user = matrices
    N = len(nodes)
    rng = np.random.default_rng(0)
    coords_flat = rng.uniform(0, 1, size=N * 2)
    cutoff, strength = 0.3, 0.3

    for repel_same_type in (True, False):
        for real_edge_epsilon in (0.0, 0.05):
            stress_loop, grad_loop = layout_core._stress_and_grad_loop(
                coords_flat, common_deg, weight, 0.5, cutoff, strength, is_user, repel_same_type,
                real_edge_epsilon
            )
            stress_np, grad_np = layout_core._stress_and_grad_numpy(
                coords_flat, common_deg, weight, 0.5, cutoff, strength, is_user, repel_same_type,
                real_edge_epsilon
            )
            assert np.isclose(stress_loop, stress_np, rtol=1e-6)
            assert np.allclose(grad_loop, grad_np, atol=1e-8)


def test_real_edge_epsilon_keeps_real_edge_term_alive_at_alpha_one(matrices):
    """
    先生からのご指摘への対応: 実エッジ項の係数を(1-alpha)ではなく
    (1-alpha+real_edge_epsilon)にすることで、alpha=1.0でも実エッジ制約が
    完全には消えないようにする。real_edge_epsilon=0.0(既定)では従来通り
    alpha=1.0で実エッジ項の係数が文字通り0になるが、epsilon>0ならそうならない
    ことを、勾配ノルムで直接確認する。
    """
    nodes, node_idx, common_deg, weight, is_user = matrices
    N = len(nodes)
    rng = np.random.default_rng(0)
    coords_flat = rng.uniform(0, 1, size=N * 2)
    cutoff, strength = 0.3, 0.3

    # epsilon=0.0(既定)では、alpha=1.0で実エッジ係数(1-alpha+0)が文字通り0になり、
    # weight>0の重みだけを0にした行列と勾配が完全に一致するはず。
    zero_weight = np.zeros_like(weight)
    _, grad_no_real_edges = layout_core.stress_and_grad(
        coords_flat, common_deg, zero_weight, 1.0, cutoff, strength, is_user, True, 0.0
    )
    _, grad_epsilon_zero = layout_core.stress_and_grad(
        coords_flat, common_deg, weight, 1.0, cutoff, strength, is_user, True, 0.0
    )
    assert np.allclose(grad_no_real_edges, grad_epsilon_zero, atol=1e-8), (
        "real_edge_epsilon=0.0の場合、alpha=1.0では実エッジ項が勾配に一切寄与しないはず"
    )

    # epsilon>0では、alpha=1.0でも実エッジ項が勾配に寄与し続けるはず
    # (real_edge_epsilon無しの場合と異なる勾配になる)。
    _, grad_epsilon_positive = layout_core.stress_and_grad(
        coords_flat, common_deg, weight, 1.0, cutoff, strength, is_user, True, 0.05
    )
    assert not np.allclose(grad_no_real_edges, grad_epsilon_positive, atol=1e-6), (
        "real_edge_epsilon>0の場合、alpha=1.0でも実エッジ項が勾配に寄与し続けるはず"
    )


def test_repel_same_type_changes_layout(matrices):
    """combined_experiment.pyから引き継いだablationの核: repel_same_type=Falseは
    実際にレイアウトを変える(黙って無視されていないか)ことをロックする回帰テスト。"""
    nodes, node_idx, common_deg, weight, is_user = matrices
    kwargs = dict(seed=0, gamma=0.0, maxiter=80)
    coords_default, *_ = compute_layout_method("B", common_deg, weight, 0.5, nodes, node_idx, is_user, None, **kwargs)
    coords_norepel, *_ = compute_layout_method(
        "B", common_deg, weight, 0.5, nodes, node_idx, is_user, None, repel_same_type=False, **kwargs
    )
    assert np.linalg.norm(coords_default - coords_norepel) > 1e-3


def test_random_init_changes_layout(matrices):
    """同様に、random_init=Trueも実際に効果を持つことをロックする。"""
    nodes, node_idx, common_deg, weight, is_user = matrices
    kwargs = dict(seed=0, gamma=0.0, maxiter=80)
    coords_default, *_ = compute_layout_method("B", common_deg, weight, 0.99, nodes, node_idx, is_user, None, **kwargs)
    coords_random, *_ = compute_layout_method(
        "B", common_deg, weight, 0.99, nodes, node_idx, is_user, None, random_init=True, **kwargs
    )
    assert np.linalg.norm(coords_default - coords_random) > 1e-6


def test_make_initial_positions_random_ignores_alpha_but_is_seed_deterministic(matrices):
    nodes, *_ = matrices
    x0_a = make_initial_positions_random(nodes, seed=7)
    x0_b = make_initial_positions_random(nodes, seed=7)
    assert np.allclose(x0_a, x0_b)


def test_method_d_anchor_weight_controls_distance_from_stage1(small_graph, matrices):
    """
    method D(逐次的束ね)はstage1(method Bでalpha混合stressのみ収束)の後、
    stage2でそのレイアウトをanchor_weightで固定しつつ方向整列を追加最適化する。
    anchor_weightを大きくするほどstage2の移動量(stage1からの距離)は小さくなる
    はずで、これがMethod Dの核となる設計(アンカーの強さで方向整列とのトレードオフを
    制御する)が実際に機能していることの直接的なロック。
    """
    nodes, node_idx, common_deg, weight, is_user = matrices
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(small_graph, node_idx)

    coords_b, _, conv_b, _, _ = compute_layout_method(
        "B", common_deg, weight, 0.5, nodes, node_idx, is_user, None, seed=0, gamma=0.0, maxiter=2000
    )
    assert conv_b, "this test assumes method B converges for the fixture graph/alpha"

    coords_d_weak, *_ = compute_layout_method(
        "D", common_deg, weight, 0.5, nodes, node_idx, is_user, direction_precomputed,
        seed=0, gamma=0.5, maxiter=2000, anchor_weight=0.01
    )
    coords_d_strong, *_ = compute_layout_method(
        "D", common_deg, weight, 0.5, nodes, node_idx, is_user, direction_precomputed,
        seed=0, gamma=0.5, maxiter=2000, anchor_weight=1e4
    )

    dist_weak = np.linalg.norm(coords_b - coords_d_weak)
    dist_strong = np.linalg.norm(coords_b - coords_d_strong)
    assert dist_strong < dist_weak
    assert dist_strong < 1e-2
