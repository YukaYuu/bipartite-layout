"""Core stress-majorization layout optimization (methods A/B/C/D)."""

import numpy as np
from scipy.optimize import minimize

from bipartite_layout.direction import direction_alignment_stress_and_grad

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def make_initial_positions(nodes, is_user, alpha, seed=0):
    rng = np.random.default_rng(seed)
    N = len(nodes)
    coords = np.zeros((N, 2))
    for i in range(N):
        if is_user[i]:
            x_min, x_max = 0.0, 1.0 - alpha
        else:
            x_min, x_max = alpha, 1.0
        if x_max - x_min < 1e-6:
            x_min, x_max = min(x_min, 1.0 - 1e-3), max(x_max, x_min + 1e-3)
        coords[i, 0] = rng.uniform(x_min, x_max)
        coords[i, 1] = rng.uniform(0.0, 1.0)
    return coords.flatten()


def make_initial_positions_random(nodes, seed=0):
    """
    make_initial_positionsと異なり、alphaに一切依存しない完全ランダム初期化
    (user/movieどちらもx,y共に[0,1]一様乱数)。二層配置がストレス関数のalpha混合
    項だけから自然に立ち上がるのか、それとも初期配置のalpha依存的な列分割に
    引きずられているだけなのかを切り分けるための比較実験用。
    """
    rng = np.random.default_rng(seed)
    N = len(nodes)
    coords = rng.uniform(0.0, 1.0, size=(N, 2))
    return coords.flatten()


def _stress_and_grad_numpy(coords_flat, common_deg, weight, alpha, cutoff, strength, is_user, repel_same_type):
    """stress_and_gradのnumpyベクトル化実装。numbaが使えない環境向けのフォールバック。"""
    N = common_deg.shape[0]
    coords = coords_flat.reshape(N, 2)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) + 1e-9

    grad = np.zeros_like(coords)
    total_stress = 0.0

    def accumulate(mask, target):
        nonlocal total_stress, grad
        err = (dist - target) * mask
        total_stress_local = np.sum(err ** 2)
        coef = (2 * err / dist)[:, :, None]
        grad_local = np.sum(coef * diff, axis=1)
        return total_stress_local, grad_local

    same_type_target = 1 - common_deg
    s_a, g_a = accumulate(common_deg > 0, same_type_target)
    total_stress += alpha * s_a
    grad += alpha * g_a

    cross_type_target = 1 - weight
    s_b, g_b = accumulate(weight > 0, cross_type_target)
    total_stress += (1 - alpha) * s_b
    grad += (1 - alpha) * g_b

    # 反発力は全ペアに適用する(has_attractionによる除外は撤回済み)。
    # ただしrepel_same_type=Falseの場合は、同タイプ同士(user-user/movie-movie)の
    # 反発だけを切り、異タイプ間の反発は残す(格子状配置が全ペア反発由来かの検証用)。
    mask_all = np.ones((N, N)) - np.eye(N)
    if not repel_same_type:
        same_type_pair = is_user[:, None] == is_user[None, :]
        mask_all = mask_all * (~same_type_pair)
    within_cutoff = (dist < cutoff) & (mask_all > 0)

    inv_diff = np.where(within_cutoff, (1.0 / dist) - (1.0 / cutoff), 0.0)
    rep_energy = strength * np.sum(inv_diff ** 2) / 2
    total_stress += rep_energy

    d_energy_d_dist = np.where(
        within_cutoff,
        strength * inv_diff * (-1.0 / (dist ** 2)),
        0.0
    )
    coef_rep = (d_energy_d_dist / dist)[:, :, None]
    grad += np.sum(coef_rep * diff, axis=1)

    return total_stress, grad.flatten()


def _stress_and_grad_loop(coords_flat, common_deg, weight, alpha, cutoff, strength, is_user, repel_same_type):
    """
    stress_and_gradの明示ループ実装。_stress_and_grad_numpyと数値的に等価
    (浮動小数点の加算順序の違いによる1e-8程度の誤差はある)だが、numbaで
    JITコンパイルすると、O(N^2)の中間配列(diff/dist/err/coef等)を毎回
    確保するnumpyベクトル化版より大幅に速い(N=12〜165で実測15〜39倍)。
    numbaの@njitがnonlocalクロージャや3次元配列のブロードキャストを
    サポートしないため、あえてこの形にしている。

    repel_same_type=Falseの場合、同タイプ同士(user-user/movie-movie)の反発だけを
    切る(異タイプ間の反発は残す)。格子状配置が全ペア反発由来かどうかの検証用。
    """
    N = common_deg.shape[0]
    coords = coords_flat.reshape(N, 2)
    grad = np.zeros((N, 2))
    total_stress = 0.0

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-9

            cd = common_deg[i, j]
            if cd > 0:
                target = 1.0 - cd
                err = dist - target
                total_stress += alpha * err * err
                coef = alpha * 2.0 * err / dist
                grad[i, 0] += coef * dx
                grad[i, 1] += coef * dy

            w = weight[i, j]
            if w > 0:
                target = 1.0 - w
                err = dist - target
                total_stress += (1.0 - alpha) * err * err
                coef = (1.0 - alpha) * 2.0 * err / dist
                grad[i, 0] += coef * dx
                grad[i, 1] += coef * dy

            # 反発力は全ペアに適用する(has_attractionによる除外は撤回済み)。
            # ただしrepel_same_type=Falseなら同タイプ間の反発だけスキップする。
            if dist < cutoff and (repel_same_type or is_user[i] != is_user[j]):
                inv_diff = (1.0 / dist) - (1.0 / cutoff)
                total_stress += strength * inv_diff * inv_diff / 2.0
                d_energy_d_dist = strength * inv_diff * (-1.0 / (dist * dist))
                coef_rep = d_energy_d_dist / dist
                grad[i, 0] += coef_rep * dx
                grad[i, 1] += coef_rep * dy

    return total_stress, grad.flatten()


if HAS_NUMBA:
    # numbaが使える場合はJITコンパイルした明示ループ版を使う(大幅に高速)。
    stress_and_grad = numba.njit(cache=True, fastmath=True)(_stress_and_grad_loop)
else:
    # numbaが無い環境ではnumpyベクトル化版にフォールバックする
    # (退化はせず、元のcombined_experiment.pyと同じ速度のまま動く)。
    stress_and_grad = _stress_and_grad_numpy


def combined_stress_and_grad(coords_flat, common_deg, weight, alpha, nodes, node_idx,
                              direction_precomputed, gamma, cutoff, strength, is_user, repel_same_type=True):
    stress_mix, grad_mix = stress_and_grad(coords_flat, common_deg, weight, alpha, cutoff, strength, is_user, repel_same_type)
    if gamma > 0:
        stress_dir, grad_dir = direction_alignment_stress_and_grad(coords_flat, nodes, node_idx, direction_precomputed)
        return stress_mix + gamma * stress_dir, grad_mix + gamma * grad_dir
    return stress_mix, grad_mix


def _anchor_stress_and_grad(coords_flat, anchor_flat, anchor_weight):
    """
    現在の座標をanchor_flat(固定した参照レイアウト)に弱く引き戻す二次ペナルティ。
    method D(逐次的束ね)のstage2で、alpha混合の同時最適化をやり直すのではなく、
    「レイアウトをおおよそ固定したまま、方向整列のためだけに動かす」ために使う。
    """
    diff = coords_flat - anchor_flat
    stress = anchor_weight * np.sum(diff ** 2)
    grad = anchor_weight * 2.0 * diff
    return stress, grad


def _postprocess_bundle_stress_and_grad(coords_flat, anchor_flat, anchor_weight, nodes, node_idx,
                                         direction_precomputed, gamma):
    """method Dのstage2の目的関数: アンカー項 + 方向整列(alpha混合stressは含まない)。"""
    s_anchor, g_anchor = _anchor_stress_and_grad(coords_flat, anchor_flat, anchor_weight)
    if gamma > 0:
        s_dir, g_dir = direction_alignment_stress_and_grad(coords_flat, nodes, node_idx, direction_precomputed)
        return s_anchor + gamma * s_dir, g_anchor + gamma * g_dir
    return s_anchor, g_anchor


def compute_layout_method(method, common_deg, weight, alpha, nodes, node_idx, is_user,
                           direction_precomputed, seed=0, base_cutoff=0.3, base_n=32,
                           strength=0.3, gamma=1.0, maxiter=500, anchor_weight=1.0,
                           repel_same_type=True, random_init=False):
    """
    method="A": 方向整列のみ(alpha混合なし)
    method="B": alpha混合のみ(gamma=0固定、方向整列なし)
    method="C": alpha混合 + 方向整列(同時最適化)
    method="D": 逐次的束ね(sequential bundling)。まずmethod Bでalpha混合stressのみを
                収束させ("stage1")、そのレイアウトをanchor_weightで弱く固定しつつ、
                方向整列のみを追加で最適化する("stage2")。method Cが同時最適化で
                不安定になるコストを払ってでも価値があるのか、それとも逐次的に
                束ねるだけで同程度の効果が得られ、より安定するのかを比較するための、
                Claudeとの相談から追加した第4の手法。

    method="B", gamma=0.0, direction_precomputed=Noneで呼べば、alpha混合stressのみを
    最適化する単独実験(compute_layout_multi_seedが使う経路)としても使える
    (gamma=0.0のときはcombined_stress_and_grad内でdirection_precomputedに
    一切アクセスしないため、Noneを渡しても安全)。

    cutoffのみをNでスケールする。strengthは常に固定値のまま。
    """
    N = len(nodes)
    cutoff = base_cutoff * np.sqrt(base_n / N)

    if method == "D":
        # stage1: method Bでalpha混合stressのみを収束させる
        coords_b, _, converged_b, _, _ = compute_layout_method(
            "B", common_deg, weight, alpha, nodes, node_idx, is_user, None,
            seed=seed, base_cutoff=base_cutoff, base_n=base_n, strength=strength,
            gamma=0.0, maxiter=maxiter, repel_same_type=repel_same_type, random_init=random_init
        )
        anchor_flat = coords_b.flatten()
        # stage2: そのレイアウトをアンカーしつつ、方向整列のみを追加で最適化する
        result = minimize(
            _postprocess_bundle_stress_and_grad, anchor_flat,
            args=(anchor_flat, anchor_weight, nodes, node_idx, direction_precomputed, gamma),
            jac=True, method="L-BFGS-B", options={"maxiter": maxiter}
        )
        converged = converged_b and result.success
        grad_norm = np.linalg.norm(result.jac)
        return result.x.reshape(N, 2), result.fun, converged, result.nit, grad_norm

    if random_init:
        x0 = make_initial_positions_random(nodes, seed=seed)
    else:
        x0 = make_initial_positions(nodes, is_user, alpha if method != "A" else 0.5, seed=seed)

    if method == "A":
        dummy_common_deg = np.zeros_like(weight)
        result = minimize(combined_stress_and_grad, x0,
                           args=(dummy_common_deg, weight, 0.0, nodes, node_idx, direction_precomputed, gamma, cutoff, strength, is_user, repel_same_type),
                           jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    else:
        g = 0.0 if method == "B" else gamma
        result = minimize(combined_stress_and_grad, x0,
                           args=(common_deg, weight, alpha, nodes, node_idx, direction_precomputed, g, cutoff, strength, is_user, repel_same_type),
                           jac=True, method="L-BFGS-B", options={"maxiter": maxiter})

    grad_norm = np.linalg.norm(result.jac)
    return result.x.reshape(N, 2), result.fun, result.success, result.nit, grad_norm
