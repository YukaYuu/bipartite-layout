"""
Koala(社内で使われている既存の二部グラフ可視化比較対象)の実体である、Andreas Noackの
LinLogエネルギーモデル(Energy Models for Graph Clustering, JGAA 11(2):453-480, 2007)の
Python移植。/Users/owner/Downloads/src/ocha/itolab/koala/core/forcedirected/MinimizerClassic.java
の数式を、このリポジトリの既存コード(layout_core.py)と同じ、numpyベクトル化+scipy L-BFGS-Bの
スタイルでそのまま再実装したもの。

koalaのexec()(GUI/アプレットから呼ばれる実体)が使っているパラメータ:
  repuExponent=0.0(対数反発), attrExponent=3.0(Fruchterman-Reingold型の「読みやすい」
  一般レイアウト。LinLogの「クラスタを計算する」ためのattrExponent=1.0とは異なる),
  gravFactor=0.0001。これを既定値として採用している。

MinimizerClassic.javaの実装は独自のline-search最小化子だが、ここではエネルギー関数
E(pos) = E_attr(pos) + E_repu(pos) + E_grav(pos) を解析的に微分し、scipyのL-BFGS-Bに
渡す形にしている(line-searchの実装が異なるだけで、最小化対象のエネルギー関数自体は
MinimizerClassic.javaの数式と同一)。
"""

import numpy as np
from scipy.optimize import minimize


def linlog_stress_and_grad(coords_flat, attr_weight, repu_exponent=0.0, attr_exponent=3.0,
                            grav_factor=0.0001, barycenter=None):
    """
    LinLogエネルギーモデルのエネルギーと勾配。attr_weightはN×Nの対称な誘引エッジ重み行列
    (0は非エッジ)。反発の強さ(node repulsion weight)は、MinimizerClassic.javaの推奨通り
    "edge-repulsion"(各ノードの誘引エッジ重みの総和)を使う。

    repuFactorの計算・エネルギー式・勾配式は、いずれもMinimizerClassic.javaの
    computeRepuFactor/getAttractionEnergy/getRepulsionEnergy/getGravitationEnergyおよび
    addAttractionDir/addRepulsionDir/addGravitationDirと数学的に同一。
    """
    N = attr_weight.shape[0]
    coords = coords_flat.reshape(N, 2)

    repu_weight = attr_weight.sum(axis=1)

    attr_sum = attr_weight.sum()
    repu_sum = repu_weight.sum()
    if repu_sum > 0 and attr_sum > 0:
        repu_factor = (attr_sum / repu_sum / repu_sum
                       * repu_sum ** (0.5 * (attr_exponent - repu_exponent)))
    else:
        repu_factor = 1.0

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist_safe = np.where(dist < 1e-12, 1e-12, dist)

    mask_off_diag = ~np.eye(N, dtype=bool)

    # --- attraction ---
    has_attr = (attr_weight > 0) & mask_off_diag
    attr_energy = np.sum(np.where(has_attr, attr_weight * dist_safe ** attr_exponent / attr_exponent, 0.0))
    attr_coef = np.where(has_attr, attr_weight * dist_safe ** (attr_exponent - 2), 0.0)
    # energyはi!=jの全順序対について足しているため(=対称行列なので各無向エッジを
    # (i,j)と(j,i)の2回数える、MinimizerClassic.javaの各ノードごとのgetEnergy呼び出しの
    # 合計と同じ規約)、その勾配も2倍になる(pos_iは(i,j)項と(j,i)=(k,i)項の両方に現れるため)。
    attr_grad = 2.0 * np.sum(attr_coef[:, :, None] * diff, axis=1)

    # --- repulsion (edge-repulsion: node weight = sum of its attraction edge weights) ---
    repu_pair = repu_weight[:, None] * repu_weight[None, :]
    has_repu = (repu_pair > 0) & mask_off_diag
    if repu_exponent == 0.0:
        repu_energy = -np.sum(np.where(has_repu, repu_factor * repu_pair * np.log(dist_safe), 0.0))
        repu_coef = np.where(has_repu, -repu_factor * repu_pair / (dist_safe ** 2), 0.0)
    else:
        repu_energy = -np.sum(np.where(
            has_repu, repu_factor * repu_pair * dist_safe ** repu_exponent / repu_exponent, 0.0))
        repu_coef = np.where(has_repu, -repu_factor * repu_pair * dist_safe ** (repu_exponent - 2), 0.0)
    repu_grad = 2.0 * np.sum(repu_coef[:, :, None] * diff, axis=1)

    # --- gravitation (barycenterは呼び出し側が現在の座標から計算し、固定値として渡す。
    # MinimizerClassic.javaもcomputeBaryCenter()を各反復の最初に1回だけ計算する
    # quasi-static な扱いをしており、それに合わせている) ---
    if barycenter is None:
        barycenter = np.zeros(2)
    to_bary = coords - barycenter[None, :]
    dist_bary = np.linalg.norm(to_bary, axis=1)
    dist_bary_safe = np.where(dist_bary < 1e-12, 1e-12, dist_bary)
    grav_energy = np.sum(grav_factor * repu_factor * repu_weight * dist_bary_safe ** attr_exponent / attr_exponent)
    grav_coef = grav_factor * repu_factor * repu_weight * dist_bary_safe ** (attr_exponent - 2)
    grav_grad = grav_coef[:, None] * to_bary

    total_energy = attr_energy + repu_energy + grav_energy
    total_grad = attr_grad + repu_grad + grav_grad
    return total_energy, total_grad.flatten()


def compute_linlog_layout(attr_weight, seed=0, maxiter=1000, repu_exponent=0.0, attr_exponent=3.0,
                           grav_factor=0.0001, x0=None):
    """
    attr_weight(N×Nの対称な誘引エッジ重み行列)からLinLogレイアウトを計算する。
    x0を渡すとその初期値からwarm startする(渡さなければランダム初期化)。
    """
    N = attr_weight.shape[0]
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-0.5, 0.5, size=N * 2)

    barycenter = x0.reshape(N, 2).mean(axis=0)

    result = minimize(
        linlog_stress_and_grad, x0,
        args=(attr_weight, repu_exponent, attr_exponent, grav_factor, barycenter),
        jac=True, method="L-BFGS-B", options={"maxiter": maxiter},
    )
    grad_norm = np.linalg.norm(result.jac)
    return result.x.reshape(N, 2), result.fun, result.success, result.nit, grad_norm


def compute_koala_alpha_layout(common_deg, weight, alpha, seed=0, maxiter=1000, x0=None,
                                repu_exponent=0.0, attr_exponent=3.0, grav_factor=0.0001,
                                real_weight_mode="constant", real_edge_epsilon=0.1):
    """
    先生の提案: 「α=0からα=1まで、常にkoala(LinLogエネルギーモデル)でレイアウトする。
    αを大きくするにつれて仮想エッジの重みを加えていく」を実装したもの。

    real_weight_mode:
      "constant": 実エッジの重みはalphaに関わらず固定(weightそのまま)。先生からの
        最初の提案「実エッジ項の計算をKoalaの計算に置き換えた上で、alphaを大きくする
        につれて仮想エッジの重みを加えていく」に対応。連続性は保たれるが、sep/nn_ratio
        がほぼ動かず、alphaを変える意味が薄れることが実験的に分かっている。
      "one_minus_alpha": 実エッジの重みも(1-alpha)でalpha依存にする。先生からの
        「alphaの値に応じて仮想エッジだけでなく実エッジの重みも変動させた方がいいか、
        という点については試してみないとわからない」という未検証の問いに対応。
        alpha=0.85〜0.97では分離が滑らかに強まるが、alpha=1.0で係数が文字通り0になり、
        stress majorization版と同じ破局的なジャンプが再現されることが実験的に分かっている。
      "epsilon_floor": one_minus_alphaと同じだが、係数を(1-alpha)ではなく
        (1-alpha)+real_edge_epsilonにし、alpha=1.0でも実エッジ項が完全には消えない
        ようにする(元のstress majorization版で有効だったreal_edge_epsilonの考え方を、
        このKoala/LinLogベースの機構にも適用したもの)。
      "none": 実エッジを一切使わない(全alphaでreal_component=0)。attr_weightは
        alpha*common_degのみになる。「実エッジを(1-alpha)のように弱めながら残す」
        のではなく「実エッジをそもそも一切使わない」場合との比較用
        (先生・共同研究者からのご指摘: この比較がまだ行われていなかった)。
        alpha=0では仮想エッジの重みも0になるため、誘引力が完全に無い(反発のみの)
        退化したレイアウトになることに注意。
    """
    if real_weight_mode == "constant":
        real_component = weight
    elif real_weight_mode == "one_minus_alpha":
        real_component = (1.0 - alpha) * weight
    elif real_weight_mode == "epsilon_floor":
        real_component = ((1.0 - alpha) + real_edge_epsilon) * weight
    elif real_weight_mode == "none":
        real_component = np.zeros_like(weight)
    else:
        raise ValueError(f"unknown real_weight_mode: {real_weight_mode!r}")

    attr_weight = real_component + alpha * common_deg
    return compute_linlog_layout(
        attr_weight, seed=seed, maxiter=maxiter, x0=x0,
        repu_exponent=repu_exponent, attr_exponent=attr_exponent, grav_factor=grav_factor,
    )
