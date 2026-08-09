"""Same-type (virtual) and cross-type (real) edge weight matrix builders."""

import numpy as np

from bipartite_layout.sampling import apply_top_k_sparsification


def build_matrices(G, threshold_common_deg=None, top_k_same_type=None, mutual_top_k_only=False):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    is_user = np.array([n.startswith("u_") for n in nodes])

    common_deg = np.zeros((N, N))
    weight = np.zeros((N, N))

    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i >= j:
                continue
            same_type = ni[0] == nj[0]
            if same_type:
                ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
                union = ni_set | nj_set
                if union:
                    sim = len(ni_set & nj_set) / len(union)
                    common_deg[i, j] = common_deg[j, i] = sim
            else:
                if G.has_edge(ni, nj):
                    w = 1.0 / np.sqrt(G.degree(ni) * G.degree(nj))
                    weight[i, j] = weight[j, i] = w

    # 異種ペアの重みを0.4〜1.0の範囲に正規化
    nonzero_mask = weight > 0
    if nonzero_mask.any():
        nz = weight[nonzero_mask]
        normalized = (nz - nz.min()) / (nz.max() - nz.min() + 1e-9)
        weight[nonzero_mask] = 0.4 + 0.6 * normalized

    n_before = int(np.sum(common_deg > 0))

    if top_k_same_type is not None:
        common_deg = apply_top_k_sparsification(
            common_deg, is_user, top_k_same_type, mutual_only=mutual_top_k_only
        )
        n_after_knn = int(np.sum(common_deg > 0))
        print(f"top-k絞り込み(k={top_k_same_type}, mutual_only={mutual_top_k_only}) "
              f"適用: 仮想エッジ数 {n_before} → {n_after_knn}")

    if threshold_common_deg is not None:
        n_before_th = int(np.sum(common_deg > 0))
        common_deg[common_deg < threshold_common_deg] = 0.0
        n_after_th = int(np.sum(common_deg > 0))
        print(f"閾値 {threshold_common_deg} 適用: 仮想エッジ数 {n_before_th} → {n_after_th}")

    # 仮想エッジ(common_deg)をweightと同じ[0.4, 1.0]に正規化する(先生からのご指摘への
    # 対応: 実エッジはweightで正規化しているのに、仮想エッジには対応する正規化が無い
    # 非対称性があった)。top-k絞り込み・閾値フィルタより後に行うことで、
    # threshold_common_degが生のJaccard類似度のスケールで機能する意味を保つ。
    nonzero_common = common_deg > 0
    if nonzero_common.any():
        nz_common = common_deg[nonzero_common]
        normalized_common = (nz_common - nz_common.min()) / (nz_common.max() - nz_common.min() + 1e-9)
        common_deg[nonzero_common] = 0.4 + 0.6 * normalized_common

    n_user_edges = int(np.sum(common_deg[np.ix_(is_user, is_user)] > 0) / 2)
    n_movie_edges = int(np.sum(common_deg[np.ix_(~is_user, ~is_user)] > 0) / 2)
    n_user_nodes = int(is_user.sum())
    n_movie_nodes = int((~is_user).sum())
    print(f"  絞り込み後の内訳: user-user {n_user_edges}本 (node数{n_user_nodes}), "
          f"movie-movie {n_movie_edges}本 (node数{n_movie_nodes})")
    if n_user_nodes > 1:
        print(f"  user側 平均次数(仮想エッジ): {2 * n_user_edges / n_user_nodes:.2f}")
    if n_movie_nodes > 1:
        print(f"  movie側 平均次数(仮想エッジ): {2 * n_movie_edges / n_movie_nodes:.2f}")

    return nodes, idx, common_deg, weight, is_user


def build_matrices_uniform_weight(G, threshold_common_deg=None, top_k_same_type=None,
                                    mutual_top_k_only=False, uniform_weight_value=1.0):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    is_user = np.array([n.startswith("u_") for n in nodes])
    common_deg = np.zeros((N, N))
    weight = np.zeros((N, N))
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i >= j:
                continue
            if ni[0] == nj[0]:
                ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
                union = ni_set | nj_set
                if union:
                    sim = len(ni_set & nj_set) / len(union)
                    common_deg[i, j] = common_deg[j, i] = sim
            else:
                if G.has_edge(ni, nj):
                    # 次数に基づく計算(1/√(deg・deg))をやめ、全て同じ値にする
                    weight[i, j] = weight[j, i] = uniform_weight_value
    if top_k_same_type is not None:
        common_deg = apply_top_k_sparsification(common_deg, is_user, top_k_same_type, mutual_top_k_only)
    if threshold_common_deg is not None:
        common_deg[common_deg < threshold_common_deg] = 0.0
    nonzero_common = common_deg > 0
    if nonzero_common.any():
        nz_common = common_deg[nonzero_common]
        normalized_common = (nz_common - nz_common.min()) / (nz_common.max() - nz_common.min() + 1e-9)
        common_deg[nonzero_common] = 0.4 + 0.6 * normalized_common
    return nodes, idx, common_deg, weight, is_user


def build_matrices_commonality_weight(G, threshold_common_deg=None, top_k_same_type=None,
                                       mutual_top_k_only=False):
    """
    build_matricesの改造版: 実エッジ(異種ペア u-m)の重みを、次数ではなく
    「隣接ノードの共通度」から設計する(先生からのご指摘への対応: 均一な重みは
    実エッジに対する意味のあるばねの自然長として機能していなかったため、
    「共通隣接数をバネの自然長として設計し直す」)。

    エッジ(u, m)の重みは以下の2つの平均とする:
      - movie側スコア: uの「他の」movie隣接群それぞれとmとの同種Jaccard類似度
        (common_deg)の平均。(mが、uが元々好んでいた他の映画とどれだけ似ているか)
      - user側スコア: mの「他の」user隣接群それぞれとuとの同種Jaccard類似度の平均。
        (uが、mを好んでいる他のuserとどれだけ似ているか)
    片方に「他の」隣接ノードが無ければもう片方のみ、両方無ければ0とする
    (0だったペアは、後段の正規化で最低値0.4に切り上げる)。

    重要: ここで使うcommon_degは、top_k_same_type/threshold_common_degによる
    絞り込み(sparsify)前の生のJaccard類似度でなければならない。絞り込み後の
    common_degを使うと、大半のペアが単に「上位k位に入らなかった」だけで0になり、
    実際には似ているペアまで「共通性なし」として扱われてしまい、重みが不当に
    低く出てしまう。
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    is_user = np.array([n.startswith("u_") for n in nodes])

    common_deg = np.zeros((N, N))
    real_edges = []

    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i >= j:
                continue
            if ni[0] == nj[0]:
                ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
                union = ni_set | nj_set
                if union:
                    sim = len(ni_set & nj_set) / len(union)
                    common_deg[i, j] = common_deg[j, i] = sim
            else:
                if G.has_edge(ni, nj):
                    real_edges.append((i, j))

    # --- 実エッジの重みを、sparsify前のcommon_degから計算する共通隣接度で決める ---
    weight = np.zeros((N, N))
    for i, j in real_edges:
        ni, nj = nodes[i], nodes[j]
        u_node, m_node = (ni, nj) if ni.startswith("u_") else (nj, ni)
        u_i, m_i = idx[u_node], idx[m_node]

        other_movies = [mm for mm in G.neighbors(u_node) if mm != m_node]
        other_users = [uu for uu in G.neighbors(m_node) if uu != u_node]

        movie_side = np.mean([common_deg[m_i, idx[mm]] for mm in other_movies]) if other_movies else None
        user_side = np.mean([common_deg[u_i, idx[uu]] for uu in other_users]) if other_users else None

        if movie_side is not None and user_side is not None:
            score = (movie_side + user_side) / 2
        elif movie_side is not None:
            score = movie_side
        elif user_side is not None:
            score = user_side
        else:
            score = 0.0

        weight[u_i, m_i] = weight[m_i, u_i] = score

    # 異種ペアの重みを0.4〜1.0の範囲に正規化(他の重み方式と同じ正規化ルール)。
    # スコアが0(両側とも他に隣接ノードが無かった実エッジ)は正規化の母数から外れるため、
    # 別途最低値0.4を明示的に割り当てる(「共通隣接度に関する情報が無い」ことを
    # 「最も共通度が低い」ことと同じ扱いにする)。
    nonzero_mask = weight > 0
    if nonzero_mask.any():
        nz = weight[nonzero_mask]
        normalized = (nz - nz.min()) / (nz.max() - nz.min() + 1e-9)
        weight[nonzero_mask] = 0.4 + 0.6 * normalized
    for i, j in real_edges:
        if weight[i, j] == 0:
            weight[i, j] = weight[j, i] = 0.4

    n_before = int(np.sum(common_deg > 0))

    if top_k_same_type is not None:
        common_deg = apply_top_k_sparsification(
            common_deg, is_user, top_k_same_type, mutual_only=mutual_top_k_only
        )
        n_after_knn = int(np.sum(common_deg > 0))
        print(f"top-k絞り込み(k={top_k_same_type}, mutual_only={mutual_top_k_only}) "
              f"適用: 仮想エッジ数 {n_before} → {n_after_knn}")

    if threshold_common_deg is not None:
        n_before_th = int(np.sum(common_deg > 0))
        common_deg[common_deg < threshold_common_deg] = 0.0
        n_after_th = int(np.sum(common_deg > 0))
        print(f"閾値 {threshold_common_deg} 適用: 仮想エッジ数 {n_before_th} → {n_after_th}")

    # 仮想エッジをweightと同じ[0.4, 1.0]に正規化する(build_matricesと同じ対応)。
    nonzero_common = common_deg > 0
    if nonzero_common.any():
        nz_common = common_deg[nonzero_common]
        normalized_common = (nz_common - nz_common.min()) / (nz_common.max() - nz_common.min() + 1e-9)
        common_deg[nonzero_common] = 0.4 + 0.6 * normalized_common

    n_user_edges = int(np.sum(common_deg[np.ix_(is_user, is_user)] > 0) / 2)
    n_movie_edges = int(np.sum(common_deg[np.ix_(~is_user, ~is_user)] > 0) / 2)
    n_user_nodes = int(is_user.sum())
    n_movie_nodes = int((~is_user).sum())
    print(f"  絞り込み後の内訳: user-user {n_user_edges}本 (node数{n_user_nodes}), "
          f"movie-movie {n_movie_edges}本 (node数{n_movie_nodes})")
    if n_user_nodes > 1:
        print(f"  user側 平均次数(仮想エッジ): {2 * n_user_edges / n_user_nodes:.2f}")
    if n_movie_nodes > 1:
        print(f"  movie側 平均次数(仮想エッジ): {2 * n_movie_edges / n_movie_nodes:.2f}")

    return nodes, idx, common_deg, weight, is_user
