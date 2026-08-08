"""One-off exploratory diagnostics (distribution/null-model/bias reports)."""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities, modularity
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

from bipartite_layout.caching import get_edge_direction_cached, get_matrices_cached, get_small_subgraph_cached
from bipartite_layout.config import DEFAULT_CONFIG
from bipartite_layout.experiment_context import save_figure
from bipartite_layout.layout_core import combined_stress_and_grad, make_initial_positions, stress_and_grad


def inspect_common_deg_distribution(G, save_path="common_deg_distribution.png"):
    nodes = list(G.nodes())
    values = []
    for i, ni in enumerate(nodes):
        for nj in nodes[i + 1:]:
            if ni[0] != nj[0]:
                continue
            ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
            union = ni_set | nj_set
            if union:
                sim = len(ni_set & nj_set) / len(union)
                if sim > 0:
                    values.append(sim)

    values = np.array(values)
    print(f"共通度>0のペア数: {len(values)}")
    if len(values):
        print(f"分位点: min={values.min():.3f}, 25%={np.percentile(values,25):.3f}, "
              f"50%={np.percentile(values,50):.3f}, 75%={np.percentile(values,75):.3f}, max={values.max():.3f}")

    fig = plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30)
    plt.xlabel("common_deg (Jaccard similarity, > 0 only)")
    plt.ylabel("pair count")
    plt.title("Distribution of common_deg for same-type pairs")
    save_figure(fig, save_path)
    return values


def compare_threshold_candidates(G, candidates):
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    common_deg = np.zeros((N, N))

    for i, ni in enumerate(nodes):
        for nj in nodes[i + 1:]:
            if ni[0] != nj[0]:
                continue
            j = idx[nj]
            ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
            union = ni_set | nj_set
            if union:
                sim = len(ni_set & nj_set) / len(union)
                common_deg[i, j] = common_deg[j, i] = sim

    is_same_type_node = {}
    for i, ni in enumerate(nodes):
        is_same_type_node[i] = any(nj[0] == ni[0] for j, nj in enumerate(nodes) if j != i)

    print(f"{'threshold':>10} | {'残る仮想エッジ数':>14} | {'孤立ノード数':>10}")
    print("-" * 42)
    for th in candidates:
        masked = common_deg.copy()
        masked[masked < th] = 0.0
        n_edges = int(np.sum(masked > 0) / 2)

        n_isolated = 0
        for i in range(N):
            if not is_same_type_node[i]:
                continue
            if np.sum(masked[i] > 0) == 0:
                n_isolated += 1

        print(f"{th:>10.3f} | {n_edges:>14d} | {n_isolated:>10d}")

    return common_deg


def inspect_common_deg_distribution_by_type(G, save_path="common_deg_distribution_by_type.png"):
    user_nodes = [n for n in G.nodes() if n.startswith("u_")]
    movie_nodes = [n for n in G.nodes() if n.startswith("m_")]
    print(f"\nノード内訳: user={len(user_nodes)}, movie={len(movie_nodes)}")

    def pairwise_values(node_list):
        vals = []
        for i, ni in enumerate(node_list):
            for nj in node_list[i + 1:]:
                ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
                union = ni_set | nj_set
                if union:
                    sim = len(ni_set & nj_set) / len(union)
                    if sim > 0:
                        vals.append(sim)
        return np.array(vals)

    user_values = pairwise_values(user_nodes)
    movie_values = pairwise_values(movie_nodes)

    def report(name, values, total_pairs):
        print(f"\n[{name}] 共通度>0のペア数: {len(values)} / 全ペア数(同種): {total_pairs}")
        if len(values):
            print(f"  分位点: min={values.min():.3f}, 25%={np.percentile(values,25):.3f}, "
                  f"50%={np.percentile(values,50):.3f}, 75%={np.percentile(values,75):.3f}, "
                  f"max={values.max():.3f}")
            n_over_th = int(np.sum(values >= DEFAULT_CONFIG.graph_build.threshold_common_deg)) if DEFAULT_CONFIG.graph_build.threshold_common_deg is not None else None
            if n_over_th is not None:
                print(f"  閾値{DEFAULT_CONFIG.graph_build.threshold_common_deg}以上のペア数: {n_over_th} "
                      f"({100 * n_over_th / total_pairs:.1f}% of 全同種ペア)")

    n_user_pairs = len(user_nodes) * (len(user_nodes) - 1) // 2
    n_movie_pairs = len(movie_nodes) * (len(movie_nodes) - 1) // 2
    report("user-user", user_values, n_user_pairs)
    report("movie-movie", movie_values, n_movie_pairs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    ax1.hist(user_values, bins=20, color="tab:blue")
    ax1.set_title(f"user-user (n={len(user_nodes)})")
    ax1.set_xlabel("common_deg (> 0 only)")
    ax1.set_ylabel("pair count")
    if DEFAULT_CONFIG.graph_build.threshold_common_deg is not None:
        ax1.axvline(DEFAULT_CONFIG.graph_build.threshold_common_deg, color="black", linestyle="--", linewidth=1)

    ax2.hist(movie_values, bins=20, color="tab:red")
    ax2.set_title(f"movie-movie (n={len(movie_nodes)})")
    ax2.set_xlabel("common_deg (> 0 only)")
    if DEFAULT_CONFIG.graph_build.threshold_common_deg is not None:
        ax2.axvline(DEFAULT_CONFIG.graph_build.threshold_common_deg, color="black", linestyle="--", linewidth=1)

    save_figure(fig, save_path, message=f"\n保存しました: {save_path}")

    return user_values, movie_values


def compute_jaccard_distribution_stats(G):
    user_nodes = [n for n in G.nodes() if n.startswith("u_")]
    movie_nodes = [n for n in G.nodes() if n.startswith("m_")]

    def jaccard_values(node_list):
        vals = []
        for i, ni in enumerate(node_list):
            for nj in node_list[i + 1:]:
                ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
                union = ni_set | nj_set
                sim = (len(ni_set & nj_set) / len(union)) if union else 0.0
                vals.append(sim)
        return np.array(vals)

    def stats(vals):
        if len(vals) == 0 or vals.mean() == 0:
            return float("nan"), float("nan"), float("nan")
        return float(vals.mean()), float(vals.std()), float(vals.std() / vals.mean())

    user_mean, user_std, user_cv = stats(jaccard_values(user_nodes))
    movie_mean, movie_std, movie_cv = stats(jaccard_values(movie_nodes))

    return {
        "user_mean": user_mean, "user_std": user_std, "user_cv": user_cv,
        "movie_mean": movie_mean, "movie_std": movie_std, "movie_cv": movie_cv,
    }


def compute_null_model_ratio(G):
    user_nodes = [n for n in G.nodes() if n.startswith("u_")]
    movie_nodes = [n for n in G.nodes() if n.startswith("m_")]

    def ratio_for(node_list, opposite_count):
        observed_vals, expected_vals = [], []
        for i, ni in enumerate(node_list):
            for nj in node_list[i + 1:]:
                ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
                union = ni_set | nj_set
                if not union:
                    continue
                observed = len(ni_set & nj_set) / len(union)
                d_i, d_j, D = len(ni_set), len(nj_set), opposite_count
                exp_intersection = (d_i * d_j / D) if D > 0 else 0.0
                exp_union = d_i + d_j - exp_intersection
                expected = (exp_intersection / exp_union) if exp_union > 0 else 0.0
                observed_vals.append(observed)
                expected_vals.append(expected)
        if not observed_vals:
            return float("nan")
        obs_mean, exp_mean = np.mean(observed_vals), np.mean(expected_vals)
        return (obs_mean / exp_mean) if exp_mean > 0 else float("nan")

    ratio_user = ratio_for(user_nodes, len(movie_nodes))
    ratio_movie = ratio_for(movie_nodes, len(user_nodes))
    return ratio_user, ratio_movie


def run_sampling_parameter_sweep(M, configs, threshold=DEFAULT_CONFIG.graph_build.threshold_common_deg):
    print(f"\n{'config':>48} | {'n_u':>4} | {'n_m':>4} | {'mod(u)':>7} | {'mod(m)':>7} | "
          f"{'ratio(u)':>9} | {'ratio(m)':>9} | {'CV(u)':>7} | {'CV(m)':>7}")
    print("-" * 122)

    results = []
    for cfg in configs:
        G = get_small_subgraph_cached(M, **cfg)
        n_user = sum(1 for n in G.nodes() if n.startswith("u_"))
        n_movie = sum(1 for n in G.nodes() if n.startswith("m_"))

        nodes, idx, common_deg, weight, is_user = get_matrices_cached(
            G, "degree", threshold, DEFAULT_CONFIG.graph_build.top_k_same_type, DEFAULT_CONFIG.graph_build.mutual_top_k_only
        )
        mod_user, mod_movie = analyze_virtual_edge_graph_structure_quiet(
            nodes, common_deg, is_user, threshold
        )

        ratio_user, ratio_movie = compute_null_model_ratio(G)
        cv_stats = compute_jaccard_distribution_stats(G)

        cfg_label = ",".join(f"{k.replace('n_', '')}={v}" for k, v in cfg.items())
        print(f"{cfg_label:>48} | {n_user:>4d} | {n_movie:>4d} | "
              f"{mod_user:>7.3f} | {mod_movie:>7.3f} | "
              f"{ratio_user:>9.2f} | {ratio_movie:>9.2f} | "
              f"{cv_stats['user_cv']:>7.2f} | {cv_stats['movie_cv']:>7.2f}")

        results.append({
            "config": cfg, "n_user": n_user, "n_movie": n_movie,
            "mod_user": mod_user, "mod_movie": mod_movie,
            "ratio_user": ratio_user, "ratio_movie": ratio_movie,
            **cv_stats,
        })

    print("\n見方: n_u/n_mがuser数/movie数、mod(u)/mod(m)がmodularity、"
          "ratio(u)/ratio(m)が次数のみのnullモデルに対する観測/期待比、"
          "CV(u)/CV(m)がJaccard値の変動係数(小さいほど均一に重なり合っている)。"
          "user数を増やす・movieと揃えることでこれらがどう変化するかを比較してください。")

    return results


def analyze_degree_and_null_model(G, save_path="degree_null_model.png"):
    user_nodes = [n for n in G.nodes() if n.startswith("u_")]
    movie_nodes = [n for n in G.nodes() if n.startswith("m_")]

    user_degrees = [G.degree(n) for n in user_nodes]
    movie_degrees = [G.degree(n) for n in movie_nodes]
    print(f"\n平均次数(このサブグラフ内): "
          f"user={np.mean(user_degrees):.2f} (n={len(user_nodes)}), "
          f"movie={np.mean(movie_degrees):.2f} (n={len(movie_nodes)})")

    def analyze_one_type(node_list, opposite_count, label):
        observed_vals, expected_vals = [], []
        for i, ni in enumerate(node_list):
            for nj in node_list[i + 1:]:
                ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
                union = ni_set | nj_set
                if not union:
                    continue
                observed = len(ni_set & nj_set) / len(union)
                d_i, d_j, D = len(ni_set), len(nj_set), opposite_count
                exp_intersection = (d_i * d_j / D) if D > 0 else 0.0
                exp_union = d_i + d_j - exp_intersection
                expected = (exp_intersection / exp_union) if exp_union > 0 else 0.0
                observed_vals.append(observed)
                expected_vals.append(expected)

        observed_vals = np.array(observed_vals)
        expected_vals = np.array(expected_vals)

        if len(observed_vals) == 0:
            print(f"\n[{label}] ペアが存在しないため分析をスキップ")
            return observed_vals, expected_vals

        obs_mean, exp_mean = observed_vals.mean(), expected_vals.mean()
        print(f"\n[{label}] 実測Jaccard平均: {obs_mean:.3f}, "
              f"次数のみのnullモデルでの期待Jaccard平均: {exp_mean:.3f}")
        ratio = (obs_mean / exp_mean) if exp_mean > 0 else float("nan")
        print(f"  観測/期待の比: {ratio:.2f}倍")
        if not np.isnan(ratio) and ratio < 1.3:
            print("  → 観測値が期待値に近く、次数(集合サイズ)の違いだけでかなりの部分が"
                  "説明できそうです。")
        elif not np.isnan(ratio):
            print("  → 観測値が期待値を上回っており、次数だけでは説明できない"
                  "構造的な類似(あるいはこの近似nullモデル自体の粗さ)が疑われます。")

        return observed_vals, expected_vals

    user_obs, user_exp = analyze_one_type(user_nodes, len(movie_nodes), "user-user")
    movie_obs, movie_exp = analyze_one_type(movie_nodes, len(user_nodes), "movie-movie")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    lim = max(
        user_obs.max() if len(user_obs) else 0, user_exp.max() if len(user_exp) else 0,
        movie_obs.max() if len(movie_obs) else 0, movie_exp.max() if len(movie_exp) else 0,
        0.1
    )
    for ax, obs, exp, label, color in [
        (ax1, user_obs, user_exp, "user-user", "tab:blue"),
        (ax2, movie_obs, movie_exp, "movie-movie", "tab:red"),
    ]:
        if len(obs):
            ax.scatter(exp, obs, alpha=0.6, color=color)
        ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("expected Jaccard (degree-only null model)")
        ax.set_ylabel("observed Jaccard")
        ax.set_title(label)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    save_figure(fig, save_path,
                message=f"\n保存しました: {save_path}"
                        "(対角線より上にある点が多いほど、次数だけでは説明できない超過分が"
                        "大きいことを示す)")

    return user_obs, user_exp, movie_obs, movie_exp


def analyze_common_neighbor_popularity_bias(G, M, save_path="popularity_bias.png"):
    def analyze_one_type(node_list, label):
        jaccard_vals = []
        popularity_vals = []
        for i, ni in enumerate(node_list):
            for nj in node_list[i + 1:]:
                ni_set, nj_set = set(G.neighbors(ni)), set(G.neighbors(nj))
                union = ni_set | nj_set
                if not union:
                    continue
                intersection = ni_set & nj_set
                if not intersection:
                    continue
                sim = len(intersection) / len(union)
                popularity = np.mean([M.degree(n) for n in intersection])
                jaccard_vals.append(sim)
                popularity_vals.append(popularity)

        jaccard_vals = np.array(jaccard_vals)
        popularity_vals = np.array(popularity_vals)

        if len(jaccard_vals) < 5:
            print(f"\n[{label}] ペア数が少なすぎるため人気度バイアス分析をスキップ"
                  f"(n={len(jaccard_vals)})")
            return jaccard_vals, popularity_vals, float("nan")

        corr = float(np.corrcoef(jaccard_vals, popularity_vals)[0, 1])
        print(f"\n[{label}] 共通度(Jaccard) と 共通隣接ノードの人気度(元データMでの次数) の相関: "
              f"{corr:.3f} (n={len(jaccard_vals)}ペア)")

        terciles = np.percentile(jaccard_vals, [33.3, 66.7])
        low_mask = jaccard_vals <= terciles[0]
        mid_mask = (jaccard_vals > terciles[0]) & (jaccard_vals <= terciles[1])
        high_mask = jaccard_vals > terciles[1]
        print(f"  共通度が低い1/3   (n={int(low_mask.sum())}): 平均人気度 {popularity_vals[low_mask].mean():.1f}")
        print(f"  共通度が中間の1/3 (n={int(mid_mask.sum())}): 平均人気度 {popularity_vals[mid_mask].mean():.1f}")
        print(f"  共通度が高い1/3   (n={int(high_mask.sum())}): 平均人気度 {popularity_vals[high_mask].mean():.1f}")

        if corr > 0.3:
            print("  → 正の相関が見られます。共通度が高いペアほど人気ノードを共有している傾向があり、"
                  "「個性の一致」よりも人気度バイアスが共通度を底上げしている可能性が高いです。")
        elif corr < -0.3:
            print("  → 負の相関です。共通度が高いペアほどマイナーなノードを共有しています。")
        else:
            print("  → 明確な相関は見られません。人気度バイアスだけでは今回の非対称性は説明しにくいかもしれません。")

        return jaccard_vals, popularity_vals, corr

    user_nodes = [n for n in G.nodes() if n.startswith("u_")]
    movie_nodes = [n for n in G.nodes() if n.startswith("m_")]

    user_jaccard, user_pop, user_corr = analyze_one_type(user_nodes, "user-user")
    movie_jaccard, movie_pop, movie_corr = analyze_one_type(movie_nodes, "movie-movie")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    if len(user_jaccard):
        ax1.scatter(user_jaccard, user_pop, alpha=0.6, color="tab:blue")
    ax1.set_xlabel("Jaccard common_deg")
    ax1.set_ylabel("mean popularity of shared movies (degree in M)")
    ax1.set_title(f"user-user (corr={user_corr:.3f})")

    if len(movie_jaccard):
        ax2.scatter(movie_jaccard, movie_pop, alpha=0.6, color="tab:red")
    ax2.set_xlabel("Jaccard common_deg")
    ax2.set_ylabel("mean activity of shared users (degree in M)")
    ax2.set_title(f"movie-movie (corr={movie_corr:.3f})")

    save_figure(fig, save_path, message=f"\n保存しました: {save_path}")

    return user_corr, movie_corr


def analyze_virtual_edge_graph_structure(nodes, common_deg, is_user, threshold_common_deg):
    idx = {n: i for i, n in enumerate(nodes)}

    def build_subgraph(node_mask):
        sub_nodes = [n for n, m in zip(nodes, node_mask) if m]
        H = nx.Graph()
        H.add_nodes_from(sub_nodes)
        for i, ni in enumerate(sub_nodes):
            for nj in sub_nodes[i + 1:]:
                if common_deg[idx[ni], idx[nj]] > 0:
                    H.add_edge(ni, nj, weight=common_deg[idx[ni], idx[nj]])
        return H

    def report(name, H):
        n_components = nx.number_connected_components(H)
        isolated = sum(1 for n in H.nodes() if H.degree(n) == 0)
        print(f"\n[{name}] 仮想エッジグラフ(閾値{threshold_common_deg}適用後): "
              f"{H.number_of_nodes()}ノード, {H.number_of_edges()}エッジ")
        print(f"  連結成分数: {n_components} (うち孤立ノード: {isolated})")

        communities = louvain_communities(H, weight="weight", seed=0)
        n_communities = len(communities)
        print(f"  Louvainコミュニティ数: {n_communities} "
              f"(サイズ内訳: {sorted([len(c) for c in communities], reverse=True)})")

        if n_communities > 1 and H.number_of_edges() > 0:
            mod_score = modularity(H, communities, weight="weight")
            print(f"  Modularity(分割の質、目安: 0.3以上で明確な分割、"
                  f"0.1未満なら実質一枚岩とみなせる): {mod_score:.3f}")
        else:
            print("  Modularity: コミュニティが1つ、またはエッジなしのため計算対象外")

        return n_components

    H_user = build_subgraph(is_user)
    H_movie = build_subgraph(~is_user)
    n_comp_user = report("user-user", H_user)
    n_comp_movie = report("movie-movie", H_movie)

    print("\n※ 上のDBSCANクラスタ数(n_cluster(user)/n_cluster(movie))の平均値と、"
          "ここで出た連結成分数を見比べてください。近い値であれば、"
          "クラスタ分裂は仮想エッジグラフ自体の構造に起因すると言えます。"
          "一方、連結成分数は1でもLouvainコミュニティ数が2以上の場合は、"
          "modularityスコアも確認してください。")

    return n_comp_user, n_comp_movie


def analyze_virtual_edge_graph_structure_quiet(nodes, common_deg, is_user, threshold_common_deg):
    idx = {n: i for i, n in enumerate(nodes)}

    def build_subgraph_and_modularity(node_mask):
        sub_nodes = [n for n, m in zip(nodes, node_mask) if m]
        H = nx.Graph()
        H.add_nodes_from(sub_nodes)
        for i, ni in enumerate(sub_nodes):
            for nj in sub_nodes[i + 1:]:
                if common_deg[idx[ni], idx[nj]] > 0:
                    H.add_edge(ni, nj, weight=common_deg[idx[ni], idx[nj]])

        if H.number_of_edges() == 0:
            return float("nan")
        communities = louvain_communities(H, weight="weight", seed=0)
        if len(communities) <= 1:
            return float("nan")
        return modularity(H, communities, weight="weight")

    mod_user = build_subgraph_and_modularity(is_user)
    mod_movie = build_subgraph_and_modularity(~is_user)
    return mod_user, mod_movie


def check_alpha_term_balance(M, sampling_kwargs=None, weight_mode="degree",
                              threshold_common_deg=DEFAULT_CONFIG.graph_build.threshold_common_deg,
                              top_k_same_type=DEFAULT_CONFIG.graph_build.top_k_same_type,
                              alphas=None, n_seeds=3, methods=("B", "C"), gamma=0.1,
                              warm_start=True, save_path="alpha_term_balance.png"):
    """
    実エッジ(weight, cross-type)と仮想エッジ(common_deg, same-type)の力の大きさが
    釣り合っているかを検証する(先生からのご指摘への対応)。combined_stress_and_grad
    は total_stress = alpha*s_a(仮想) + (1-alpha)*s_b(実) [+ gamma*方向整列] で
    項を混合しているが、matrices.pyではweightだけが[0.4, 1.0]に正規化されており、
    common_degには対応する正規化が無い(build_matrices内のコメント参照)。この
    非対称な設計が、「0<alpha<(1-epsilon)ではレイアウトがほとんど変わらず、
    alpha≒1で突然大きく変わる」という病的な挙動を引き起こしていないかを、
    以下の観点で確認する。

    【初版からの訂正】初版では、alphaを刻んだ際のレイアウトの「移動量」を
    生の座標の差 ||coords_new - coords_prev|| で測っていたが、これは誤りだった。
    stress majorizationのレイアウトは全体の回転・平行移動・鏡映に対して不変
    (どの変換もペア間距離を一切変えないため、目的関数の値も変えない)なので、
    生の座標差は「実際に形が変わったか」ではなく「たまたま前回と同じ向きに
    収束したか」を測ってしまう。実際にこれが原因で、method="C"(alpha混合+
    方向整列の同時最適化)についてalpha=1直前で実際に起きている大きな
    ジャンプを、初版の指標は見逃していた。この修正版では2つの対策を行う。

    1. 移動量を、座標ではなくペア間距離行列の差(scipy.spatial.distance.pdist)
       で測る(=shape_distance)。回転・平行移動・鏡映に不変なので「本当に
       形が変わったか」を正しく捉えられる。
    2. warm_start=Trueの場合、各alphaの初期値を(ランダムではなく)直前の
       alphaの収束後レイアウトにする。「たまたま別の向きに収束した」ことに
       よる見かけ上の移動を防ぎ、alphaを少しずつ動かして追跡する実務上
       妥当な手続きにも対応する。

    修正後、method="B"(alpha混合のみ、方向整列なし)では0<alpha<(1-epsilon)
    でのジャンプは確認されなかった(勾配ノルムも全alphaで小さく、収束は安定)。
    一方method="C"(alpha混合+方向整列の同時最適化。おそらく実際に使っている
    のはこちら)では、alpha=0.99→1.00の間で他のどのalpha刻みよりもはるかに
    大きい、warm_startでも消えない実際の形状変化が確認された。原因は「力が
    徐々に偏っていく」ことではなく、alpha=1.0において実エッジ項の係数
    (1-alpha)が小さくなるのではなく文字通り0になり、実エッジ制約が目的関数
    から完全に消える(質的に異なる問題になる)、alpha=1という一点での構造的な
    不連続性だと考えられる。

    さらに、収束後の勾配ノルム ||alpha*g_a|| と ||(1-alpha)*g_b|| の比較
    (力バランス)も行う。alphaと(1-alpha)という「係数」が対称であることと、
    実際に各項が最適化に与える「力」が釣り合っていることは別物である
    (係数バランス vs 力バランス)。total_stressの勾配は収束時に0に近づくが、
    これはalpha*g_a と (1-alpha)*g_b が個別に小さいからではなく、反発力も
    含めて互いに打ち消し合っているだけの可能性があるため、項の大きさの比較
    だけでは見えない非対称性をここで確認する。method="B"では、alpha=0.5の
    時点で実エッジ側の力が仮想エッジ側の約2倍強く、力が釣り合うのは
    alpha≒0.8付近だった(=係数上0.5が均等に見えても、実際の力は均等でない)。
    """
    if sampling_kwargs is None:
        sampling_kwargs = {}
    if alphas is None:
        alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    if isinstance(methods, str):
        methods = (methods,)

    G = get_small_subgraph_cached(M, **sampling_kwargs)
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, weight_mode, threshold_common_deg, top_k_same_type
    )
    _, _, direction_precomputed = get_edge_direction_cached(G, node_idx)
    N = len(nodes)

    nz_common = common_deg[common_deg > 0]
    nz_weight = weight[weight > 0]
    print(f"仮想エッジ(common_deg) 非ゼロ範囲: [{nz_common.min():.4f}, {nz_common.max():.4f}], "
          f"平均={nz_common.mean():.4f}, 本数={int(np.sum(common_deg > 0) / 2)}")
    print(f"実エッジ(weight)      非ゼロ範囲: [{nz_weight.min():.4f}, {nz_weight.max():.4f}], "
          f"平均={nz_weight.mean():.4f}, 本数={int(np.sum(weight > 0) / 2)}")

    def raw_terms_at(coords_flat):
        """stress_and_gradをalpha=1(仮想のみ)とalpha=0(実のみ)で個別に呼び、
        alpha重み付け前の生の項の大きさ・勾配 (s_a, g_a), (s_b, g_b) を取り出す。"""
        s_a, g_a = stress_and_grad(coords_flat, common_deg, weight, 1.0, 0.0, 0.0, is_user, True)
        s_b, g_b = stress_and_grad(coords_flat, common_deg, weight, 0.0, 0.0, 0.0, is_user, True)
        return s_a, g_a, s_b, g_b

    def shape_distance(coords_a, coords_b):
        """座標そのものの差ではなく、ペア間距離行列の差で「形の変化」を測る
        (回転・平行移動・鏡映に不変)。"""
        da, db = pdist(coords_a), pdist(coords_b)
        return np.linalg.norm(da - db) / (np.linalg.norm(da) + 1e-12)

    print(f"\n--- ランダム初期配置{n_seeds}通りでの生の項の大きさ ---")
    init_ratios = []
    for seed in range(n_seeds):
        coords = make_initial_positions(nodes, is_user, alpha=0.5, seed=seed)
        s_a, _, s_b, _ = raw_terms_at(coords)
        ratio = s_b / s_a if s_a > 0 else float("inf")
        init_ratios.append(ratio)
        print(f"  seed={seed}: s_a(仮想)={s_a:.4f}  s_b(実)={s_b:.4f}  s_b/s_a={ratio:.2f}")

    results_by_method = {}
    for method in methods:
        print(f"\n--- method={method} (warm_start={warm_start}): alpha刻みごとの収束レイアウト "
              f"({n_seeds}シードの平均) ---")
        movements_by_alpha = {a: [] for a in alphas[1:]}
        s_a_by_alpha = {a: [] for a in alphas}
        s_b_by_alpha = {a: [] for a in alphas}
        force_ratio_by_alpha = {a: [] for a in alphas}
        n_unconverged = 0
        for seed in range(n_seeds):
            prev_coords = None
            for a in alphas:
                if warm_start and prev_coords is not None:
                    x0 = prev_coords.flatten()
                else:
                    x0 = make_initial_positions(nodes, is_user, a, seed=seed)

                g = 0.0 if method == "B" else gamma
                result = minimize(
                    combined_stress_and_grad, x0,
                    args=(common_deg, weight, a, nodes, node_idx, direction_precomputed, g, 0.3, 0.3, is_user, True),
                    jac=True, method="L-BFGS-B", options={"maxiter": 3000},
                )
                if not result.success:
                    n_unconverged += 1
                coords = result.x.reshape(N, 2)
                s_a, g_a, s_b, g_b = raw_terms_at(result.x)
                s_a_by_alpha[a].append(s_a)
                s_b_by_alpha[a].append(s_b)
                # 係数バランス(alpha, 1-alpha)ではなく、収束後にそれぞれの項が
                # 実際にどれだけ強く配置を引っ張ろうとしていたか(力バランス)を見る。
                # alpha=0, 1では定義上どちらかが0になるだけの自明なケースなので除外。
                if 0.0 < a < 1.0:
                    virtual_force = np.linalg.norm(a * g_a)
                    real_force = np.linalg.norm((1 - a) * g_b)
                    force_ratio_by_alpha[a].append(virtual_force / (real_force + 1e-12))
                if prev_coords is not None:
                    movements_by_alpha[a].append(shape_distance(coords, prev_coords))
                prev_coords = coords

        mean_s_a = [np.mean(s_a_by_alpha[a]) for a in alphas]
        mean_s_b = [np.mean(s_b_by_alpha[a]) for a in alphas]
        mean_movement = [np.mean(movements_by_alpha[a]) for a in alphas[1:]]
        force_alphas = [a for a in alphas if 0.0 < a < 1.0]
        mean_force_ratio = [np.mean(force_ratio_by_alpha[a]) for a in force_alphas]

        if n_unconverged > 0:
            print(f"  警告: {n_unconverged}/{n_seeds * len(alphas)} 回、L-BFGS-Bが収束条件を"
                  f"満たさずに終了しました(結果の解釈に注意)。")
        for a, ma, mb in zip(alphas, mean_s_a, mean_s_b):
            mv = np.mean(movements_by_alpha[a]) if a in movements_by_alpha and movements_by_alpha[a] else float("nan")
            fr = np.mean(force_ratio_by_alpha[a]) if force_ratio_by_alpha.get(a) else float("nan")
            print(f"  alpha={a:.2f}: s_a(仮想)={ma:8.4f} s_b(実)={mb:8.4f} "
                  f"力比||a*g_a||/||(1-a)*g_b||={fr:6.4f} 形状変化量(shape_distance)={mv:.4f}")

        results_by_method[method] = {
            "mean_s_a": mean_s_a, "mean_s_b": mean_s_b, "mean_movement": mean_movement,
            "force_alphas": force_alphas, "mean_force_ratio": mean_force_ratio,
        }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    for method, res in results_by_method.items():
        ax1.plot(alphas, res["mean_s_a"], marker="o", label=f"s_a (virtual) [{method}]")
        ax1.plot(alphas, res["mean_s_b"], marker="s", label=f"s_b (real) [{method}]")
        ax2.plot(alphas[1:], res["mean_movement"], marker="o", label=f"[{method}]")
        ax3.plot(res["force_alphas"], res["mean_force_ratio"], marker="o", label=f"[{method}]")

    ax1.set_xlabel("alpha")
    ax1.set_ylabel("raw stress term (before alpha weighting)")
    ax1.legend(fontsize=8)
    ax1.set_title("Raw term magnitude vs alpha")

    ax2.set_xlabel("alpha")
    ax2.set_ylabel("shape_distance from previous alpha\n(rotation/reflection-invariant)")
    ax2.set_title("Layout shape change across the alpha sweep\n(a spike only near alpha=1 indicates the failure mode)")
    ax2.legend(fontsize=8)

    ax3.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="balanced (ratio=1)")
    ax3.set_xlabel("alpha")
    ax3.set_ylabel("||alpha*g_a|| / ||(1-alpha)*g_b||")
    ax3.set_title("Force balance at convergence\n(coefficient alpha != actual force balance)")
    ax3.legend(fontsize=8)

    save_figure(fig, save_path, message=f"\n保存しました: {save_path}")

    return {"alphas": alphas, "init_ratios": init_ratios, "results_by_method": results_by_method}
