"""One-off exploratory diagnostics (distribution/null-model/bias reports)."""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities, modularity

from bipartite_layout.caching import get_matrices_cached, get_small_subgraph_cached
from bipartite_layout.config import MUTUAL_TOP_K_ONLY, THRESHOLD_COMMON_DEG, TOP_K_SAME_TYPE


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

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30)
    plt.xlabel("common_deg (Jaccard similarity, > 0 only)")
    plt.ylabel("pair count")
    plt.title("Distribution of common_deg for same-type pairs")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"保存しました: {save_path}")
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
            n_over_th = int(np.sum(values >= THRESHOLD_COMMON_DEG)) if THRESHOLD_COMMON_DEG is not None else None
            if n_over_th is not None:
                print(f"  閾値{THRESHOLD_COMMON_DEG}以上のペア数: {n_over_th} "
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
    if THRESHOLD_COMMON_DEG is not None:
        ax1.axvline(THRESHOLD_COMMON_DEG, color="black", linestyle="--", linewidth=1)

    ax2.hist(movie_values, bins=20, color="tab:red")
    ax2.set_title(f"movie-movie (n={len(movie_nodes)})")
    ax2.set_xlabel("common_deg (> 0 only)")
    if THRESHOLD_COMMON_DEG is not None:
        ax2.axvline(THRESHOLD_COMMON_DEG, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n保存しました: {save_path}")

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


def run_sampling_parameter_sweep(M, configs, threshold=THRESHOLD_COMMON_DEG):
    print(f"\n{'config':>48} | {'n_u':>4} | {'n_m':>4} | {'mod(u)':>7} | {'mod(m)':>7} | "
          f"{'ratio(u)':>9} | {'ratio(m)':>9} | {'CV(u)':>7} | {'CV(m)':>7}")
    print("-" * 122)

    results = []
    for cfg in configs:
        G = get_small_subgraph_cached(M, **cfg)
        n_user = sum(1 for n in G.nodes() if n.startswith("u_"))
        n_movie = sum(1 for n in G.nodes() if n.startswith("m_"))

        nodes, idx, common_deg, weight, is_user = get_matrices_cached(
            G, "degree", threshold, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n保存しました: {save_path}"
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n保存しました: {save_path}")

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
