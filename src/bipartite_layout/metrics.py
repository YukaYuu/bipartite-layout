"""Layout quality metrics: separation, clustering, NMI, cluster quality."""

from collections import Counter

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import normalized_mutual_info_score

from bipartite_layout.config import DEFAULT_CONFIG


def compute_separation_metrics(coords, is_user):
    centroid_user = coords[is_user].mean(axis=0)
    centroid_movie = coords[~is_user].mean(axis=0)
    centroid_dist = np.linalg.norm(centroid_user - centroid_movie)

    overall_centroid = coords.mean(axis=0)
    overall_spread = np.mean(np.linalg.norm(coords - overall_centroid, axis=1)) + 1e-9
    centroid_separation = centroid_dist / overall_spread

    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    N = len(coords)
    ratios = []
    for i in range(N):
        same_mask = (is_user == is_user[i]); same_mask[i] = False
        opp_mask = ~same_mask; opp_mask[i] = False
        if same_mask.sum() == 0 or opp_mask.sum() == 0:
            continue
        nn_same = dist[i][same_mask].min()
        nn_opp = dist[i][opp_mask].min()
        ratios.append(nn_opp / (nn_same + 1e-9))

    nn_ratio = float(np.mean(ratios)) if ratios else float("nan")
    return centroid_separation, nn_ratio


def compute_cluster_metrics(coords, mask, min_samples=DEFAULT_CONFIG.cluster.dbscan_min_samples,
                             eps_scale=DEFAULT_CONFIG.cluster.dbscan_eps_scale):
    subset = coords[mask]
    n = len(subset)
    if n <= min_samples:
        return np.nan, np.nan

    dist = np.linalg.norm(subset[:, None, :] - subset[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    k = min(min_samples, n - 1)
    kth_dist = np.sort(dist, axis=1)[:, k - 1]
    eps = eps_scale * np.median(kth_dist)
    if eps <= 0:
        eps = 1e-6

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(subset)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = float(np.mean(labels == -1))
    return float(n_clusters), noise_ratio


def compute_split_nmi(nodes, is_user, common_deg, edges, edge_labels):
    node_graph = nx.Graph()
    node_graph.add_nodes_from(nodes)
    N = len(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            if common_deg[i, j] > 0:
                node_graph.add_edge(nodes[i], nodes[j], weight=common_deg[i, j])
    node_communities = louvain_communities(node_graph, weight="weight", seed=0)
    node_cluster_of = {}
    for cid, comm in enumerate(node_communities):
        for n in comm:
            node_cluster_of[n] = cid

    dominant_edge_cluster_of = {}
    for n in nodes:
        connected = [edge_labels[k] for k, (u, v) in enumerate(edges) if u == n or v == n]
        if connected:
            dominant_edge_cluster_of[n] = Counter(connected).most_common(1)[0][0]

    def compute_nmi_for_subset(node_subset):
        common = [n for n in node_subset if n in dominant_edge_cluster_of and n in node_cluster_of]
        if len(common) < 2:
            return float("nan"), len(common)
        node_labels_arr = [node_cluster_of[n] for n in common]
        edge_dom_labels_arr = [dominant_edge_cluster_of[n] for n in common]
        if len(set(node_labels_arr)) < 2 or len(set(edge_dom_labels_arr)) < 2:
            return float("nan"), len(common)
        return normalized_mutual_info_score(node_labels_arr, edge_dom_labels_arr), len(common)

    user_nodes = [n for n in nodes if n.startswith("u_")]
    movie_nodes = [n for n in nodes if n.startswith("m_")]
    node_cluster_labels = set(node_cluster_of.get(n) for n in user_nodes if n in node_cluster_of)
    edge_cluster_labels = set(dominant_edge_cluster_of.get(n) for n in user_nodes if n in dominant_edge_cluster_of)
    print(f"user側のノードクラスタの種類数: {len(node_cluster_labels)}")
    print(f"user側の支配的エッジクラスタの種類数: {len(edge_cluster_labels)}")

    nmi_all, n_all = compute_nmi_for_subset(nodes)
    nmi_user, n_user = compute_nmi_for_subset(user_nodes)
    nmi_movie, n_movie = compute_nmi_for_subset(movie_nodes)

    print(f"全体のNMI: {nmi_all:.3f} (n={n_all})")
    print(f"user側のみのNMI: {nmi_user:.3f} (n={n_user})")
    print(f"movie側のみのNMI: {nmi_movie:.3f} (n={n_movie})")
    return nmi_all, nmi_user, nmi_movie


def compute_cluster_quality(nodes, is_user, common_deg, coords):
    """
    Cluster Quality (CQ): レイアウトの座標が、真の構造的クラスタ(common_degの
    Louvainコミュニティ)をどれだけ忠実に再現しているかを測る指標。Claudeとの
    相談から追加(cluster faithfulness)。既存のnn_ratio(user/movie間の分離度)や
    n_cluster(DBSCANでのクラスタ数)とは異なり、「レイアウト上の空間的な
    まとまりが、実際の構造的クラスタと一致しているか」を直接測る点が新しい。

    手順:
      1. common_deg(仮想エッジ)にLouvain法を適用し、"正解"クラスタを得る
         (compute_split_nmiと同じ計算方法。common_degはuser-user/movie-movie
         ブロックが非連結なため、実質的にtype別々にクラスタリングしているのと同じ)。
      2. user側・movie側それぞれについて、その正解クラスタ数kでレイアウト座標に
         k-meansを適用し、幾何的なクラスタを得る。
      3. 各正解クラスタについて、Jaccard類似度(|交差|/|和集合|)が最大となる
         幾何クラスタとマッチングし、そのJaccard値を採用する
         (どちらの正解クラスタも同じ幾何クラスタを最良と判断してよい、片方向マッチング)。
      4. 正解クラスタのサイズで重み付けした平均を、そのtypeのCQとする
         (大きいクラスタほど、その一致度の重要性を高く見る、という設計判断)。

    正解クラスタが2個未満(Louvainで1コミュニティにまとまってしまった等)の場合は
    compute_split_nmiと同様にnanを返す。

    戻り値: (cq_user, cq_movie)
    """
    node_graph = nx.Graph()
    node_graph.add_nodes_from(nodes)
    N = len(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            if common_deg[i, j] > 0:
                node_graph.add_edge(nodes[i], nodes[j], weight=common_deg[i, j])
    node_communities = louvain_communities(node_graph, weight="weight", seed=0)
    node_idx = {n: i for i, n in enumerate(nodes)}

    def cq_for_type(type_mask):
        type_nodes = set(n for n, m in zip(nodes, type_mask) if m)
        ground_truth = [c & type_nodes for c in node_communities]
        ground_truth = [c for c in ground_truth if len(c) > 0]
        if len(ground_truth) < 2:
            return float("nan")

        type_node_list = sorted(type_nodes, key=lambda n: node_idx[n])
        type_coords = np.array([coords[node_idx[n]] for n in type_node_list])
        k = len(ground_truth)
        km_labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(type_coords)
        geometric_clusters = [
            set(n for n, lbl in zip(type_node_list, km_labels) if lbl == cluster_id)
            for cluster_id in range(k)
        ]

        total_weight = 0
        weighted_score = 0.0
        for c_i in ground_truth:
            best_js = 0.0
            for c_j in geometric_clusters:
                union = c_i | c_j
                if union:
                    js = len(c_i & c_j) / len(union)
                    best_js = max(best_js, js)
            weighted_score += len(c_i) * best_js
            total_weight += len(c_i)
        return weighted_score / total_weight if total_weight > 0 else float("nan")

    cq_user = cq_for_type(is_user)
    cq_movie = cq_for_type(~is_user)
    return cq_user, cq_movie


def compute_nn_distance_cv(coords):
    """
    レイアウトの「格子っぽさ」を定量化する簡易指標: 各ノードの最近傍距離の
    変動係数(CV = 標準偏差/平均)。格子状に均一配置されているほど最近傍距離が
    揃うためCVは小さく、疎密のあるクラスタ状配置ほどCVは大きくなる
    (点パターン統計のClark-Evans最近傍指数に近い発想)。
    """
    N = coords.shape[0]
    if N < 2:
        return float("nan")
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn_dist = dist.min(axis=1)
    mean = nn_dist.mean()
    if mean < 1e-12:
        return float("nan")
    return float(nn_dist.std() / mean)
