"""Edge-direction alignment: similarity, clustering, and the stress term."""

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def calc_edge_similarity(G):
    edges = list(G.edges())
    similarity = {}
    for i, (u1, v1) in enumerate(edges):
        for j, (u2, v2) in enumerate(edges):
            if i >= j:
                continue
            neighbors1 = set(G.neighbors(u1)) | set(G.neighbors(v1))
            neighbors2 = set(G.neighbors(u2)) | set(G.neighbors(v2))
            common = len(neighbors1 & neighbors2)
            union = len(neighbors1 | neighbors2)
            similarity[(i, j)] = common / (union + 1)
    return edges, similarity


def cluster_edges_by_community(edges, similarity, sim_threshold=0.0, resolution=1.0, seed=0):
    edge_graph = nx.Graph()
    edge_graph.add_nodes_from(range(len(edges)))
    for (i, j), sim in similarity.items():
        if sim > sim_threshold:
            edge_graph.add_edge(i, j, weight=sim)
    communities = louvain_communities(edge_graph, weight="weight", resolution=resolution, seed=seed)
    labels = np.zeros(len(edges), dtype=int)
    for cluster_id, community in enumerate(communities):
        for edge_idx in community:
            labels[edge_idx] = cluster_id
    return labels


def precompute_direction_pairs(edges, similarity, edge_labels, node_idx):
    """
    node_idxを受け取り、m_idx_arr/u_idx_arr(各エッジのmovie側/user側ノードの整数インデックス)を
    ここで1回だけ計算しておく。direction_alignment_stress_and_gradは最適化の反復のたびに
    呼ばれるため、以前のように毎回node_idx[n]の辞書引きでこれを作り直すのは無駄だった。
    """
    edge_m_idx, edge_u_idx = [], []
    for u, v in edges:
        m_node, u_node = (u, v) if u.startswith("m_") else (v, u)
        edge_m_idx.append(m_node)
        edge_u_idx.append(u_node)
    pair_i, pair_j, pair_w = [], [], []
    for (i, j), sim in similarity.items():
        if sim > 0 and edge_labels[i] == edge_labels[j]:
            pair_i.append(i); pair_j.append(j); pair_w.append(sim)
    return {
        "edge_m_idx": edge_m_idx, "edge_u_idx": edge_u_idx,
        "m_idx_arr": np.array([node_idx[n] for n in edge_m_idx], dtype=np.int64),
        "u_idx_arr": np.array([node_idx[n] for n in edge_u_idx], dtype=np.int64),
        "pair_i": np.array(pair_i, dtype=np.int64),
        "pair_j": np.array(pair_j, dtype=np.int64),
        "pair_w": np.array(pair_w, dtype=np.float64),
    }


def _direction_alignment_numpy(coords_flat, N, m_idx_arr, u_idx_arr, pair_i, pair_j, pair_w):
    """direction_alignment_stress_and_gradのnumpyベクトル化実装(numba不使用時のフォールバック)。"""
    coords = coords_flat.reshape(N, 2)
    m_coords, u_coords = coords[m_idx_arr], coords[u_idx_arr]
    diff = u_coords - m_coords
    length = np.linalg.norm(diff, axis=1)
    length_safe = np.where(length < 1e-9, 1e-9, length)
    directions = diff / length_safe[:, None]

    if len(pair_i) == 0:
        return 0.0, np.zeros((N, 2)).flatten()

    d_i, d_j = directions[pair_i], directions[pair_j]
    cos_sim = np.sum(d_i * d_j, axis=1)
    loss = float(np.sum(pair_w * (1 - cos_sim) / 2))

    grad_directions = np.zeros_like(directions)
    coef = -pair_w / 2
    np.add.at(grad_directions, pair_i, coef[:, None] * d_j)
    np.add.at(grad_directions, pair_j, coef[:, None] * d_i)
    dot = np.sum(directions * grad_directions, axis=1)
    grad_diff = (grad_directions - directions * dot[:, None]) / length_safe[:, None]

    grad_coords = np.zeros((N, 2))
    np.add.at(grad_coords, u_idx_arr, grad_diff)
    np.add.at(grad_coords, m_idx_arr, -grad_diff)
    return loss, grad_coords.flatten()


def _direction_alignment_loop(coords_flat, N, m_idx_arr, u_idx_arr, pair_i, pair_j, pair_w):
    """
    direction_alignment_stress_and_gradの明示ループ実装。numbaはnp.add.at(スキャッター加算)を
    サポートしないため、各エッジ・各ペアについて明示的にループしながらgrad_dir_x/y、
    grad_coordsへ加算する形に書き換えている。_direction_alignment_numpyと数値的に等価。
    """
    coords = coords_flat.reshape(N, 2)
    n_edges = m_idx_arr.shape[0]

    dir_x = np.zeros(n_edges)
    dir_y = np.zeros(n_edges)
    length_arr = np.zeros(n_edges)
    for k in range(n_edges):
        mi, ui = m_idx_arr[k], u_idx_arr[k]
        dx = coords[ui, 0] - coords[mi, 0]
        dy = coords[ui, 1] - coords[mi, 1]
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            length = 1e-9
        length_arr[k] = length
        dir_x[k] = dx / length
        dir_y[k] = dy / length

    n_pairs = pair_i.shape[0]
    loss = 0.0
    grad_dir_x = np.zeros(n_edges)
    grad_dir_y = np.zeros(n_edges)

    for k in range(n_pairs):
        i = pair_i[k]
        j = pair_j[k]
        w = pair_w[k]
        cos_sim = dir_x[i] * dir_x[j] + dir_y[i] * dir_y[j]
        loss += w * (1.0 - cos_sim) / 2.0
        coef = -w / 2.0
        grad_dir_x[i] += coef * dir_x[j]
        grad_dir_y[i] += coef * dir_y[j]
        grad_dir_x[j] += coef * dir_x[i]
        grad_dir_y[j] += coef * dir_y[i]

    grad_coords = np.zeros((N, 2))
    for k in range(n_edges):
        mi, ui = m_idx_arr[k], u_idx_arr[k]
        dot = dir_x[k] * grad_dir_x[k] + dir_y[k] * grad_dir_y[k]
        grad_diff_x = (grad_dir_x[k] - dir_x[k] * dot) / length_arr[k]
        grad_diff_y = (grad_dir_y[k] - dir_y[k] * dot) / length_arr[k]
        grad_coords[ui, 0] += grad_diff_x
        grad_coords[ui, 1] += grad_diff_y
        grad_coords[mi, 0] -= grad_diff_x
        grad_coords[mi, 1] -= grad_diff_y

    return loss, grad_coords.flatten()


if HAS_NUMBA:
    _direction_alignment_core = numba.njit(cache=True, fastmath=True)(_direction_alignment_loop)
else:
    _direction_alignment_core = _direction_alignment_numpy


def direction_alignment_stress_and_grad(coords_flat, nodes, node_idx, precomputed):
    """
    公開API(シグネチャは従来通り)。実体はprecomputedに入っているm_idx_arr/u_idx_arr等の
    numpy配列だけを使う_direction_alignment_core(numba版 or numpyフォールバック版)に委譲する。
    """
    N = len(nodes)
    return _direction_alignment_core(
        coords_flat, N,
        precomputed["m_idx_arr"], precomputed["u_idx_arr"],
        precomputed["pair_i"], precomputed["pair_j"], precomputed["pair_w"]
    )


def calc_direction_alignment_score(coords, nodes, node_idx, precomputed):
    loss, _ = direction_alignment_stress_and_grad(coords.flatten(), nodes, node_idx, precomputed)
    total_w = precomputed["pair_w"].sum()
    return float("nan") if total_w == 0 else 1.0 - (loss / total_w)
