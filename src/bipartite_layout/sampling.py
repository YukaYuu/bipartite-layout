"""Subgraph sampling from the full MovieLens/DBLP graph."""

import numpy as np
from networkx.algorithms.community import louvain_communities

from bipartite_layout.config import N_FOCUS_USERS_PER_HUB, N_HUB_MOVIES, N_MOVIES_PER_FOCUS_USER


def select_hub_movies_by_community(subgraph, n_hub_movies):
    communities = louvain_communities(subgraph, weight=None, seed=0)
    communities_sorted = sorted(communities, key=len, reverse=True)

    print(f"\n候補プール内でのコミュニティ検出: {len(communities_sorted)}個のコミュニティ "
          f"(サイズ内訳: {[len(c) for c in communities_sorted]})")

    hub_movies = []
    for i, community in enumerate(communities_sorted):
        if len(hub_movies) >= n_hub_movies:
            break
        movie_candidates = [n for n in community if n.startswith("m_")]
        if not movie_candidates:
            continue
        top_movie = max(movie_candidates, key=lambda n: subgraph.degree(n))
        hub_movies.append(top_movie)
        print(f"  コミュニティ{i}(サイズ{len(community)})から {top_movie} をハブに選出")

    if len(hub_movies) < n_hub_movies:
        sub_movie_nodes = [n for n in subgraph.nodes() if n.startswith("m_")]
        remaining = sorted(
            [n for n in sub_movie_nodes if n not in hub_movies],
            key=lambda n: subgraph.degree(n), reverse=True
        )
        n_missing = n_hub_movies - len(hub_movies)
        fallback = remaining[:n_missing]
        if fallback:
            print(f"  コミュニティ数が不足していたため、次数上位の映画を{len(fallback)}本補充: {fallback}")
        hub_movies.extend(fallback)

    return hub_movies


def build_small_subgraph(M, n_seed_movies=5, n_users_per_movie=20,
                          n_movies_per_user=5, n_hub_movies=N_HUB_MOVIES,
                          n_focus_users_per_hub=N_FOCUS_USERS_PER_HUB,
                          n_movies_per_focus_user=N_MOVIES_PER_FOCUS_USER):
    movie_nodes = [n for n in M.nodes() if n.startswith("m_")]
    movie_degrees = sorted(movie_nodes, key=lambda n: M.degree(n), reverse=True)

    seed_movies = movie_degrees[:n_seed_movies]
    subgraph_nodes = set(seed_movies)

    for movie in seed_movies:
        users = list(M.neighbors(movie))
        subgraph_nodes.update(users[:n_users_per_movie])
        for user in users[:n_users_per_movie]:
            other_movies = list(M.neighbors(user))
            subgraph_nodes.update(other_movies[:n_movies_per_user])

    subgraph = M.subgraph(subgraph_nodes)

    hub_movies = select_hub_movies_by_community(subgraph, n_hub_movies)

    small_nodes = set(hub_movies)
    used_users = set()  # 同じuserが複数ハブに重複して数えられるのを防ぐ
    for hub in hub_movies:
        candidates = [u for u in subgraph.neighbors(hub) if u not in used_users]
        hub_users = candidates[:n_focus_users_per_hub]
        used_users.update(hub_users)
        small_nodes.update(hub_users)
        for user in hub_users:
            movies = list(subgraph.neighbors(user))[:n_movies_per_focus_user]
            small_nodes.update(movies)

    return subgraph.subgraph(small_nodes).copy()


def apply_top_k_sparsification(common_deg, is_user, top_k, mutual_only=False):
    sparsified = np.zeros_like(common_deg)

    for type_mask in [is_user, ~is_user]:
        idxs = np.where(type_mask)[0]
        if len(idxs) <= 1:
            continue
        sub = common_deg[np.ix_(idxs, idxs)]
        n = len(idxs)
        k = min(top_k, n - 1)
        if k <= 0:
            continue

        topk_mask = np.zeros_like(sub, dtype=bool)
        for row in range(n):
            row_vals = sub[row]
            if np.all(row_vals == 0):
                continue
            top_idx = np.argsort(row_vals)[-k:]
            topk_mask[row, top_idx] = True

        keep_mask = (topk_mask & topk_mask.T) if mutual_only else (topk_mask | topk_mask.T)
        sub_sparsified = np.where(keep_mask, sub, 0.0)
        sparsified[np.ix_(idxs, idxs)] = sub_sparsified

    return sparsified
