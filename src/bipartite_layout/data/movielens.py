"""MovieLens data loading."""

import networkx as nx


def load_movielens_graph(path):
    """MovieLens形式のデータからユーザ-映画の二部グラフを構築する"""
    edges = []
    with open(path, "r") as f:
        for line in f:
            nums = list(map(int, line.split()))
            user_id = nums[0]
            movie_ids = nums[1:]
            for movie_id in movie_ids:
                edges.append((f"u_{user_id}", f"m_{movie_id}"))

    G = nx.Graph()
    G.add_edges_from(edges)
    return G
