"""
手法A: 方向整列のみ(alpha混合なし)
手法B: alpha混合のみ(方向整列なし)
手法C: alpha混合 + 方向整列(組み合わせ)
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN, KMeans
from collections import Counter
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import normalized_mutual_info_score
from lxml import etree

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = "/Users/owner/Downloads/filtered-ml-1m/train.txt"
DBLP_PATH = "/Users/owner/Downloads/dblp.xml"
DBLP_DTD_PATH = "/Users/owner/Downloads/dblp.dtd"
DBLP_MAX_PAPERS = 300_000
# load_dblp_graphの結果をキャッシュするpickleファイル。3.8GBのXMLを毎回パースし直すのは
# 無駄なので、DBLP_PATH/DBLP_MAX_PAPERSを変えない限り2回目以降はここから読み込む。
DBLP_CACHE_PATH = "/Users/owner/Downloads/dblp_graph_cache.pkl"

THRESHOLD_COMMON_DEG = 0.25
TOP_K_SAME_TYPE = 5
MUTUAL_TOP_K_ONLY = False

DBSCAN_MIN_SAMPLES = 2
DBSCAN_EPS_SCALE = 1.5

N_HUB_MOVIES = 3
N_FOCUS_USERS_PER_HUB = 4
N_MOVIES_PER_FOCUS_USER = 3

# --- 大規模サンプリング用パラメータ(userを大幅に増やし、movieと揃える実験用) ---
LARGE_N_SEED_MOVIES = 15
LARGE_N_USERS_PER_MOVIE = 60
LARGE_N_MOVIES_PER_USER = 8
LARGE_N_HUB_MOVIES = 15
LARGE_N_FOCUS_USERS_PER_HUB = 5
LARGE_N_MOVIES_PER_FOCUS_USER = 4

SMALL_BUILD_KWARGS = dict(
    n_hub_movies=N_HUB_MOVIES,
    n_focus_users_per_hub=N_FOCUS_USERS_PER_HUB,
    n_movies_per_focus_user=N_MOVIES_PER_FOCUS_USER,
)

LARGE_BUILD_KWARGS = dict(
    n_seed_movies=LARGE_N_SEED_MOVIES,
    n_users_per_movie=LARGE_N_USERS_PER_MOVIE,
    n_movies_per_user=LARGE_N_MOVIES_PER_USER,
    n_hub_movies=LARGE_N_HUB_MOVIES,
    n_focus_users_per_hub=LARGE_N_FOCUS_USERS_PER_HUB,
    n_movies_per_focus_user=LARGE_N_MOVIES_PER_FOCUS_USER,
)

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

class _FixedPathResolver(etree.Resolver):
    """DOCTYPEが参照する外部DTDを、実際のファイル名/場所によらず固定パスから読み込ませる。"""
    def __init__(self, dtd_path):
        self._dtd_path = dtd_path

    def resolve(self, system_url, public_id, context):
        return self.resolve_filename(self._dtd_path, context)


def load_dblp_graph(path, max_papers=None, dtd_path=None):
    """
    DBLPのXMLダンプ(article/inproceedings)からauthor-paperの二部グラフを構築する。

    - tag=paper_typesをiterparseに渡すことで、author/title/year等の子要素ごとに
      Python側でno-op判定していた分の無駄なイベント発火・ループ回数を減らす
      (1論文あたり5〜8個ある子要素の分だけ、以前は無駄にPythonループが回っていた)。
    - dtd_pathを指定すると、dblp.xmlと同じディレクトリにdblp.dtdが無い/別名の場合でも、
      指定したファイルをDTDとして読み込むカスタムResolverを登録する。
    - max_papersを指定すると、その件数に達した時点で打ち切る。build_small_subgraphは
      結局数百ノードしかサンプリングしないため、全件読み込みは通常不要。
    - エッジはリストにためて一定件数ごとにG.add_edges_fromでまとめて追加する
      (G.add_edgeを1件ずつ呼ぶより関数呼び出しオーバーヘッドが少ない)。
    """
    G = nx.Graph()
    paper_types = ("article", "inproceedings")  # 必要に応じてbook, incollection等も追加可能

    context = etree.iterparse(
        path, events=("end",), tag=paper_types,
        load_dtd=True, resolve_entities=True, no_network=True
    )
    if dtd_path is not None:
        context.resolvers.add(_FixedPathResolver(dtd_path))

    n_papers = 0
    edges = []
    EDGE_BATCH_SIZE = 50_000

    for event, elem in context:
        key = elem.get("key")
        authors = [a.text for a in elem.findall("author") if a.text]
        if key and authors:
            paper_node = f"m_{key}"
            for author in authors:
                edges.append((f"u_{author}", paper_node))
            n_papers += 1
            if n_papers % 100000 == 0:
                print(f"{n_papers}件の論文を処理しました...")
            if len(edges) >= EDGE_BATCH_SIZE:
                G.add_edges_from(edges)
                edges.clear()

        elem.clear()
        # 親要素に残る参照もクリアして、メモリリークを防ぐ
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if max_papers is not None and n_papers >= max_papers:
            break

    if edges:
        G.add_edges_from(edges)

    print(f"完了: {n_papers}件の論文, {G.number_of_nodes()}ノード, {G.number_of_edges()}エッジ")
    return G


def load_dblp_graph_cached(path=DBLP_PATH, max_papers=DBLP_MAX_PAPERS, dtd_path=DBLP_DTD_PATH,
                            cache_path=DBLP_CACHE_PATH):
    """
    load_dblp_graphの結果をpickleでキャッシュするラッパー。3.8GBのXMLをiterparseで
    パースするのはmax_papers=300,000でも数分かかるが、build_small_subgraphが最終的に
    使うのはそこから抽出したごく小さいサブグラフだけなので、同じ(path, max_papers)の
    組み合わせで何度も実験し直す際は2回目以降キャッシュから読み込めば十分。
    max_papersなどを変えたい場合はcache_pathのファイルを削除するか、別のcache_pathを
    指定してください。cache_path=Noneでキャッシュを無効化できる。
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"DBLPグラフのキャッシュを読み込み中: {cache_path}")
        with open(cache_path, "rb") as f:
            G = pickle.load(f)
        print(f"読み込み完了: {G.number_of_nodes()}ノード, {G.number_of_edges()}エッジ")
        return G

    G = load_dblp_graph(path, max_papers=max_papers, dtd_path=dtd_path)

    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"DBLPグラフをキャッシュに保存しました: {cache_path}")

    return G


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


# ---------------------------------------------------------------
# 計算のメモ化: 同じ(M, kwargs)やGに対してbuild_small_subgraph/build_matrices/
# calc_edge_similarity+cluster_edges_by_community+precompute_direction_pairsが
# 複数のmain_*関数から重複して呼ばれていたのを避けるためのキャッシュ層。
# 1回のプロセス実行内(=1回のrun_all_experiments_for_dataset呼び出し)でのみ有効。
# ---------------------------------------------------------------

_small_subgraph_cache = {}
_matrices_cache = {}
_edge_direction_cache = {}


def get_small_subgraph_cached(M, **kwargs):
    """build_small_subgraph(M, **kwargs)をメモ化する(同じM・同じkwargsなら再計算しない)。"""
    key = (id(M), tuple(sorted(kwargs.items())))
    if key not in _small_subgraph_cache:
        _small_subgraph_cache[key] = build_small_subgraph(M, **kwargs)
    return _small_subgraph_cache[key]


def get_matrices_cached(G, weight_mode, threshold_common_deg, top_k_same_type,
                         mutual_top_k_only=False, uniform_weight_value=1.0):
    """
    build_matrices/build_matrices_uniform_weight/build_matrices_commonality_weightを
    メモ化する(同じG・同じweight_mode・同じパラメータなら再計算しない)。
    """
    key = (id(G), weight_mode, threshold_common_deg, top_k_same_type,
           mutual_top_k_only, uniform_weight_value)
    if key not in _matrices_cache:
        if weight_mode == "uniform":
            _matrices_cache[key] = build_matrices_uniform_weight(
                G, threshold_common_deg=threshold_common_deg, top_k_same_type=top_k_same_type,
                mutual_top_k_only=mutual_top_k_only, uniform_weight_value=uniform_weight_value
            )
        elif weight_mode == "degree":
            _matrices_cache[key] = build_matrices(
                G, threshold_common_deg=threshold_common_deg, top_k_same_type=top_k_same_type,
                mutual_top_k_only=mutual_top_k_only
            )
        elif weight_mode == "commonality":
            _matrices_cache[key] = build_matrices_commonality_weight(
                G, threshold_common_deg=threshold_common_deg, top_k_same_type=top_k_same_type,
                mutual_top_k_only=mutual_top_k_only
            )
        else:
            raise ValueError(
                f"unknown weight_mode: {weight_mode!r} (must be 'uniform', 'degree', or 'commonality')"
            )
    return _matrices_cache[key]


def get_edge_direction_cached(G, node_idx):
    """
    calc_edge_similarity + cluster_edges_by_community + precompute_direction_pairsの
    結果をGごとにメモ化する。戻り値: (edges, edge_labels, direction_precomputed)。
    """
    key = id(G)
    if key not in _edge_direction_cache:
        edges, similarity = calc_edge_similarity(G)
        edge_labels = cluster_edges_by_community(edges, similarity, sim_threshold=0.0, resolution=1.0)
        direction_precomputed = precompute_direction_pairs(edges, similarity, edge_labels, node_idx)
        _edge_direction_cache[key] = (edges, edge_labels, direction_precomputed)
    return _edge_direction_cache[key]

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


def _multi_seed_worker(args):
    """compute_layout_multi_seedの1seed分の計算(ProcessPoolExecutorに渡すため関数化)。"""
    common_deg, weight, alpha, nodes, is_user, seed = args
    coords, final_stress, converged, n_iter, grad_norm = compute_layout_method(
        "B", common_deg, weight, alpha, nodes, None, is_user, None, seed=seed, gamma=0.0
    )
    centroid_sep, nn_ratio = compute_separation_metrics(coords, is_user)
    n_clusters_user, noise_user = compute_cluster_metrics(coords, is_user)
    n_clusters_movie, noise_movie = compute_cluster_metrics(coords, ~is_user)
    return (coords, final_stress, centroid_sep, nn_ratio, n_clusters_user, n_clusters_movie,
            noise_user, noise_movie, converged, n_iter, grad_norm)


def compute_layout_multi_seed(common_deg, weight, alpha, nodes, is_user, n_seeds=8, n_workers=None):
    """
    複数のseedでレイアウトを計算し、局所解による結果のばらつきを確認する
    (alpha混合のみ = compute_layout_method("B", gamma=0.0)を複数seedで実行)。

    n_workersを指定すると(例: n_workers=4)、seedごとの計算をProcessPoolExecutorで
    並列実行する。デフォルトNoneは従来通り逐次実行(結果は完全に同一、速度のみ異なる)。
    """
    tasks = [(common_deg, weight, alpha, nodes, is_user, seed) for seed in range(n_seeds)]
    if n_workers:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_multi_seed_worker, tasks))
    else:
        results = [_multi_seed_worker(t) for t in tasks]

    best_coords, best_stress = None, np.inf
    centroid_seps, nn_ratios = [], []
    n_clusters_user_list, n_clusters_movie_list = [], []
    noise_ratio_user_list, noise_ratio_movie_list = [], []
    converged_list, n_iter_list, grad_norm_list = [], [], []

    for (coords, final_stress, centroid_sep, nn_ratio, n_clusters_user, n_clusters_movie,
         noise_user, noise_movie, converged, n_iter, grad_norm) in results:
        centroid_seps.append(centroid_sep)
        nn_ratios.append(nn_ratio)
        n_clusters_user_list.append(n_clusters_user)
        n_clusters_movie_list.append(n_clusters_movie)
        noise_ratio_user_list.append(noise_user)
        noise_ratio_movie_list.append(noise_movie)
        converged_list.append(converged)
        n_iter_list.append(n_iter)
        grad_norm_list.append(grad_norm)

        if final_stress < best_stress:
            best_stress = final_stress
            best_coords = coords

    return (best_coords,
            np.array(centroid_seps), np.array(nn_ratios),
            np.array(n_clusters_user_list, dtype=float),
            np.array(n_clusters_movie_list, dtype=float),
            np.array(noise_ratio_user_list, dtype=float),
            np.array(noise_ratio_movie_list, dtype=float),
            np.array(converged_list, dtype=bool),
            np.array(n_iter_list, dtype=float),
            np.array(grad_norm_list, dtype=float))

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


def compute_cluster_metrics(coords, mask, min_samples=DBSCAN_MIN_SAMPLES, eps_scale=DBSCAN_EPS_SCALE):
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

def run_full_experiment(M, build_kwargs, label, n_seeds=20,
                         threshold=THRESHOLD_COMMON_DEG, run_heavy_diagnostics=True):
    G = get_small_subgraph_cached(M, **build_kwargs)
    n_user_nodes = sum(1 for n in G.nodes() if n.startswith("u_"))
    n_movie_nodes = sum(1 for n in G.nodes() if n.startswith("m_"))
    print(f"\n{'=' * 20} [{label}] サブグラフ実験開始 {'=' * 20}")
    print(f"サブグラフ(固定): {G.number_of_nodes()} ノード "
          f"(user={n_user_nodes}, movie={n_movie_nodes}), {G.number_of_edges()} エッジ")
    print(f"サンプリングパラメータ: {build_kwargs}")

    if run_heavy_diagnostics:
        analyze_common_neighbor_popularity_bias(
            G, M, save_path=f"popularity_bias_{label}.png"
        )
        analyze_degree_and_null_model(G, save_path=f"degree_null_model_{label}.png")

        cv_stats = compute_jaccard_distribution_stats(G)
        print(f"\n[user-user] Jaccard分布: 平均={cv_stats['user_mean']:.3f}, "
              f"標準偏差={cv_stats['user_std']:.3f}, CV={cv_stats['user_cv']:.3f}")
        print(f"[movie-movie] Jaccard分布: 平均={cv_stats['movie_mean']:.3f}, "
              f"標準偏差={cv_stats['movie_std']:.3f}, CV={cv_stats['movie_cv']:.3f}")

        threshold_candidates = [0.0, 0.1, 0.167, 0.25, 0.375]
        print(f"\n{'threshold':>10} | {'user modularity':>16} | {'movie modularity':>17}")
        print("-" * 50)
        for th in threshold_candidates:
            nodes, idx, common_deg, weight, is_user = build_matrices(G, threshold_common_deg=th)
            mod_user, mod_movie = analyze_virtual_edge_graph_structure_quiet(
                nodes, common_deg, is_user, th
            )
            print(f"{th:>10.3f} | {mod_user:>16.3f} | {mod_movie:>17.3f}")

        values = inspect_common_deg_distribution(
            G, save_path=f"common_deg_distribution_{label}.png"
        )
        if len(values):
            candidates = sorted(set(
                list(np.round(np.percentile(values, [0, 25, 50, 75, 90, 100]), 3))
            ))
            print("\n閾値候補ごとの比較 (分位点ベース、user/movieプール):")
            compare_threshold_candidates(G, candidates)

        inspect_common_deg_distribution_by_type(
            G, save_path=f"common_deg_distribution_by_type_{label}.png"
        )

    if threshold is None:
        print("\nTHRESHOLD_COMMON_DEG が未設定です。閾値を決めてから再実行してください。")
        return

    nodes, idx, common_deg, weight, is_user = get_matrices_cached(
        G, "degree", threshold, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY)

    if run_heavy_diagnostics:
        analyze_virtual_edge_graph_structure(nodes, common_deg, is_user, threshold)

    fine_range = np.round(np.arange(0.25, 0.76, 0.1), 2)
    alphas = sorted(set([0.0, 1.0] + list(fine_range)))

    fig, axes = plt.subplots(1, len(alphas), figsize=(4.0 * len(alphas), 4.0))

    print(f"\n[{label}] 各alphaにつき {n_seeds} 個のseedで実行し、ばらつきを確認します。")
    header = (f"{'alpha':>6} | {'centroid_sep':>16} | {'nn_ratio':>16} | "
              f"{'n_cluster(user)':>16} | {'n_cluster(movie)':>17}")
    print(f"\n{header}")
    print("-" * len(header))

    sep_means, sep_stds = [], []
    nn_means, nn_stds = [], []
    ncl_u_means, ncl_u_stds = [], []
    ncl_m_means, ncl_m_stds = [], []

    for ax, alpha in zip(axes, alphas):
        (best_coords, centroid_seps, nn_ratios,
         n_clusters_user, n_clusters_movie,
         noise_ratio_user, noise_ratio_movie,
         converged_arr, n_iter_arr, grad_norm_arr) = compute_layout_multi_seed(
            common_deg, weight, alpha, nodes, is_user, n_seeds=n_seeds
        )

        sep_mean, sep_std = centroid_seps.mean(), centroid_seps.std()
        nn_mean, nn_std = nn_ratios.mean(), nn_ratios.std()
        ncl_u_mean, ncl_u_std = np.nanmean(n_clusters_user), np.nanstd(n_clusters_user)
        ncl_m_mean, ncl_m_std = np.nanmean(n_clusters_movie), np.nanstd(n_clusters_movie)

        sep_means.append(sep_mean); sep_stds.append(sep_std)
        nn_means.append(nn_mean); nn_stds.append(nn_std)
        ncl_u_means.append(ncl_u_mean); ncl_u_stds.append(ncl_u_std)
        ncl_m_means.append(ncl_m_mean); ncl_m_stds.append(ncl_m_std)

        print(f"{alpha:>6.2f} | {sep_mean:>7.3f}±{sep_std:<7.3f} | "
              f"{nn_mean:>7.3f}±{nn_std:<7.3f} | "
              f"{ncl_u_mean:>7.2f}±{ncl_u_std:<7.2f} | "
              f"{ncl_m_mean:>7.2f}±{ncl_m_std:<7.2f}")

        n_converged = int(converged_arr.sum())
        mean_iter = n_iter_arr.mean()
        mean_grad_norm = grad_norm_arr.mean()
        print(f"        └ 収束: {n_converged}/{n_seeds}回成功, "
              f"平均反復回数={mean_iter:.1f}, 平均勾配ノルム={mean_grad_norm:.4f}")

        ax.scatter(best_coords[is_user, 0], best_coords[is_user, 1], c="tab:blue", label="user", s=60)
        ax.scatter(best_coords[~is_user, 0], best_coords[~is_user, 1], c="tab:red", label="movie", s=60)
        for i, j in G.edges():
            xi, xj = best_coords[idx[i]], best_coords[idx[j]]
            ax.plot([xi[0], xj[0]], [xi[1], xj[1]], color="gray", alpha=0.3, linewidth=0.7)
        ax.set_title(f"alpha={alpha}\nsep={sep_mean:.2f} ncl(u,m)=({ncl_u_mean:.1f},{ncl_m_mean:.1f})",
                     fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(f"[{label}] threshold = {threshold}, "
                 f"n_user={n_user_nodes}, n_movie={n_movie_nodes}")
    plt.tight_layout()
    out_path = f"alpha_layout_th{threshold}_{label}.png"
    plt.savefig(out_path, dpi=150)
    print(f"保存しました: {out_path}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    ax1, ax2, ax3 = axes2

    ax1.errorbar(alphas, sep_means, yerr=sep_stds, marker="o", capsize=4)
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("centroid separation (normalized)")
    ax1.set_title(f"[{label}] Centroid separation vs alpha ({n_seeds} seeds)")

    ax2.errorbar(alphas, nn_means, yerr=nn_stds, marker="o", color="tab:orange", capsize=4)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("alpha")
    ax2.set_ylabel("nn_ratio (opposite / same)")
    ax2.set_title(f"[{label}] Nearest-neighbor ratio vs alpha ({n_seeds} seeds)")

    ax3.errorbar(alphas, ncl_u_means, yerr=ncl_u_stds, marker="o", color="tab:blue",
                 capsize=4, label="user")
    ax3.errorbar(alphas, ncl_m_means, yerr=ncl_m_stds, marker="s", color="tab:red",
                 capsize=4, label="movie")
    ax3.set_xlabel("alpha")
    ax3.set_ylabel("DBSCAN cluster count")
    ax3.set_title(f"[{label}] Cluster count vs alpha ({n_seeds} seeds)")
    ax3.legend()

    plt.tight_layout()
    trend_path = f"separation_and_cluster_trend_th{threshold}_{label}.png"
    plt.savefig(trend_path, dpi=150)
    print(f"保存しました: {trend_path}")

    return {
        "label": label, "n_user": n_user_nodes, "n_movie": n_movie_nodes,
        "alphas": alphas, "sep_means": sep_means, "nn_means": nn_means,
        "ncl_u_means": ncl_u_means, "ncl_m_means": ncl_m_means,
    }

def gamma_sweep(common_deg, weight, nodes, node_idx, is_user, direction_precomputed,
                alpha=0.5, gammas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0), n_seeds=10):
    """STEP1: gammaを振って、収束が安定する値を探す(手法C固定)。"""
    print(f"\n=== gammaスイープ(手法C、alpha={alpha}固定) ===")
    print(f"{'gamma':>8} | {'dir_score':>10} | {'nn_ratio':>10} | {'converged':>10}")
    print("-" * 50)
    for gamma in gammas:
        dir_scores, nn_ratios, n_converged = [], [], 0
        for seed in range(n_seeds):
            coords, _, converged, _, _ = compute_layout_method(
                "C", common_deg, weight, alpha, nodes, node_idx, is_user,
                direction_precomputed, seed=seed, gamma=gamma
            )
            dir_scores.append(calc_direction_alignment_score(coords, nodes, node_idx, direction_precomputed))
            nn_ratios.append(compute_separation_metrics(coords, is_user)[1])
            n_converged += int(converged)
        print(f"{gamma:>8.2f} | {np.nanmean(dir_scores):>10.3f} | {np.nanmean(nn_ratios):>10.3f} | "
              f"{n_converged:>7d}/{n_seeds}")
    print("\n見方: 収束率が高く保たれつつ、dir_scoreがgamma=0(手法B相当)より明確に高い、"
          "一番小さいgammaを選ぶのが良い候補です。")


def run_three_method_comparison(G, common_deg, weight, nodes, node_idx, is_user,
                                 alphas, n_seeds=10, gamma=0.1):
    edges, similarity = calc_edge_similarity(G)
    edge_labels = cluster_edges_by_community(edges, similarity, sim_threshold=0.0, resolution=1.0)
    direction_precomputed = precompute_direction_pairs(edges, similarity, edge_labels, node_idx)
    print(f"\nエッジクラスタ数: {edge_labels.max() + 1}, 方向整列の対象ペア数: {len(direction_precomputed['pair_i'])}")

    def run_multi_seed(method, alpha):
        nn_ratios, ncl_u_list, ncl_m_list, dir_scores, n_converged = [], [], [], [], 0
        for seed in range(n_seeds):
            coords, _, converged, _, _ = compute_layout_method(
                method, common_deg, weight, alpha, nodes, node_idx, is_user,
                direction_precomputed, seed=seed, gamma=gamma
            )
            nn_ratios.append(compute_separation_metrics(coords, is_user)[1])
            ncl_u_list.append(compute_cluster_metrics(coords, is_user)[0])
            ncl_m_list.append(compute_cluster_metrics(coords, ~is_user)[0])
            dir_scores.append(calc_direction_alignment_score(coords, nodes, node_idx, direction_precomputed))
            n_converged += int(converged)
        return {"nn_ratio": np.nanmean(nn_ratios), "ncl_u": np.nanmean(ncl_u_list),
                "ncl_m": np.nanmean(ncl_m_list), "dir_score": np.nanmean(dir_scores), "converged": n_converged}

    print(f"\n{'method':>8} | {'alpha':>6} | {'nn_ratio':>10} | {'ncl(u)':>8} | {'ncl(m)':>8} | "
          f"{'dir_score':>10} | {'converged':>10}")
    print("-" * 80)

    res_a = run_multi_seed("A", alpha=0.5)
    print(f"{'A':>8} | {'-':>6} | {res_a['nn_ratio']:>10.3f} | {res_a['ncl_u']:>8.2f} | "
          f"{res_a['ncl_m']:>8.2f} | {res_a['dir_score']:>10.3f} | {res_a['converged']:>7d}/{n_seeds}")

    for alpha in alphas:
        res_b = run_multi_seed("B", alpha)
        res_c = run_multi_seed("C", alpha)
        print(f"{'B':>8} | {alpha:>6.2f} | {res_b['nn_ratio']:>10.3f} | {res_b['ncl_u']:>8.2f} | "
              f"{res_b['ncl_m']:>8.2f} | {res_b['dir_score']:>10.3f} | {res_b['converged']:>7d}/{n_seeds}")
        print(f"{'C':>8} | {alpha:>6.2f} | {res_c['nn_ratio']:>10.3f} | {res_c['ncl_u']:>8.2f} | "
              f"{res_c['ncl_m']:>8.2f} | {res_c['dir_score']:>10.3f} | {res_c['converged']:>7d}/{n_seeds}")


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


def plot_and_save(coords, is_user, G, node_idx, title, filename, edge_labels=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[is_user, 0], coords[is_user, 1], c="black", marker="s", label="user", s=40, zorder=3)
    ax.scatter(coords[~is_user, 0], coords[~is_user, 1], c="black", marker="o", label="movie", s=40, zorder=3)

    edges = list(G.edges())
    if edge_labels is not None:
        n_clusters = int(edge_labels.max()) + 1
        cmap = plt.colormaps["tab10"].resampled(max(n_clusters, 1))
        edge_colors = [cmap(edge_labels[k]) for k in range(len(edges))]
    else:
        edge_colors = ["gray"] * len(edges)

    for k, (u, v) in enumerate(edges):
        i, j = node_idx[u], node_idx[v]
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color=edge_colors[k], alpha=0.7, linewidth=1.2, zorder=1)

    ax.set_title(title, fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"保存しました: {filename}")


def plot_alpha_grid(G, common_deg, weight, nodes, node_idx, is_user,
                     alphas, gamma=0.1, seed=0, filename="alpha_grid.png",
                     methods=("B", "C"), anchor_weight=1.0):
    """
    行=methods(既定ではB, C。methods=("B","C","D")のようにDも含められる)、
    列=alphaの値、というグリッドでレイアウトを一覧表示する。
    """
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    n_alphas = len(alphas)
    max_cols_per_row = 7  # 1行あたり最大7列(alpha)に折り返す
    n_chunks = int(np.ceil(n_alphas / max_cols_per_row))
    n_rows = len(methods)

    for chunk_idx in range(n_chunks):
        chunk_alphas = alphas[chunk_idx * max_cols_per_row: (chunk_idx + 1) * max_cols_per_row]
        n_cols = len(chunk_alphas)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows))
        if n_cols == 1:
            axes = axes.reshape(n_rows, 1)
        if n_rows == 1:
            axes = axes.reshape(1, n_cols)

        for col, alpha in enumerate(chunk_alphas):
            method_results = [
                (label,) + compute_layout_method(
                    label, common_deg, weight, alpha, nodes, node_idx, is_user,
                    direction_precomputed, seed=seed, gamma=gamma, anchor_weight=anchor_weight
                )
                for label in methods
            ]

            for row, (label, coords, _, conv, _, _) in enumerate(method_results):
                ax = axes[row, col]
                ax.scatter(coords[is_user, 0], coords[is_user, 1], c="black", marker="s", s=10, zorder=2)
                ax.scatter(coords[~is_user, 0], coords[~is_user, 1], c="black", marker="o", s=10, zorder=2)
                for k, (u, v) in enumerate(edges):
                    i, j = node_idx[u], node_idx[v]
                    ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                            color=plt.colormaps["tab10"].resampled(edge_labels.max() + 1)(edge_labels[k]),
                            linewidth=0.6, alpha=0.6, zorder=1)
                # 行(B/C/D...)ごとに、そのサブプロット自身の収束状況を明示する。
                conv_mark = "" if conv else " (未収束)"
                if row == 0:
                    title = f"α={alpha:.2f} [{label}]{conv_mark}"
                else:
                    title = f"[{label}]{conv_mark}"
                ax.set_title(title, fontsize=9, color=("black" if conv else "red"))
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(label, fontsize=12, rotation=0, labelpad=20)

        plt.tight_layout()
        out_name = filename.replace(".png", f"_part{chunk_idx+1}.png")
        plt.savefig(out_name, dpi=130)
        plt.close(fig)
        print(f"保存しました: {out_name}（alpha: {[f'{a:.2f}' for a in chunk_alphas]}）")


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


def plot_ablation_grid(G, common_deg, weight, nodes, node_idx, is_user,
                        alphas, configs, method="B", gamma=0.1, seed=0,
                        filename="ablation_grid.png"):
    """
    plot_alpha_gridがmethod(B/C/D)間の比較用なのに対し、これはcompute_layout_methodへの
    追加キーワード引数(repel_same_type, random_initなど)違いによるconfig間比較用。
    行=configs(各要素は(label, kwargs)のタプル)、列=alphaの値。methodは1つに固定する。

    例:
      configs = [("通常(全ペア反発)", {}), ("同タイプ反発なし", {"repel_same_type": False})]
      configs = [("通常(alpha依存初期配置)", {}), ("完全ランダム初期配置", {"random_init": True})]
    """
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    n_alphas = len(alphas)
    max_cols_per_row = 7  # 1行あたり最大7列(alpha)に折り返す
    n_chunks = int(np.ceil(n_alphas / max_cols_per_row))
    n_rows = len(configs)

    for chunk_idx in range(n_chunks):
        chunk_alphas = alphas[chunk_idx * max_cols_per_row: (chunk_idx + 1) * max_cols_per_row]
        n_cols = len(chunk_alphas)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows))
        if n_cols == 1:
            axes = axes.reshape(n_rows, 1)
        if n_rows == 1:
            axes = axes.reshape(1, n_cols)

        for col, alpha in enumerate(chunk_alphas):
            for row, (label, kwargs) in enumerate(configs):
                coords, _, conv, _, _ = compute_layout_method(
                    method, common_deg, weight, alpha, nodes, node_idx, is_user,
                    direction_precomputed, seed=seed, gamma=gamma, **kwargs
                )
                ax = axes[row, col]
                ax.scatter(coords[is_user, 0], coords[is_user, 1], c="black", marker="s", s=10, zorder=2)
                ax.scatter(coords[~is_user, 0], coords[~is_user, 1], c="black", marker="o", s=10, zorder=2)
                for k, (u, v) in enumerate(edges):
                    i, j = node_idx[u], node_idx[v]
                    ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                            color=plt.colormaps["tab10"].resampled(edge_labels.max() + 1)(edge_labels[k]),
                            linewidth=0.6, alpha=0.6, zorder=1)
                conv_mark = "" if conv else " (未収束)"
                title = f"α={alpha:.2f}{conv_mark}" if row == 0 else conv_mark
                ax.set_title(title, fontsize=9, color=("black" if conv else "red"))
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=45, ha="right")

        plt.tight_layout()
        out_name = filename.replace(".png", f"_part{chunk_idx+1}.png")
        plt.savefig(out_name, dpi=130)
        plt.close(fig)
        print(f"保存しました: {out_name}（alpha: {[f'{a:.2f}' for a in chunk_alphas]}）")


def main_repulsion_and_init_ablation(M, dataset_label, alphas=None, gamma=0.1, seed=0, method="B"):
    """
    2つのablationを実証・比較する:
    (1) 同タイプノード間の反発を切ると、格子状に見える均一配置は消えるか
        (repel_same_type=False)
    (2) 初期配置をalphaに依存させず完全ランダムにしても、二層配置はストレス関数の
        alpha混合項だけから自然に立ち上がるか(random_init=True)
    画像グリッド(plot_ablation_grid)に加え、最近傍距離CV(compute_nn_distance_cv、
    小さいほど格子的)の数表も出力し、視覚的印象だけでなく数値でも比較できるようにする。
    """
    print(f"\n{'#' * 20} [{dataset_label}] 反発・初期配置のablation実験 {'#' * 20}")

    G = get_small_subgraph_cached(M)
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, "degree", THRESHOLD_COMMON_DEG, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY
    )
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    if alphas is None:
        alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)

    repulsion_configs = [
        ("通常(全ペア反発)", {}),
        ("同タイプ反発なし", {"repel_same_type": False}),
    ]
    init_configs = [
        ("通常(alpha依存初期配置)", {}),
        ("完全ランダム初期配置", {"random_init": True}),
    ]

    plot_ablation_grid(G, common_deg, weight, nodes, node_idx, is_user, alphas,
                        repulsion_configs, method=method, gamma=gamma, seed=seed,
                        filename=f"ablation_repulsion_{dataset_label}.png")
    plot_ablation_grid(G, common_deg, weight, nodes, node_idx, is_user, alphas,
                        init_configs, method=method, gamma=gamma, seed=seed,
                        filename=f"ablation_init_{dataset_label}.png")

    all_configs = [("repel_all", repulsion_configs[0]), ("repel_cross_only", repulsion_configs[1]),
                   ("init_default", init_configs[0]), ("init_random", init_configs[1])]
    print(f"\n--------------- [{dataset_label}] 最近傍距離CV(小さいほど格子的、大きいほど疎密がある) ---------------")
    print("alpha".ljust(8) + "".join(f"{name:>20}" for name, _ in all_configs))
    for alpha in alphas:
        row_vals = []
        for _, (_, kwargs) in all_configs:
            coords, _, conv, _, _ = compute_layout_method(
                method, common_deg, weight, alpha, nodes, node_idx, is_user,
                direction_precomputed, seed=seed, gamma=gamma, **kwargs
            )
            row_vals.append(compute_nn_distance_cv(coords))
        print(f"{alpha:<8.2f}" + "".join(f"{v:>20.4f}" for v in row_vals))


def main_weight_mode_image_comparison(M, dataset_label, weight_modes=("uniform", "degree", "commonality"),
                                       alphas=None, gamma=0.1, seed=0):
    """
    次数ベース(degree)・均一(uniform)・共通隣接度(commonality)、指定した全ての実エッジ重み方式で
    alpha_gridの画像を生成し、直接見比べられるようにする。同じG(build_small_subgraphの結果)に
    対して重み方式だけを変えるため、レイアウトの違いが純粋に重み方式の違いに起因すると言える。
    ファイル名にweight_modeを含めるため、他の実験の出力(alpha_grid_{label}_part*.png)とも
    衝突しない。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_weight_mode_image_comparison {'#' * 20}")

    G = get_small_subgraph_cached(M)
    if alphas is None:
        alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)

    for weight_mode in weight_modes:
        nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
            G, weight_mode, THRESHOLD_COMMON_DEG, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY
        )
        print(f"\n--- [{dataset_label}] weight_mode={weight_mode} の画像を生成 ---")
        plot_alpha_grid(G, common_deg, weight, nodes, node_idx, is_user, alphas,
                         gamma=gamma, seed=seed, filename=f"alpha_grid_{dataset_label}_{weight_mode}.png")


def run_three_method_comparison_with_plots(G, common_deg, weight, nodes, node_idx, is_user,
                                            alphas, gamma=0.1, plot_seed=0, out_dir="."):
    edges, similarity = calc_edge_similarity(G)
    edge_labels = cluster_edges_by_community(edges, similarity, sim_threshold=0.0, resolution=1.0)
    direction_precomputed = precompute_direction_pairs(edges, similarity, edge_labels, node_idx)

    # 手法A(alpha非依存、1回だけ)
    coords_a, _, conv_a, _, _ = compute_layout_method(
        "A", common_deg, weight, 0.5, nodes, node_idx, is_user,
        direction_precomputed, seed=plot_seed, gamma=gamma
    )
    dir_score_a = calc_direction_alignment_score(coords_a, nodes, node_idx, direction_precomputed)
    plot_and_save(coords_a, is_user, G, node_idx,
                  f"Method A (direction alignment only)\ndir_score={dir_score_a:.3f}, 収束={conv_a}",
                  f"{out_dir}/layout_A.png", edge_labels=edge_labels)

    for alpha in alphas:
        coords_b, _, conv_b, _, _ = compute_layout_method(
            "B", common_deg, weight, alpha, nodes, node_idx, is_user,
            direction_precomputed, seed=plot_seed, gamma=gamma
        )
        dir_score_b = calc_direction_alignment_score(coords_b, nodes, node_idx, direction_precomputed)
        nn_ratio_b = compute_separation_metrics(coords_b, is_user)[1]
        plot_and_save(coords_b, is_user, G, node_idx,
                      f"重み付けのみ, alpha={alpha}\n方向整列スコア={dir_score_b:.3f}, 異種/同種 最近傍距離比={nn_ratio_b:.3f}, 収束={conv_b}",
                      f"{out_dir}/layout_B_alpha{alpha}.png", edge_labels=edge_labels)

        coords_c, _, conv_c, _, _ = compute_layout_method(
            "C", common_deg, weight, alpha, nodes, node_idx, is_user,
            direction_precomputed, seed=plot_seed, gamma=gamma
        )
        dir_score_c = calc_direction_alignment_score(coords_c, nodes, node_idx, direction_precomputed)
        nn_ratio_c = compute_separation_metrics(coords_c, is_user)[1]
        plot_and_save(coords_c, is_user, G, node_idx,
                      f"方向整列との組み合わせ, alpha={alpha}\n方向整列スコア={dir_score_c:.3f}, 異種/同種 最近傍距離比={nn_ratio_c:.3f}, 収束={conv_c}",
                      f"{out_dir}/layout_C_alpha{alpha}.png", edge_labels=edge_labels)

def main_balance_experiment(M, dataset_label):
    """
    experiment_balance5.pyの本実験: 小規模サンプリングと、userを大幅に増やした
    大規模サンプリングを比較する。Mは既に読み込み済みのグラフ(MovieLensでもDBLPでも可)、
    dataset_labelは出力ファイル名・見出しの接頭辞("movielens"/"dblp"等)。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_balance_experiment {'#' * 20}")

    run_sampling_parameter_sweep(M, [SMALL_BUILD_KWARGS, LARGE_BUILD_KWARGS])

    run_full_experiment(M, SMALL_BUILD_KWARGS, label=f"{dataset_label}_small", n_seeds=20)
    run_full_experiment(M, LARGE_BUILD_KWARGS, label=f"{dataset_label}_large", n_seeds=10)

    print("\n" + "=" * 60)
    print(f"[{dataset_label}_small]と[{dataset_label}_large]、それぞれのalpha_layout_th*.pngと"
          "separation_and_cluster_trend_*.pngを見比べてください。特にuser側のn_cluster(user)が"
          "alphaに応じて動くようになったかどうかが、「userノード数が少なすぎたこと」がどれだけ"
          "非対称性に寄与していたかの直接的な確認になります。")


def beta_transform(alpha):
    """
    β変換: α<0.5ではβ=0.5α、α>=0.5ではβ=0.75+0.5(α-0.5)というピースワイズ線形変換。
    α=0.5の前後でβが0.25→0.75へ不連続にジャンプする(β≈0.5付近の値を意図的に避ける)。
    「同種・異種の重みがβ≈0.5付近で拮抗すること自体が、中間alphaで観測される収束不安定性の
    主因かどうか」を検証するための変換(その領域を素通りさせても不安定性が残るなら、
    主因ではないと言える)。
    """
    if alpha < 0.5:
        return 0.5 * alpha
    else:
        return 0.75 + 0.5 * (alpha - 0.5)


def _bc_seed_worker(args):
    """run_b_vs_c_alpha_sweepの1(method, seed)分の計算(ProcessPoolExecutorに渡すため関数化)。"""
    (method, common_deg, weight, effective_alpha, nodes, node_idx, is_user,
     direction_precomputed, seed, gamma) = args
    coords, _, converged, _, _ = compute_layout_method(
        method, common_deg, weight, effective_alpha, nodes, node_idx, is_user,
        direction_precomputed, seed=seed, gamma=gamma
    )
    if converged:
        dir_score = calc_direction_alignment_score(coords, nodes, node_idx, direction_precomputed)
        nn_ratio = compute_separation_metrics(coords, is_user)[1]
        cq_user, cq_movie = compute_cluster_quality(nodes, is_user, common_deg, coords)
    else:
        dir_score, nn_ratio, cq_user, cq_movie = (float("nan"),) * 4
    return converged, dir_score, nn_ratio, cq_user, cq_movie


def run_b_vs_c_alpha_sweep(G, dataset_label, weight_mode="uniform", alphas=None, n_seeds=20, gamma=0.1,
                           alpha_transform=None, n_workers=None):
    """
    手法B(alpha混合のみ)とC(alpha混合+方向整列)を、同じサブグラフGに対してalphaを
    振りながら比較する。weight_modeで異種ペア(実エッジ)の重みづけ方式を切り替えられる:
      - "uniform"    : build_matrices_uniform_weight (先生のご提案の検証用、全ての実エッジを同じ重みにする)
      - "degree"     : build_matrices (次数ベースのresource allocation指数で重みづけ、0.4〜1.0に正規化)
      - "commonality": build_matrices_commonality_weight (隣接ノードの共通度=sparsify前の
                       同種Jaccard類似度から、実エッジの自然長を設計する)
    同じ関数・同じG・同じ絞り込みロジックで重み方式だけを変えられるため、
    「実エッジの重み設計がα=1.0付近の劣化や中間alphaでの収束不安定性の主因かどうか」を
    直接比較できる。

    alpha_transformを指定すると、alphasでスイープする値(表示・比較の基準となる「見かけの
    alpha」)はそのままに、実際にcompute_layout_methodへ渡す値だけをalpha_transform(alpha)に
    差し替える(例: beta_transform)。同じ見かけ上のalphaグリッドのまま、内部で使う
    混合係数だけを変えた場合の効果を比較できる。

    n_workersを指定すると(例: n_workers=4)、各alphaにつきB/C×n_seeds回のレイアウト計算を
    ProcessPoolExecutorで並列実行する。デフォルトNoneは従来通り逐次実行で、結果(収束数・
    各種平均値)は並列実行でも完全に同一になる(seedで決まる計算内容自体は変わらないため)。
    """
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, weight_mode, THRESHOLD_COMMON_DEG, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY
    )
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    if alphas is None:
        alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)

    print(f"\n{'-' * 15} [{dataset_label}] B vs C alphaスイープ (weight_mode={weight_mode}) {'-' * 15}")
    print(f"{'alpha':>6} | {'B conv':>8} | {'C conv':>8} | {'B dir(conv)':>12} | {'C dir(conv)':>12} | {'B nn(conv)':>11} | {'C nn(conv)':>11}")

    rows = []
    executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers else None
    try:
        for alpha in alphas:
            effective_alpha = alpha_transform(alpha) if alpha_transform is not None else alpha

            tasks = [
                (method, common_deg, weight, effective_alpha, nodes, node_idx, is_user,
                 direction_precomputed, seed, gamma)
                for method in ("B", "C") for seed in range(n_seeds)
            ]
            if executor is not None:
                results = list(executor.map(_bc_seed_worker, tasks))
            else:
                results = [_bc_seed_worker(t) for t in tasks]

            b_results, c_results = results[:n_seeds], results[n_seeds:]
            n_conv_b = sum(1 for conv, _, _, _, _ in b_results if conv)
            n_conv_c = sum(1 for conv, _, _, _, _ in c_results if conv)
            dir_b = [d for conv, d, _, _, _ in b_results if conv]
            dir_c = [d for conv, d, _, _, _ in c_results if conv]
            nn_b = [n for conv, _, n, _, _ in b_results if conv]
            nn_c = [n for conv, _, n, _, _ in c_results if conv]
            # cq_*はそのtypeの正解クラスタが2個未満だとnanを含みうるため、
            # 平均を取る前に非nan値だけへ絞る(np.nanmeanの"Mean of empty slice"警告を避けるため)。
            cq_user_b = [v for conv, _, _, v, _ in b_results if conv and not np.isnan(v)]
            cq_user_c = [v for conv, _, _, v, _ in c_results if conv and not np.isnan(v)]
            cq_movie_b = [v for conv, _, _, _, v in b_results if conv and not np.isnan(v)]
            cq_movie_c = [v for conv, _, _, _, v in c_results if conv and not np.isnan(v)]

            row = {
                "alpha": alpha, "n_conv_b": n_conv_b, "n_conv_c": n_conv_c,
                "dir_b": np.nanmean(dir_b) if dir_b else float("nan"),
                "dir_c": np.nanmean(dir_c) if dir_c else float("nan"),
                "nn_b": np.nanmean(nn_b) if nn_b else float("nan"),
                "nn_c": np.nanmean(nn_c) if nn_c else float("nan"),
                "cq_user_b": np.mean(cq_user_b) if cq_user_b else float("nan"),
                "cq_user_c": np.mean(cq_user_c) if cq_user_c else float("nan"),
                "cq_movie_b": np.mean(cq_movie_b) if cq_movie_b else float("nan"),
                "cq_movie_c": np.mean(cq_movie_c) if cq_movie_c else float("nan"),
            }
            rows.append(row)

            db = f"{row['dir_b']:.3f}" if dir_b else "N/A"
            dc = f"{row['dir_c']:.3f}" if dir_c else "N/A"
            nb = f"{row['nn_b']:.3f}" if nn_b else "N/A"
            nc = f"{row['nn_c']:.3f}" if nn_c else "N/A"
            print(f"{alpha:>6.2f} | {n_conv_b:>4d}/{n_seeds:<3d}| {n_conv_c:>4d}/{n_seeds:<3d}| "
                  f"{db:>12} | {dc:>12} | {nb:>11} | {nc:>11}")
    finally:
        if executor is not None:
            executor.shutdown()

    print("\n[Cluster Quality (CQ): レイアウトのk-meansクラスタが真の構造クラスタとどれだけ一致するか]")
    print(f"{'alpha':>6} | {'B cq(u)':>9} | {'C cq(u)':>9} | {'B cq(m)':>9} | {'C cq(m)':>9}")
    for row in rows:
        print(f"{row['alpha']:>6.2f} | {row['cq_user_b']:>9.3f} | {row['cq_user_c']:>9.3f} | "
              f"{row['cq_movie_b']:>9.3f} | {row['cq_movie_c']:>9.3f}")

    return {
        "nodes": nodes, "node_idx": node_idx, "common_deg": common_deg, "weight": weight,
        "is_user": is_user, "edges": edges, "edge_labels": edge_labels, "rows": rows,
    }


def _bcd_seed_worker(args):
    """run_b_c_d_alpha_sweepの1(method, seed)分の計算(ProcessPoolExecutorに渡すため関数化)。"""
    (method, common_deg, weight, alpha, nodes, node_idx, is_user,
     direction_precomputed, seed, gamma, anchor_weight) = args
    coords, _, converged, _, _ = compute_layout_method(
        method, common_deg, weight, alpha, nodes, node_idx, is_user,
        direction_precomputed, seed=seed, gamma=gamma, anchor_weight=anchor_weight
    )
    if converged:
        dir_score = calc_direction_alignment_score(coords, nodes, node_idx, direction_precomputed)
        nn_ratio = compute_separation_metrics(coords, is_user)[1]
        cq_user, cq_movie = compute_cluster_quality(nodes, is_user, common_deg, coords)
    else:
        dir_score, nn_ratio, cq_user, cq_movie = (float("nan"),) * 4
    return converged, dir_score, nn_ratio, cq_user, cq_movie


def run_b_c_d_alpha_sweep(G, dataset_label, weight_mode="uniform", alphas=None, n_seeds=20, gamma=0.1,
                          anchor_weight=1.0, methods=("B", "C", "D"), n_workers=None):
    """
    手法B(alpha混合のみ)・C(alpha混合+方向整列を同時最適化)・
    D(alpha混合で収束させた後、そのレイアウトを弱くアンカーしつつ方向整列のみを
    追加で最適化する逐次的束ね)を比較する。

    「方向整列を同時に最適化することの不安定性コストは、逐次的に(まずレイアウトを
    決めてから)束ねることで解消できるか」を検証するための実験。Dの収束率がCより
    明確に高く、dir_score/nn_ratioの低下がわずかであれば、「同時最適化はその
    不安定化コストに見合わない」という具体的な根拠になる。逆にDのdir_scoreが
    大きく劣化するなら、同時最適化でなければ得られない効果がある、という根拠になる。

    dir_score/nn_ratioに加えて、Cluster Quality(CQ、Claudeとの相談で追加)も
    集計する。CQは「レイアウトの座標をk-meansでクラスタリングしたものが、
    common_degのLouvainコミュニティ(真の構造的クラスタ)とどれだけ一致するか」を
    Jaccard類似度のベストマッチで測る指標(compute_cluster_quality参照)。
    nn_ratio/n_clusterでは捉えられない「レイアウトが構造的クラスタを空間的に
    忠実に再現できているか」を直接測るための指標。

    anchor_weightはstage2でレイアウトをどれだけ強く固定するかを決めるパラメータ
    (大きいほどstage1のレイアウトを厳密に保つが、方向整列による調整の余地が減る)。
    run_b_vs_c_alpha_sweepとは別の独立した関数にしてあり、既存のB/C比較の
    実験結果には一切影響しない。
    """
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, weight_mode, THRESHOLD_COMMON_DEG, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY
    )
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    if alphas is None:
        alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)

    print(f"\n{'-' * 15} [{dataset_label}] B/C/D alphaスイープ "
          f"(weight_mode={weight_mode}, anchor_weight={anchor_weight}) {'-' * 15}")
    print("[収束数・方向整列スコア・分離度]")
    header = (f"{'alpha':>6} | " + " | ".join(f"{m + ' conv':>9}" for m in methods) + " | " +
              " | ".join(f"{m + ' dir':>9}" for m in methods) + " | " +
              " | ".join(f"{m + ' nn':>9}" for m in methods))
    print(header)

    rows = []
    executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers else None
    try:
        for alpha in alphas:
            tasks = [
                (method, common_deg, weight, alpha, nodes, node_idx, is_user,
                 direction_precomputed, seed, gamma, anchor_weight)
                for method in methods for seed in range(n_seeds)
            ]
            if executor is not None:
                results = list(executor.map(_bcd_seed_worker, tasks))
            else:
                results = [_bcd_seed_worker(t) for t in tasks]

            row = {"alpha": alpha}
            for mi, method in enumerate(methods):
                m_results = results[mi * n_seeds:(mi + 1) * n_seeds]
                n_conv = sum(1 for conv, _, _, _, _ in m_results if conv)
                dir_vals = [d for conv, d, _, _, _ in m_results if conv]
                nn_vals = [n for conv, _, n, _, _ in m_results if conv]
                # cq_user_vals/cq_movie_valsは、そのtypeの正解クラスタが2個未満だった
                # 収束済みseedについてnanを含みうる(compute_cluster_quality参照)。
                # 全部nanのままnp.nanmeanに渡すと"Mean of empty slice"警告が出るため、
                # 事前に非nan値だけへ絞ってから平均する。
                cq_user_vals = [cqu for conv, _, _, cqu, _ in m_results if conv]
                cq_movie_vals = [cqm for conv, _, _, _, cqm in m_results if conv]
                cq_user_valid = [v for v in cq_user_vals if not np.isnan(v)]
                cq_movie_valid = [v for v in cq_movie_vals if not np.isnan(v)]
                row[f"n_conv_{method}"] = n_conv
                row[f"dir_{method}"] = np.nanmean(dir_vals) if dir_vals else float("nan")
                row[f"nn_{method}"] = np.nanmean(nn_vals) if nn_vals else float("nan")
                row[f"cq_user_{method}"] = np.mean(cq_user_valid) if cq_user_valid else float("nan")
                row[f"cq_movie_{method}"] = np.mean(cq_movie_valid) if cq_movie_valid else float("nan")
            rows.append(row)

            conv_cells = " | ".join(f"{row[f'n_conv_{m}']:>6d}/{n_seeds:<2d}" for m in methods)
            dir_cells = " | ".join(f"{row[f'dir_{m}']:>9.3f}" for m in methods)
            nn_cells = " | ".join(f"{row[f'nn_{m}']:>9.3f}" for m in methods)
            print(f"{alpha:>6.2f} | {conv_cells} | {dir_cells} | {nn_cells}")
    finally:
        if executor is not None:
            executor.shutdown()

    print("\n[Cluster Quality (CQ): レイアウトのk-meansクラスタが真の構造クラスタとどれだけ一致するか]")
    cq_header = (f"{'alpha':>6} | " + " | ".join(f"{m + ' cq(u)':>10}" for m in methods) + " | " +
                 " | ".join(f"{m + ' cq(m)':>10}" for m in methods))
    print(cq_header)
    for row in rows:
        cq_u_cells = " | ".join(f"{row[f'cq_user_{m}']:>10.3f}" for m in methods)
        cq_m_cells = " | ".join(f"{row[f'cq_movie_{m}']:>10.3f}" for m in methods)
        print(f"{row['alpha']:>6.2f} | {cq_u_cells} | {cq_m_cells}")

    return {
        "nodes": nodes, "node_idx": node_idx, "common_deg": common_deg, "weight": weight,
        "is_user": is_user, "edges": edges, "edge_labels": edge_labels, "rows": rows,
    }


def main_three_method_gamma_and_nmi(M, dataset_label):
    """
    three_method_experiment.pyの本実験: build_matrices_uniform_weightで
    手法B/Cのalphaスイープ(収束率・方向整列スコア・nn_ratio)を行い、
    続けてノードクラスタとエッジクラスタのNMIをuser/movie別に計算する。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_three_method_gamma_and_nmi {'#' * 20}")

    G = get_small_subgraph_cached(M)
    result = run_b_vs_c_alpha_sweep(G, dataset_label, weight_mode="uniform", n_seeds=20, gamma=0.1)

    nmi_all, nmi_user, nmi_movie = compute_split_nmi(
        result["nodes"], result["is_user"], result["common_deg"], result["edges"], result["edge_labels"]
    )
    print(f"全体のNMI: {nmi_all:.3f}")
    print(f"user側のみのNMI: {nmi_user:.3f}")
    print(f"movie側のみのNMI: {nmi_movie:.3f}")


def main_weight_mode_comparison(M, dataset_label, weight_modes=("uniform", "degree", "commonality"),
                                 n_seeds=15, gamma=0.1):
    """
    「実エッジの重みをどう設計するかが、α=1.0付近の劣化や中間alphaでの収束不安定性に
    どれだけ効くか」を、同じサブグラフGに対してrun_b_vs_c_alpha_sweepをweight_modesで
    指定した全方式(既定ではuniform/degree/commonalityの3方式)で実行し、直接比較する。
    weight_modeの略称は表中でu=uniform, d=degree, k=commonality(共通隣接度、
    method Cの"C"と紛らわしいkoetsuu/commonalityの頭文字)を使う。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_weight_mode_comparison {'#' * 20}")

    G = get_small_subgraph_cached(M)
    results = {
        wm: run_b_vs_c_alpha_sweep(G, dataset_label, weight_mode=wm, n_seeds=n_seeds, gamma=gamma)
        for wm in weight_modes
    }
    abbrev = {"uniform": "u", "degree": "d", "commonality": "k"}
    tags = [abbrev.get(wm, wm[:1]) for wm in weight_modes]

    print(f"\n{'=' * 100}\n[{dataset_label}] weight_mode比較 ({' vs '.join(weight_modes)})\n{'=' * 100}")

    header = f"{'alpha':>6} | " + " | ".join(f"{'B(' + t + ')':>8}" for t in tags) + " | " + \
             " | ".join(f"{'C(' + t + ')':>8}" for t in tags)
    print("[収束数]")
    print(header)
    all_rows = list(zip(*[results[wm]["rows"] for wm in weight_modes]))
    for row_tuple in all_rows:
        alpha = row_tuple[0]["alpha"]
        assert all(r["alpha"] == alpha for r in row_tuple)
        b_cells = " | ".join(f"{r['n_conv_b']:>5d}/{n_seeds:<2d}" for r in row_tuple)
        c_cells = " | ".join(f"{r['n_conv_c']:>5d}/{n_seeds:<2d}" for r in row_tuple)
        print(f"{alpha:>6.2f} | {b_cells} | {c_cells}")

    print("\n[手法Cの方向整列スコア(dir)・分離度(nn)]")
    header2 = f"{'alpha':>6} | " + " | ".join(f"{'dir(' + t + ')':>8}" for t in tags) + " | " + \
              " | ".join(f"{'nn(' + t + ')':>8}" for t in tags)
    print(header2)
    for row_tuple in all_rows:
        alpha = row_tuple[0]["alpha"]
        dir_cells = " | ".join(f"{r['dir_c']:>8.3f}" for r in row_tuple)
        nn_cells = " | ".join(f"{r['nn_c']:>8.3f}" for r in row_tuple)
        print(f"{alpha:>6.2f} | {dir_cells} | {nn_cells}")

    print(f"\n見方: {', '.join(f'{t}={wm}' for t, wm in zip(tags, weight_modes))}。alpha=1.0付近で"
          "収束数が低いほど、その重み方式がα=1.0付近の劣化に寄与していることの直接的な確認になる。"
          "一方、中間alpha(0.05〜0.95)の収束数がどの方式でも同程度に不安定であれば、"
          "重み設計は中間alphaの不安定性の主因ではないと言える。")

    return results


def main_beta_transform_comparison(M, dataset_label, weight_mode="uniform", n_seeds=15, gamma=0.1):
    """
    β変換(beta_transform)を混合係数に適用した場合、しない場合(通常のalpha)を
    同じサブグラフGに対して比較する。「β=0.5付近(同種・異種の重みが拮抗する領域)を
    素通りさせることで、中間alphaで観測される収束不安定性が改善するか」を検証するための実験。
    改善が見られなければ、「β≈0.5付近での拮抗」自体は不安定性の主因ではないと言える。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_beta_transform_comparison (weight_mode={weight_mode}) {'#' * 20}")

    G = get_small_subgraph_cached(M)
    result_raw = run_b_vs_c_alpha_sweep(
        G, dataset_label, weight_mode=weight_mode, n_seeds=n_seeds, gamma=gamma, alpha_transform=None
    )
    result_beta = run_b_vs_c_alpha_sweep(
        G, dataset_label, weight_mode=weight_mode, n_seeds=n_seeds, gamma=gamma, alpha_transform=beta_transform
    )

    print(f"\n{'=' * 100}\n[{dataset_label}] β変換比較 (raw alpha vs beta_transform(alpha), weight_mode={weight_mode})\n{'=' * 100}")
    print(f"{'alpha':>6} | {'beta':>6} | {'B conv(raw)':>11} | {'B conv(β)':>9} | {'C conv(raw)':>11} | {'C conv(β)':>9} | "
          f"{'C dir(raw)':>10} | {'C dir(β)':>9} | {'C nn(raw)':>9} | {'C nn(β)':>8}")
    for row_raw, row_beta in zip(result_raw["rows"], result_beta["rows"]):
        assert row_raw["alpha"] == row_beta["alpha"]
        alpha = row_raw["alpha"]
        print(f"{alpha:>6.2f} | {beta_transform(alpha):>6.2f} | "
              f"{row_raw['n_conv_b']:>8d}/{n_seeds:<2d} | {row_beta['n_conv_b']:>6d}/{n_seeds:<2d} | "
              f"{row_raw['n_conv_c']:>8d}/{n_seeds:<2d} | {row_beta['n_conv_c']:>6d}/{n_seeds:<2d} | "
              f"{row_raw['dir_c']:>10.3f} | {row_beta['dir_c']:>9.3f} | "
              f"{row_raw['nn_c']:>9.3f} | {row_beta['nn_c']:>8.3f}")

    print("\n見方: conv(raw)/conv(β)が通常のalpha/β変換適用時それぞれの収束数。"
          "β変換はalpha<0.5でβ<0.25、alpha>=0.5でβ>=0.75になるようジャンプさせ、"
          "β≈0.5付近を素通りさせる。中間alpha域の収束数がconv(β)でも改善しなければ、"
          "「β≈0.5付近での同種・異種重みの拮抗」自体は中間alphaの不安定性の主因ではない"
          "ことの直接的な確認になる。")

    return result_raw, result_beta


def compute_split_nmi_for_config(M, build_kwargs, label, threshold=THRESHOLD_COMMON_DEG):
    G = get_small_subgraph_cached(M, **build_kwargs)
    n_user = sum(1 for n in G.nodes() if n.startswith("u_"))
    n_movie = sum(1 for n in G.nodes() if n.startswith("m_"))
    print(f"\n{'=' * 20} [{label}] 分割NMIチェック (n_user={n_user}, n_movie={n_movie}) {'=' * 20}")

    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, "uniform", threshold, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY
    )
    edges, edge_labels, direction_precomputed = get_edge_direction_cached(G, node_idx)

    nmi_all, nmi_user, nmi_movie = compute_split_nmi(nodes, is_user, common_deg, edges, edge_labels)
    return {"label": label, "n_user": n_user, "n_movie": n_movie,
            "nmi_all": nmi_all, "nmi_user": nmi_user, "nmi_movie": nmi_movie}


def main_split_nmi_size_comparison(M, dataset_label):
    """
    「userノード数が少なすぎたことが、user側の分割NMIがnanになる原因だったのか」を、
    small(既定サンプリング)とlarge(userを大幅に増やしたサンプリング)で直接比較する。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_split_nmi_size_comparison {'#' * 20}")

    result_small = compute_split_nmi_for_config(M, SMALL_BUILD_KWARGS, label=f"{dataset_label}_small")
    result_large = compute_split_nmi_for_config(M, LARGE_BUILD_KWARGS, label=f"{dataset_label}_large")

    print("\n" + "=" * 60)
    print(f"{'config':>8} | {'n_user':>6} | {'n_movie':>7} | {'NMI(all)':>9} | {'NMI(user)':>10} | {'NMI(movie)':>11}")
    for r in [result_small, result_large]:
        print(f"{r['label']:>8} | {r['n_user']:>6d} | {r['n_movie']:>7d} | "
              f"{r['nmi_all']:>9.3f} | {r['nmi_user']:>10.3f} | {r['nmi_movie']:>11.3f}")
    print("\n見方: smallでuser側NMIがnanで、largeで実数値になれば、"
          "「user側は本質的にクラスタ構造がない」のではなく「userノード数が少なすぎて"
          "Louvainがuser側を1コミュニティにまとめてしまっていた」ことが直接確認できる。")

    return result_small, result_large


def main_alpha_grid_plot(M, dataset_label):
    """
    three_experiment2.py / plot.pyの本実験: build_matrices(degree-weighted)で
    alpha_gridの可視化画像を作る。
    """
    print(f"\n{'#' * 20} [{dataset_label}] main_alpha_grid_plot {'#' * 20}")

    G = get_small_subgraph_cached(M)
    nodes, node_idx, common_deg, weight, is_user = get_matrices_cached(
        G, "degree", THRESHOLD_COMMON_DEG, TOP_K_SAME_TYPE, MUTUAL_TOP_K_ONLY
    )

    alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)  # 21点
    plot_alpha_grid(G, common_deg, weight, nodes, node_idx, is_user, alphas,
                     gamma=0.1, seed=0, filename=f"alpha_grid_{dataset_label}.png")


def run_all_experiments_for_dataset(M, dataset_label):
    """
    グラフM(MovieLensでもDBLPでも可)に対して、7つの実験一式
    (サンプリング比較・小/大規模の本実験、3手法比較+NMI、重み方式(uniform/degree)比較、
    β変換比較、規模別分割NMI比較、alpha_gridの可視化、反発・初期配置のablation)を
    全て実行する。dataset_labelは出力ファイル名・見出しの接頭辞として使われるため、
    複数データセットの結果が混ざらず、同じディレクトリに出しても後で見比べられる
    (例: alpha_layout_th0.25_movielens_small.png vs alpha_layout_th0.25_dblp_small.png)。
    """
    print(f"\n{'=' * 70}\n[{dataset_label}] 全実験を開始します\n{'=' * 70}")
    main_balance_experiment(M, dataset_label)
    main_three_method_gamma_and_nmi(M, dataset_label)
    main_weight_mode_comparison(M, dataset_label)
    main_beta_transform_comparison(M, dataset_label)
    main_split_nmi_size_comparison(M, dataset_label)
    # main_alpha_grid_plot(M, dataset_label)  # main_weight_mode_image_comparisonが
    # degree版を含む形で置き換えるため、二重に計算しないようコメントアウト
    main_weight_mode_image_comparison(M, dataset_label)
    main_repulsion_and_init_ablation(M, dataset_label)


def main_compare_movielens_and_dblp():
    """
    MovieLensとDBLPに対して全く同じ実験一式(run_all_experiments_for_dataset)を実行し、
    両者の結果を出力ファイル名(dataset_label付き)で見比べられるようにする。
    """
    M_movielens = load_movielens_graph(DATA_PATH)
    run_all_experiments_for_dataset(M_movielens, "movielens")

    M_dblp = load_dblp_graph_cached()
    run_all_experiments_for_dataset(M_dblp, "dblp")


if __name__ == "__main__":
    main_compare_movielens_and_dblp()
