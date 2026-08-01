"""DBLP data loading (XML streaming parse + pickle cache)."""

import os
import pickle

import networkx as nx
from lxml import etree

from bipartite_layout.config import DBLP_CACHE_PATH, DBLP_DTD_PATH, DBLP_MAX_PAPERS, DBLP_PATH


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
