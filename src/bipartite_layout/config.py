"""
Experiment configuration dataclasses.

Frozen (immutable) dataclasses replace the scattered module constants that
used to live here as plain values. Frozen dataclass instances are safe to
use as ordinary function-parameter defaults (same evaluate-once-at-def-time
semantics as a plain int/float/str constant, no mutable-default bug) --
they just aren't valid as class-body defaults for *other* dataclasses
without `field(default_factory=...)`.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PathsConfig:
    data_path: str = "/Users/owner/Downloads/filtered-ml-1m/train.txt"
    dblp_path: str = "/Users/owner/Downloads/dblp.xml"
    dblp_dtd_path: str = "/Users/owner/Downloads/dblp.dtd"
    dblp_max_papers: int = 300_000
    # load_dblp_graphの結果をキャッシュするpickleファイル。3.8GBのXMLを毎回パースし直すのは
    # 無駄なので、dblp_path/dblp_max_papersを変えない限り2回目以降はここから読み込む。
    dblp_cache_path: str = "/Users/owner/Downloads/dblp_graph_cache.pkl"


@dataclass(frozen=True)
class SamplingConfig:
    n_seed_movies: int = 5
    n_users_per_movie: int = 20
    n_movies_per_user: int = 5
    n_hub_movies: int = 3
    n_focus_users_per_hub: int = 4
    n_movies_per_focus_user: int = 3


SMALL_SAMPLING = SamplingConfig()
# --- 大規模サンプリング用パラメータ(userを大幅に増やし、movieと揃える実験用) ---
LARGE_SAMPLING = SamplingConfig(
    n_seed_movies=15,
    n_users_per_movie=60,
    n_movies_per_user=8,
    n_hub_movies=15,
    n_focus_users_per_hub=5,
    n_movies_per_focus_user=4,
)


@dataclass(frozen=True)
class GraphBuildConfig:
    threshold_common_deg: float = 0.25
    top_k_same_type: int = 5
    mutual_top_k_only: bool = False


@dataclass(frozen=True)
class ClusterConfig:
    dbscan_min_samples: int = 2
    dbscan_eps_scale: float = 1.5


@dataclass(frozen=True)
class LayoutConfig:
    base_cutoff: float = 0.3
    base_n: int = 32
    strength: float = 0.3
    gamma: float = 1.0
    anchor_weight: float = 1.0
    maxiter: int = 500
    # 実エッジ項の係数を(1-alpha)ではなく(1-alpha+real_edge_epsilon)にすることで、
    # alpha=1.0でも実エッジ制約を完全には消さない(先生からのご指摘への対応)。
    # デフォルト0.0は既存の(1-alpha)と完全に同じ挙動を保つ(後方互換)。
    real_edge_epsilon: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    graph_build: GraphBuildConfig = field(default_factory=GraphBuildConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    small_sampling: SamplingConfig = field(default_factory=lambda: SMALL_SAMPLING)
    large_sampling: SamplingConfig = field(default_factory=lambda: LARGE_SAMPLING)


DEFAULT_CONFIG = ExperimentConfig()

# 後方互換のためのdictビュー(get_small_subgraph_cached(M, **SMALL_BUILD_KWARGS)のような
# 呼び出し方をそのまま残せるようにするための橋渡し)。
SMALL_BUILD_KWARGS = dict(
    n_hub_movies=SMALL_SAMPLING.n_hub_movies,
    n_focus_users_per_hub=SMALL_SAMPLING.n_focus_users_per_hub,
    n_movies_per_focus_user=SMALL_SAMPLING.n_movies_per_focus_user,
)

LARGE_BUILD_KWARGS = dict(
    n_seed_movies=LARGE_SAMPLING.n_seed_movies,
    n_users_per_movie=LARGE_SAMPLING.n_users_per_movie,
    n_movies_per_user=LARGE_SAMPLING.n_movies_per_user,
    n_hub_movies=LARGE_SAMPLING.n_hub_movies,
    n_focus_users_per_hub=LARGE_SAMPLING.n_focus_users_per_hub,
    n_movies_per_focus_user=LARGE_SAMPLING.n_movies_per_focus_user,
)
