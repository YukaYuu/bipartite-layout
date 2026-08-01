"""
Experiment configuration constants.

Plain module constants for now (mirrors the original combined_experiment.py
exactly) — these get replaced with frozen dataclasses in a follow-up commit.
Centralizing them here means that later change only touches this file plus
the handful of call sites that reference them by name.
"""

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
