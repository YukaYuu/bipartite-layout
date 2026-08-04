import dataclasses

import pytest

from bipartite_layout.config import DEFAULT_CONFIG, LARGE_BUILD_KWARGS, SMALL_BUILD_KWARGS


def test_graph_build_defaults():
    assert DEFAULT_CONFIG.graph_build.threshold_common_deg == 0.25
    assert DEFAULT_CONFIG.graph_build.top_k_same_type == 5
    assert DEFAULT_CONFIG.graph_build.mutual_top_k_only is False


def test_cluster_defaults():
    assert DEFAULT_CONFIG.cluster.dbscan_min_samples == 2
    assert DEFAULT_CONFIG.cluster.dbscan_eps_scale == 1.5


def test_sampling_kwargs_shims_match_dataclasses():
    assert SMALL_BUILD_KWARGS["n_hub_movies"] == DEFAULT_CONFIG.small_sampling.n_hub_movies
    assert LARGE_BUILD_KWARGS["n_seed_movies"] == DEFAULT_CONFIG.large_sampling.n_seed_movies
    assert LARGE_BUILD_KWARGS["n_seed_movies"] == 15


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONFIG.graph_build.threshold_common_deg = 0.5
