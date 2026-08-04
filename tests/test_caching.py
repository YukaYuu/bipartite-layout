import pytest

from bipartite_layout.caching import get_matrices_cached, get_small_subgraph_cached


def test_get_small_subgraph_cached_is_a_true_cache(full_graph):
    G1 = get_small_subgraph_cached(full_graph, n_seed_movies=3, n_users_per_movie=5)
    G2 = get_small_subgraph_cached(full_graph, n_seed_movies=3, n_users_per_movie=5)
    assert G1 is G2, "identical (M, kwargs) should hit the cache and return the same object"


def test_get_small_subgraph_cached_differentiates_kwargs(full_graph):
    G1 = get_small_subgraph_cached(full_graph, n_seed_movies=3, n_users_per_movie=5)
    G2 = get_small_subgraph_cached(full_graph, n_seed_movies=5, n_users_per_movie=5)
    assert G1 is not G2


def test_get_matrices_cached_is_a_true_cache(small_graph):
    result1 = get_matrices_cached(small_graph, "degree", 0.0, 3)
    result2 = get_matrices_cached(small_graph, "degree", 0.0, 3)
    assert result1 is result2


def test_get_matrices_cached_differentiates_weight_mode(small_graph):
    result_degree = get_matrices_cached(small_graph, "degree", 0.0, 3)
    result_uniform = get_matrices_cached(small_graph, "uniform", 0.0, 3)
    assert result_degree is not result_uniform


def test_get_matrices_cached_rejects_unknown_weight_mode(small_graph):
    with pytest.raises(ValueError):
        get_matrices_cached(small_graph, "not_a_real_mode", 0.0, 3)
