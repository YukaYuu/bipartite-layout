import numpy as np

from bipartite_layout.linlog import compute_linlog_layout, linlog_stress_and_grad


def test_linlog_gradient_matches_finite_difference():
    """
    koala(Andreas NoackのLinLogエネルギーモデル、MinimizerClassic.javaの移植)の
    エネルギー関数の勾配が、有限差分と一致することを確認する(このモデルを使った
    実験結果を信用する前提となる、唯一かつ最重要のテスト)。
    """
    rng = np.random.default_rng(0)
    N = 8
    attr_weight = rng.uniform(0, 1, size=(N, N))
    attr_weight = (attr_weight + attr_weight.T) / 2
    np.fill_diagonal(attr_weight, 0)
    attr_weight[attr_weight < 0.5] = 0

    x0 = rng.uniform(-1, 1, size=N * 2)
    barycenter = x0.reshape(N, 2).mean(axis=0)

    for repu_exponent in (0.0, -1.0):
        for attr_exponent in (1.0, 3.0):
            _, grad = linlog_stress_and_grad(
                x0, attr_weight, repu_exponent, attr_exponent, 0.0001, barycenter
            )

            eps = 1e-6
            numeric_grad = np.zeros_like(x0)
            for i in range(len(x0)):
                xp, xm = x0.copy(), x0.copy()
                xp[i] += eps
                xm[i] -= eps
                ep, _ = linlog_stress_and_grad(xp, attr_weight, repu_exponent, attr_exponent, 0.0001, barycenter)
                em, _ = linlog_stress_and_grad(xm, attr_weight, repu_exponent, attr_exponent, 0.0001, barycenter)
                numeric_grad[i] = (ep - em) / (2 * eps)

            assert np.allclose(grad, numeric_grad, atol=1e-5, rtol=1e-4), (
                f"gradient mismatch at repu_exponent={repu_exponent}, attr_exponent={attr_exponent}"
            )


def test_linlog_pulls_connected_nodes_together():
    """
    最小限の定性チェック: 強く結びついた2つのペアが、レイアウト後に
    それぞれのペア内では近く、ペア間では遠くなることを確認する。
    """
    N = 4
    attr_weight = np.zeros((N, N))
    attr_weight[0, 1] = attr_weight[1, 0] = 5.0
    attr_weight[2, 3] = attr_weight[3, 2] = 5.0

    coords, _, converged, _, _ = compute_linlog_layout(attr_weight, seed=0, maxiter=500)
    assert converged

    dist = lambda i, j: np.linalg.norm(coords[i] - coords[j])
    within_pair = (dist(0, 1) + dist(2, 3)) / 2
    across_pairs = (dist(0, 2) + dist(0, 3) + dist(1, 2) + dist(1, 3)) / 4
    assert within_pair < across_pairs
