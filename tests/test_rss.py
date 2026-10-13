"""Properties of the RSS selector that the paper relies on. Run with `pytest` from the repository root."""
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pytest

from rss import held_out_rate, rss

QUICKSTART = np.array([[2.0, 0.0, -1.0, 0.3],
                       [1.8, 0.0, -1.2, -0.3],
                       [2.2, 0.5, -0.9, 0.3],
                       [1.9, 0.0, -1.1, -0.3],
                       [0.0, 0.0, -1.0, 0.3]])


def random_delta(seed, n=5, F=300):
    """Sparse changes with a few features that move consistently, like SAE activations under rewrites."""
    rng = np.random.default_rng(seed)
    delta = rng.normal(0, 1, (n, F)) * (rng.random((n, F)) < 0.3)
    lead = rng.choice(F, 25, replace=False)
    delta[:, lead] += rng.choice([-1, 1], 25) * rng.uniform(1, 4, 25)
    return delta


def counts(delta, tau):
    return (delta > tau).sum(0), (delta < -tau).sum(0)


@pytest.mark.parametrize("readme", ["README.md", "README_cn.md"])
def test_readme_quickstart_runs_and_prints_what_it_says(readme):
    """The Quickstart block of each README is executed, and its output must match the comment under the print."""
    text = (Path(__file__).resolve().parents[1] / readme).read_text(encoding="utf8")
    block = re.search(r"## 🚀 [^\n]+\s+```python\n(.*?)```", text, re.S).group(1)
    expected = re.search(r"^# (.+)$", block.strip().splitlines()[-1]).group(1)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(block, {})
    assert out.getvalue().strip() == expected


def test_quickstart_values():
    support, direction, mean_rss = rss(QUICKSTART)
    assert support.tolist() == [0, 2] and direction.tolist() == [1, -1]
    assert mean_rss == pytest.approx((1.58 + 1.04) / 2)


def test_threshold_is_strict():
    """A change of exactly tau is not a response, so n_f counts changes greater than tau."""
    delta = np.full((5, 1), 0.5)
    assert rss(delta, tau=0.5)[0].size == 0
    assert rss(delta, tau=0.4999)[0].tolist() == [0]


def test_empty_support_is_unavailable_not_zero():
    support, direction, mean_rss = rss(np.zeros((5, 10)))
    assert support.size == 0 and direction.size == 0 and np.isnan(mean_rss)
    assert support.dtype.kind == "i" and direction.dtype.kind == "i"


def test_opposite_changes_do_not_cancel_in_h_but_fail_consistency():
    """A feature that flips sign has a large h_f, enters the pool, and is still rejected by kappa."""
    flip = np.array([[3.0], [-3.0], [3.0], [-3.0], [3.0]])
    small = np.full((5, 1), 0.1)
    delta = np.hstack([flip, small])
    assert np.abs(delta).mean(0)[0] == 3.0
    assert rss(delta, M=1)[0].size == 0        # only the flipping feature is in the pool
    assert rss(delta)[0].tolist() == [1]


@pytest.mark.parametrize("seed", range(20))
def test_proposition_2_directional_margin(seed):
    """With n = 5 and rho = kappa = 0.8 every kept feature moves its way at least 4 times and against it at most once."""
    delta = random_delta(seed)
    support, direction, _ = rss(delta)
    up, down = counts(delta, 1e-5)
    dominant = np.where(direction > 0, up[support], down[support])
    opposing = np.where(direction > 0, down[support], up[support])
    assert (dominant >= 4).all() and (opposing <= 1).all()
    assert (direction == np.sign(up[support] - down[support])).all()


@pytest.mark.parametrize("seed", range(20))
def test_proposition_1_top_k_of_the_eligible_set(seed):
    """The support is the k = min(K, |E_M|) eligible features with the largest h_f, sorted by h_f."""
    delta, M, K, tau = random_delta(seed), 60, 10, 1e-5
    h = np.abs(delta).mean(0)
    up, down = counts(delta, tau)
    moved = up + down
    c = np.where(moved > 0, np.maximum(up, down) / np.maximum(moved, 1), 0.0)
    pool = np.argsort(-h, kind="stable")[:M]
    eligible = [f for f in pool if moved[f] / 5 >= 0.8 and c[f] >= 0.8]
    expected = sorted(eligible, key=lambda f: -h[f])[:K]
    support, _, mean_rss = rss(delta, M=M, K=K)
    assert support.tolist() == expected
    assert mean_rss == pytest.approx(h[expected].mean())


@pytest.mark.parametrize("seed", range(10))
def test_scale_feature_order_and_variant_order(seed):
    delta = random_delta(seed)
    support, direction, mean_rss = rss(delta)
    # Scaling all changes and tau by the same factor keeps the support and scales MeanRSS.
    s2, d2, m2 = rss(delta * 7.5, tau=7.5e-5)
    assert s2.tolist() == support.tolist() and d2.tolist() == direction.tolist()
    assert m2 == pytest.approx(7.5 * mean_rss)
    # Reordering the variants changes nothing; renaming the features renames the support.
    rng = np.random.default_rng(seed)
    assert rss(delta[rng.permutation(5)])[0].tolist() == support.tolist()
    perm = rng.permutation(delta.shape[1])
    assert sorted(perm[rss(delta[:, perm])[0]].tolist()) == sorted(support.tolist())


def test_support_size_pool_and_order():
    delta = random_delta(0)
    h = np.abs(delta).mean(0)
    for M, K in [(5, 20), (400, 3), (1, 1)]:
        support, _, _ = rss(delta, M=M, K=K)
        assert len(support) <= min(K, M)
        assert set(support) <= set(np.argsort(-h, kind="stable")[:M])
        assert (np.diff(h[support]) <= 0).all()


def test_held_out_rate():
    support, direction, _ = rss(QUICKSTART)
    held_out = np.array([[1.0, 0.0, -0.5, 0.0],
                         [-1.0, 0.0, -0.5, 0.0],
                         [1.0, 0.0, 0.5, 0.0],
                         [0.0, 0.0, -0.5, 0.0]])
    assert held_out_rate(held_out, support, direction).tolist() == [0.5, 0.75]
    assert held_out_rate(held_out, support[:0], direction[:0]).size == 0


@pytest.mark.parametrize("delta, kwargs, message", [
    (np.array([1.0, 2.0]), {}, "shape"),
    (np.zeros((0, 3)), {}, "shape"),
    (np.array([[np.inf, 0.0]]), {}, "nan or inf"),
    (np.array([[np.nan, 0.0]]), {}, "nan or inf"),
    (QUICKSTART, {"K": -1}, "K must be a positive integer"),
    (QUICKSTART, {"K": 0}, "K must be a positive integer"),
    (QUICKSTART, {"K": 2.5}, "K must be a positive integer"),
    (QUICKSTART, {"K": True}, "K must be a positive integer"),
    (QUICKSTART, {"M": -1}, "M must be a positive integer"),
    (QUICKSTART, {"tau": -1.0}, "tau"),
    (QUICKSTART, {"tau": np.nan}, "tau"),
    (QUICKSTART, {"rho": 1.5}, "rho"),
    (QUICKSTART, {"kappa": 0.5}, "kappa"),
    (QUICKSTART, {"kappa": 1.2}, "kappa"),
])
def test_invalid_input_raises(delta, kwargs, message):
    with pytest.raises(ValueError, match=message):
        rss(delta, **kwargs)


def test_held_out_rate_checks_its_input():
    support, direction, _ = rss(QUICKSTART)
    with pytest.raises(ValueError, match="outside"):
        held_out_rate(np.zeros((3, 2)), support, direction)
    with pytest.raises(ValueError, match="differ in length"):
        held_out_rate(QUICKSTART, support, direction[:1])
    with pytest.raises(ValueError, match="direction"):
        held_out_rate(QUICKSTART, support, np.zeros_like(direction))
