"""Redistribution Sensitivity Score (RSS).

For one prompt and one transformation, RSS keeps the SAE features that change a lot,
in most variants, and in the same direction. MeanRSS is their mean change.
"""
import numbers

import numpy as np


def _changes(delta, name):
    """delta as a finite float array of shape (n_variants, n_features); anything else is an error."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[0] == 0 or delta.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_variants, n_features) with both sizes at least 1, "
                         f"got shape {delta.shape}")
    bad = int(np.size(delta) - np.isfinite(delta).sum())
    if bad:
        raise ValueError(f"{name} has {bad} entries that are nan or inf")
    return delta


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _threshold(tau):
    if not np.isfinite(tau) or tau < 0:
        raise ValueError(f"tau must be finite and at least 0, got {tau!r}")
    return float(tau)


def rss(delta, rho=0.8, kappa=0.8, tau=1e-5, M=400, K=20):
    """Select the responsive support of one prompt under one transformation.

    delta: array of shape (n_variants, n_features), delta[v, f] = a_f(x_v) - a_f(x_0).
    Returns (support, direction, mean_rss): feature ids sorted by change size,
    their dominant direction (+1 / -1), and MeanRSS (nan when nothing is selected).
    Requires 0 <= rho <= 1, 1/2 < kappa <= 1 (so the direction is never a tie), tau >= 0
    and positive integers M and K. Invalid input raises ValueError instead of returning a number.
    """
    delta = _changes(delta, "delta")
    if not 0 <= rho <= 1:
        raise ValueError(f"rho must lie in [0, 1], got {rho!r}")
    if not 0.5 < kappa <= 1:
        raise ValueError(f"kappa must lie in (0.5, 1] so that every kept feature has one direction, got {kappa!r}")
    tau, M, K = _threshold(tau), _count(M, "M"), _count(K, "K")

    n = delta.shape[0]
    h = np.abs(delta).mean(axis=0)                     # mean absolute change
    up = (delta > tau).sum(axis=0)                     # variants where f goes up
    down = (delta < -tau).sum(axis=0)                  # variants where f goes down
    moved = up + down
    r = moved / n                                      # recurrence
    c = np.divide(np.maximum(up, down), moved, out=np.zeros(h.shape), where=moved > 0)  # consistency

    pool = np.argsort(-h, kind="stable")[:M]           # M largest changes
    ok = pool[(r[pool] >= rho) & (c[pool] >= kappa)]   # recurrent and one-directional
    ok = ok[np.lexsort((ok, -c[ok], -h[ok]))]          # largest h first; ties: consistency, id
    support = ok[:K]

    direction = np.sign(up[support] - down[support]).astype(int)
    mean_rss = float(h[support].mean()) if len(support) else float("nan")
    return support, direction, mean_rss


def held_out_rate(delta_ho, support, direction, tau=1e-5):
    """Share of held out variants in which each selected feature moves in its selected direction.

    delta_ho: changes on variants that were not used for selection, shape (n_variants, n_features).
    support, direction: the output of rss on the selection variants.
    Returns one rate per selected feature, 1[direction * delta > tau] averaged over the held out variants.
    """
    delta_ho = _changes(delta_ho, "delta_ho")
    tau = _threshold(tau)
    support = np.asarray(support, dtype=np.int64).reshape(-1)
    direction = np.asarray(direction).reshape(-1)
    if len(support) != len(direction):
        raise ValueError(f"support and direction differ in length, {len(support)} and {len(direction)}")
    if len(support) and (support.min() < 0 or support.max() >= delta_ho.shape[1]):
        raise ValueError(f"support has feature ids outside 0 to {delta_ho.shape[1] - 1}")
    if not np.isin(direction, (-1, 1)).all():
        raise ValueError("direction must contain only +1 and -1")
    return (direction * delta_ho[:, support] > tau).mean(axis=0)
