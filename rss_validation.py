"""Out-of-selection validation of RSS support and response direction.

The support and its direction are learned only from `selection` variants.  The
`heldout` variants are never used by RSS, its candidate pool, or the baseline.
No neural-network dependency is required: feed already-extracted SAE features.
"""
from dataclasses import dataclass

import numpy as np

from rss import held_out_rate, rss


@dataclass(frozen=True)
class PersistenceResult:
    support: np.ndarray
    direction: np.ndarray
    mean_rss: float
    heldout_rates: np.ndarray
    magnitude_support: np.ndarray
    magnitude_direction: np.ndarray
    magnitude_heldout_rates: np.ndarray

    @property
    def mean_rate(self):
        return float(self.heldout_rates.mean()) if self.heldout_rates.size else float("nan")

    @property
    def magnitude_mean_rate(self):
        return float(self.magnitude_heldout_rates.mean()) if self.magnitude_heldout_rates.size else float("nan")


def _activations(clean, selection, heldout):
    clean = np.asarray(clean, dtype=np.float64)
    selection = np.asarray(selection, dtype=np.float64)
    heldout = np.asarray(heldout, dtype=np.float64)
    if clean.ndim != 1 or not clean.size:
        raise ValueError("clean must be a nonempty vector (n_features,)")
    for name, a in (("selection", selection), ("heldout", heldout)):
        if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] != clean.size:
            raise ValueError(f"{name} must have shape (n_variants, {clean.size}), n_variants >= 1")
        if not np.isfinite(a).all():
            raise ValueError(f"{name} must contain finite activations")
    if not np.isfinite(clean).all():
        raise ValueError("clean must contain finite activations")
    return clean, selection, heldout


def _majority_direction(delta, tau, indices):
    """Direction for magnitude-only controls, with a deterministic tie-break.

    A tie between up and down counts is broken by the sign of the total change;
    a further exact tie uses +1. This is an illustrative matched-size control,
    not a claim to reproduce the paper's magnitude-only baseline precisely.
    """
    d = delta[:, indices]
    up = (d > tau).sum(axis=0)
    down = (d < -tau).sum(axis=0)
    total = d.sum(axis=0)
    return np.where(up > down, 1, np.where(up < down, -1, np.where(total < 0, -1, 1))).astype(int)


def evaluate_persistence(clean, selection, heldout, *, rho=0.8, kappa=0.8,
                         tau=1e-5, M=400, K=20):
    """Select on `selection`, evaluate only on `heldout`.

    All arrays contain *raw SAE features*, not standardized probe features.
    The magnitude-only comparison uses the same size as the RSS support and
    the same top-M mean-absolute-change candidate pool, but no recurrence or
    directional-consistency filter. An empty RSS support stays unavailable.
    """
    clean, selection, heldout = _activations(clean, selection, heldout)
    selected_delta = selection - clean[None, :]
    holdout_delta = heldout - clean[None, :]

    support, direction, mean_rss = rss(selected_delta, rho=rho, kappa=kappa,
                                      tau=tau, M=M, K=K)
    rates = held_out_rate(holdout_delta, support, direction, tau=tau)
    h = np.abs(selected_delta).mean(axis=0)
    pool = np.argsort(-h, kind="stable")[:min(M, len(h))]
    magnitude_support = pool[:len(support)]
    magnitude_direction = _majority_direction(selected_delta, tau, magnitude_support)
    mag_rates = held_out_rate(holdout_delta, magnitude_support, magnitude_direction, tau=tau)

    return PersistenceResult(support, direction, mean_rss, rates,
                             magnitude_support, magnitude_direction, mag_rates)
