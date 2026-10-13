"""Frozen linear-readout evaluation for RSS support, without fitting a probe.

Implements the definitions in the RSS paper/README for selection-set erosion,
residual-margin risk, recurrent failure on disjoint held-out variants, held-out
logit-change reconstruction, margin-contribution retention, and readout-only
counterfactual restoration. Raw SAE activations select RSS features; only the
frozen probe uses standardized activations.

This is an evaluator, not a release of the paper's (unavailable) safety probes,
checkpoints, datasets, or reported AUROCs. Its inputs must come from an
independently trained, frozen readout and a held-out annotated dataset.
"""
from dataclasses import dataclass

import numpy as np

from rss import rss
from rss_validation import _activations


@dataclass(frozen=True)
class FrozenLinearReadout:
    weights: np.ndarray
    bias: float = 0.0
    threshold: float = 0.0
    train_mean: np.ndarray | None = None
    train_scale: np.ndarray | None = None

    def __post_init__(self):
        w = np.asarray(self.weights, dtype=np.float64)
        if w.ndim != 1 or not w.size or not np.isfinite(w).all():
            raise ValueError("weights must be a nonempty finite 1-D vector")
        if not np.isfinite(self.bias) or not np.isfinite(self.threshold):
            raise ValueError("bias and threshold must be finite")
        object.__setattr__(self, "weights", w)
        if (self.train_mean is None) != (self.train_scale is None):
            raise ValueError("train_mean and train_scale must be supplied together")
        if self.train_mean is not None:
            mu = np.asarray(self.train_mean, dtype=np.float64)
            scale = np.asarray(self.train_scale, dtype=np.float64)
            if mu.shape != w.shape or scale.shape != w.shape:
                raise ValueError("train_mean and train_scale must have the same shape as weights")
            if not np.isfinite(mu).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("train_mean must be finite and train_scale finite and strictly positive")
            object.__setattr__(self, "train_mean", mu)
            object.__setattr__(self, "train_scale", scale)

    def standardize(self, activations):
        a = np.asarray(activations, dtype=np.float64)
        if a.ndim < 1 or a.shape[-1] != self.weights.size:
            raise ValueError(f"expected trailing feature dimension {self.weights.size}, got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("non-finite readout activations")
        return a if self.train_mean is None else (a - self.train_mean) / self.train_scale

    def score(self, activations):
        return self.standardize(activations) @ self.weights + self.bias


@dataclass(frozen=True)
class ReadoutCase:
    support: np.ndarray
    direction: np.ndarray
    mean_rss: float
    label: int
    clean_margin: float
    erosion: float
    support_erosion: float
    residual_margin: float
    support_residual_margin: float
    heldout_margins: np.ndarray
    heldout_dlogits: np.ndarray
    predicted_heldout_dlogits: np.ndarray
    restored_margins: np.ndarray
    heldout_margin_contribution_retention: float
    recurrent_failure: bool

    @property
    def heldout_errors(self):
        return self.heldout_margins < 0

    @property
    def reversed_errors(self):
        return self.heldout_errors & (self.restored_margins > 0)


def analyze_case(clean, selection, heldout, label, probe, *, min_failures=2,
                 rho=0.8, kappa=0.8, tau=1e-5, M=400, K=20):
    """Compute one clean-prompt x transformation case with no held-out leakage.

    clean: (F,) raw SAE activations, selection: (5,F), heldout: (10,F),
    label: ground-truth signed class in {-1,+1}; probe is already frozen.
    Selection alone determines the RSS support, feature directions and erosion.
    Heldout is used ONLY to derive ground-truth evaluation targets/metrics.
    """
    if not isinstance(probe, FrozenLinearReadout):
        raise TypeError("probe must be a FrozenLinearReadout")
    if label not in (-1, 1):
        raise ValueError("label must be -1 or +1")
    if isinstance(min_failures, (bool, np.bool_)) or not isinstance(min_failures, (int, np.integer)) or min_failures < 1:
        raise ValueError("min_failures must be a positive integer")
    clean, selection, heldout = _activations(clean, selection, heldout)
    if clean.size != probe.weights.size:
        raise ValueError("SAE activation dimension differs from the probe feature dimension")

    # RSS must be computed on ORIGINAL SAE features, not standardized features.
    support, direction, mean_rss = rss(selection - clean, rho=rho, kappa=kappa,
                                      tau=tau, M=M, K=K)

    clean_std, sel_std, held_std = (probe.standardize(x) for x in (clean, selection, heldout))
    sel_delta_std = sel_std - clean_std
    held_delta_std = held_std - clean_std
    y = int(label)
    m0 = y * (float(probe.score(clean)) - probe.threshold)

    # e_f^sel = -y * w_f * mean_v(a_tilde(x_v) - a_tilde(x_0)).
    ef = -y * probe.weights * sel_delta_std.mean(axis=0)
    erosion = float(ef.sum())
    erosion_s = float(ef[support].sum())
    res = float(m0 - erosion)
    res_s = float(m0 - erosion_s)

    held_margins = y * (np.asarray(probe.score(heldout)) - probe.threshold)
    true_delta_logit = held_delta_std @ probe.weights
    predicted_delta_logit = held_delta_std[:, support] @ probe.weights[support]
    # A readout-only intervention, NOT a counterfactual language model run.
    restored_margins = held_margins - y * predicted_delta_logit

    ef_ho = -y * probe.weights * held_delta_std.mean(axis=0)
    denominator = float(np.abs(ef_ho).sum())
    retention = float(np.abs(ef_ho[support]).sum() / denominator) if denominator > 0 else float("nan")
    recurrent = bool(np.count_nonzero(held_margins < 0) >= min_failures)
    return ReadoutCase(support, direction, mean_rss, y, float(m0), erosion, erosion_s,
                       res, res_s, np.asarray(held_margins),
                       np.asarray(true_delta_logit), np.asarray(predicted_delta_logit),
                       np.asarray(restored_margins), retention, recurrent)


def auroc(labels, risk):
    """AUROC with half credit for ties. Returns NaN if only one class exists."""
    labels = np.asarray(labels)
    risk = np.asarray(risk, dtype=np.float64)
    if labels.ndim != 1 or risk.ndim != 1 or len(labels) != len(risk) or not len(labels):
        raise ValueError("labels and risk must be equally long nonempty 1-D arrays")
    if not np.isin(labels, (0, 1, False, True)).all() or not np.isfinite(risk).all():
        raise ValueError("binary labels and finite risk scores required")
    n_pos = int((labels == 1).sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(risk, kind="stable")
    sorted_risk = risk[order]
    rank = np.empty(len(risk), dtype=np.float64)
    start = 0
    while start < len(risk):
        end = start + 1
        while end < len(risk) and sorted_risk[end] == sorted_risk[start]:
            end += 1
        rank[order[start:end]] = (start + 1 + end) / 2  # mean of 1-based ranks
        start = end
    return float((rank[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize_cases(cases):
    """Dataset-level, strictly held-out metrics for a list of ReadoutCase values."""
    cases = list(cases)
    if not cases:
        raise ValueError("at least one readout case is required")
    if any(not isinstance(x, ReadoutCase) for x in cases):
        raise TypeError("cases must all be ReadoutCase instances")
    failure = np.array([c.recurrent_failure for c in cases], dtype=int)
    risk = {
        "clean_margin": [-c.clean_margin for c in cases],
        "rss_erosion": [c.support_erosion for c in cases],
        "full_erosion": [c.erosion for c in cases],
        "rss_residual_margin": [-c.support_residual_margin for c in cases],
        "full_residual_margin": [-c.residual_margin for c in cases],
    }
    all_true = np.concatenate([c.heldout_dlogits for c in cases])
    all_pred = np.concatenate([c.predicted_heldout_dlogits for c in cases])
    ss_tot = float(np.square(all_true - all_true.mean()).sum())
    r2 = 1 - float(np.square(all_true - all_pred).sum()) / ss_tot if ss_tot > 0 else float("nan")
    num_errors = sum(np.count_nonzero(c.heldout_errors) for c in cases)
    num_reversed = sum(np.count_nonzero(c.reversed_errors) for c in cases)
    valid_retention = [c.heldout_margin_contribution_retention for c in cases
                       if np.isfinite(c.heldout_margin_contribution_retention)]
    return {
        "n_cases": len(cases),
        "n_recurrent_failures": int(failure.sum()),
        "median_support_size": float(np.median([len(c.support) for c in cases])),
        "auroc": {name: auroc(failure, np.array(v)) for name, v in risk.items()},
        "heldout_logit_change_r2": float(r2),
        "margin_contribution_retention": float(np.mean(valid_retention)) if valid_retention else float("nan"),
        "heldout_error_count": int(num_errors),
        "heldout_restored_count": int(num_reversed),
        "heldout_restoration_rate": float(num_reversed / num_errors) if num_errors else float("nan"),
    }


def analyze_dataset(clean, selection, heldout, labels, probe, *, min_failures=2,
                    rho=0.8, kappa=0.8, tau=1e-5, M=400, K=20):
    """Process arrays in case-sized slices (supports mmap arrays for large SAEs).

    clean (N,F), selection (N,n_sel,F), heldout (N,n_ho,F), labels (N,).
    Multiple transformation settings can be represented by separate rows.
    """
    if not isinstance(probe, FrozenLinearReadout):
        raise TypeError("probe must be a FrozenLinearReadout")
    if len(clean.shape) != 2 or len(selection.shape) != 3 or len(heldout.shape) != 3:
        raise ValueError("clean must be (N,F), selection and heldout must be (N,V,F)")
    N, F = clean.shape
    if N < 1 or F != probe.weights.size or selection.shape[0] != N or heldout.shape[0] != N:
        raise ValueError("mismatched case or feature dimensions")
    if selection.shape[1] < 1 or heldout.shape[1] < 1 or selection.shape[2] != F or heldout.shape[2] != F:
        raise ValueError("invalid selection or heldout variant dimensions")
    labels = np.asarray(labels)
    if labels.shape != (N,) or not np.isin(labels, (-1, 1)).all():
        raise ValueError("labels must have shape (N,) and contain only -1 or +1")
    return [analyze_case(clean[i], selection[i], heldout[i], labels[i], probe,
                         min_failures=min_failures, rho=rho, kappa=kappa,
                         tau=tau, M=M, K=K) for i in range(N)]
