"""Held-out validation and frozen linear readout evaluation."""

import numpy as np
import pytest

from heldout_variants import SETTINGS, ensure_disjoint, heldout_variants
from rss_readout import (FrozenLinearReadout, analyze_case, analyze_dataset,
                         auroc, summarize_cases)
from rss_validation import evaluate_persistence
from assets.demo.demo_readout import get_probe, load_arrays, run, save_arrays, synthetic_example


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("base", [
    ("Can you", "explain", "why the sky looks blue during the day?"),
    ("Please", "describe", "how a rainbow forms after a storm."),
    ("Could you", "summarize", "why sleep matters for memory."),
])
def test_heldout_prompts_are_distinct_and_aligned(setting, base):
    opening, verb, target = base
    texts = heldout_variants(opening, verb, target, setting)
    assert len(texts) == 10 and len(set(texts)) == 10
    assert all(x.count(target) == 1 for x in texts)


def test_split_fails_on_overlap_or_bad_target():
    selection = [f"sel-{i} / target" for i in range(5)]
    heldout = [f"held-{i} / target" for i in range(10)]
    ensure_disjoint("clean target", selection, heldout, "target")
    with pytest.raises(ValueError, match="overlap"):
        ensure_disjoint("clean target", selection, [selection[0], *heldout[1:]], "target")
    with pytest.raises(ValueError, match="target phrase"):
        ensure_disjoint("clean target", selection, ["no marker", *heldout[1:]], "target")
    with pytest.raises(ValueError, match="expected"):
        ensure_disjoint("clean target", selection, heldout[:-1], "target")


def sample_activation_split():
    clean = np.zeros(4)
    # Feature 0 moves up in all selected variants; feature 2 goes down.
    selection = np.array([[2.0, 0, -1, 0], [1.8, 0, -1.2, 0],
                          [2.2, 0, -0.9, 0], [1.9, 0, -1.1, 0],
                          [1.7, 0, -1.0, 0]])
    heldout = np.array([[1, 0, -1, 0]] * 7 + [[-1, 0, 0.3, 0]] * 3, float)
    return clean, selection, heldout


def test_persistence_uses_selection_only():
    clean, selection, heldout = sample_activation_split()
    r = evaluate_persistence(clean, selection, heldout, K=2)
    assert r.support.tolist() == [0, 2]
    assert r.direction.tolist() == [1, -1]
    assert r.heldout_rates.tolist() == [0.7, 0.7]
    # Deliberately invert ALL held-out responses: selected features do not change.
    inverted = evaluate_persistence(clean, selection, -heldout, K=2)
    assert np.array_equal(inverted.support, r.support)
    assert np.array_equal(inverted.direction, r.direction)
    assert inverted.mean_rss == pytest.approx(r.mean_rss)
    assert inverted.mean_rate == pytest.approx(0.3)


def test_persistence_empty_support_is_unavailable():
    c = np.zeros(3)
    r = evaluate_persistence(c, np.zeros((5, 3)), np.zeros((10, 3)))
    assert len(r.support) == 0
    assert np.isnan(r.mean_rate) and np.isnan(r.magnitude_mean_rate)


@pytest.mark.parametrize("name, val", [("selection", np.zeros((3, 2))),
                                       ("heldout", np.zeros((4, 3)))])
def test_persistence_rejects_wrong_dimensions(name, val):
    clean, selection, heldout = sample_activation_split()
    if name == "selection":
        selection = val
    else:
        heldout = val
    with pytest.raises(ValueError, match=name):
        evaluate_persistence(clean, selection, heldout)


def test_probe_standardization_and_threshold():
    probe = FrozenLinearReadout(weights=np.array([2., -1.]), bias=0.3,
                                threshold=0.2, train_mean=np.array([1., 2.]),
                                train_scale=np.array([2., 4.]))
    assert float(probe.score(np.array([3., 6.]))) == pytest.approx(1.3)
    with pytest.raises(ValueError, match="together"):
        FrozenLinearReadout(np.ones(2), train_mean=np.zeros(2))
    with pytest.raises(ValueError, match="strictly positive"):
        FrozenLinearReadout(np.ones(2), train_mean=np.zeros(2), train_scale=np.array([1., 0.]))


def test_margin_erosion_exactly_reproduces_selection_margin():
    probe = FrozenLinearReadout(np.array([1., 2., -1.]), bias=0.4, threshold=0.2,
                                train_mean=np.array([1., 1., 1.]), train_scale=np.array([2., 1., 4.]))
    clean = np.array([3., 1., 5.])
    selection = clean + np.tile([2., -0.5, 1.], (5, 1))
    heldout = clean + np.tile([2., -1., 1.], (10, 1))
    r = analyze_case(clean, selection, heldout, label=1, probe=probe)
    assert r.residual_margin == pytest.approx(np.mean(probe.score(selection) - probe.threshold))
    assert r.clean_margin - r.erosion == pytest.approx(r.residual_margin)
    assert r.support_residual_margin == pytest.approx(r.clean_margin - r.support_erosion)
    assert np.allclose(r.heldout_dlogits, probe.score(heldout) - probe.score(clean))
    assert np.allclose(r.restored_margins,
                       r.heldout_margins - r.predicted_heldout_dlogits)


def test_negative_class_and_strict_failure():
    probe = FrozenLinearReadout(np.array([1., 0.]))
    c = np.array([-1., 0.])
    sel = np.tile([2., 0.], (5, 1))
    held = np.tile([2., 0.], (10, 1))
    r = analyze_case(c, sel, held, label=-1, probe=probe)
    assert r.clean_margin == 1
    assert r.erosion > 0
    assert np.all(r.heldout_margins < 0)
    assert r.recurrent_failure
    assert np.all(r.restored_margins > 0)
    with pytest.raises(ValueError, match="label"):
        analyze_case(c, sel, held, label=0, probe=probe)


def test_recurrent_requires_two_negative_heldout_margins():
    probe = FrozenLinearReadout(np.array([1.]))
    c = np.array([1.])
    sel = np.ones((5, 1))
    held = np.ones((10, 1))
    held[0, 0] = -1.0
    r = analyze_case(c, sel, held, label=1, probe=probe)
    assert not r.recurrent_failure
    held[1, 0] = -0.5
    r2 = analyze_case(c, sel, held, label=1, probe=probe)
    assert r2.recurrent_failure
    assert np.isnan(r2.mean_rss)  # no RSS support on selection variants


def test_auroc_ties_and_undefined_single_class():
    assert auroc(np.array([1, 1, 0, 0]), np.array([1., 0., 0., 0.])) == pytest.approx(0.75)
    assert auroc(np.array([1, 0]), np.array([2., 1.])) == pytest.approx(1.)
    assert auroc(np.array([1, 0]), np.array([1., 2.])) == pytest.approx(0.)
    assert np.isnan(auroc(np.ones(3), np.ones(3)))
    with pytest.raises(ValueError, match="binary"):
        auroc(np.array([-1, 1]), np.array([0., 1.]))


def test_dataset_summaries_on_synthetic_fixture():
    arrays = synthetic_example(n_cases=45)
    summary = run(arrays)
    assert summary["n_cases"] == 45
    assert 0 < summary["n_recurrent_failures"] < 45
    for score in summary["auroc"].values():
        assert 0 <= score <= 1
    assert summary["heldout_logit_change_r2"] <= 1
    assert 0 <= summary["heldout_restoration_rate"] <= 1
    assert summary["heldout_error_count"] > 0


def test_standardized_probe_does_not_change_raw_rss_selection():
    # A large raw change can be small after standardization. RSS selects raw.
    probe = FrozenLinearReadout([1., 1.], train_mean=[0., 0.], train_scale=[100., 1.])
    sel = np.tile([2., 1.], (5, 1))
    held = np.tile([2., 1.], (10, 1))
    case = analyze_case(np.array([0., 0.]), sel, held, 1, probe, K=1)
    assert case.support.tolist() == [0]


def test_readout_inputs_are_frozen_and_export_roundtrips(tmp_path):
    arrays = synthetic_example(n_cases=18)
    save_arrays(tmp_path, arrays)
    loaded = load_arrays(tmp_path)
    assert all(isinstance(loaded[n], np.memmap) for n in ("clean", "selection", "heldout"))
    a, b = run(arrays), run(loaded)
    assert a["n_cases"] == b["n_cases"]
    assert b["auroc"] == pytest.approx(a["auroc"])
    assert b["heldout_logit_change_r2"] == pytest.approx(a["heldout_logit_change_r2"])
    assert get_probe(loaded).weights.shape == (32,)


def test_unmatched_feature_dims_and_labels_are_rejected():
    arrays = synthetic_example(n_cases=9)
    bad = dict(arrays)
    bad["labels"] = np.zeros(9)
    with pytest.raises(ValueError, match="labels"):
        run(bad)
    bad = dict(arrays)
    bad["weights"] = np.ones(31)
    with pytest.raises(ValueError, match="same shape as weights"):
        run(bad)


def test_empty_set_summary_and_malformed_mmap(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        summarize_cases([])
    with pytest.raises(ValueError, match="missing required"):
        load_arrays(tmp_path)


def test_no_full_model_interventions_in_restoration():
    """A readout-only counterfactual leaves unselected features untouched."""
    probe = FrozenLinearReadout(np.array([1.0, 1.0, 0.0]))
    clean = np.array([1., 2., 0.])
    sel = np.tile([0., 2., 0.], (5, 1))  # only feature 0 moves
    held = np.tile([-1., 3., 0.], (10, 1))
    c = analyze_case(clean, sel, held, label=1, probe=probe, K=1)
    assert c.support.tolist() == [0]
    assert c.restored_margins[0] == pytest.approx(4.0)  # clean feature 0 + modified feature 1


def test_heldout_values_never_change_probe_selection_risk():
    probe = FrozenLinearReadout([2., -0.5, 1.])
    clean = np.array([0.8, 0.2, 0.1])
    selected = np.tile(np.array([1.2, 0.4, 0.2]), (5, 1))
    heldout_a = np.tile(np.array([0.1, 0.3, 0.0]), (10, 1))
    heldout_b = np.tile(np.array([-30., 90., -20.]), (10, 1))
    a = analyze_case(clean, selected, heldout_a, 1, probe)
    b = analyze_case(clean, selected, heldout_b, 1, probe)
    assert a.support.tolist() == b.support.tolist()
    assert a.direction.tolist() == b.direction.tolist()
    assert a.clean_margin == b.clean_margin
    assert a.erosion == b.erosion
    assert a.support_erosion == b.support_erosion
    assert a.residual_margin == b.residual_margin
    assert a.support_residual_margin == b.support_residual_margin
    assert not np.array_equal(a.heldout_margins, b.heldout_margins)


def test_undefined_auroc_json_is_not_serialized_as_nan(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    arrays = synthetic_example(n_cases=9)
    # Force every transformed variant to be exactly the clean activation:
    # there can be no negative held-out margins and thus no AUROC.
    arrays["heldout"][:] = arrays["clean"][:, None, :]
    save_arrays(tmp_path / "data", arrays)
    report = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "assets" / "demo" / "demo_readout.py"
    p = subprocess.run([sys.executable, str(script), "--data-dir", str(tmp_path / "data"),
                        "--json", str(report)], capture_output=True, text=True, check=True)
    assert "nan" in p.stdout.lower()
    d = json.loads(report.read_text())
    assert d["auroc"]["clean_margin"] is None
