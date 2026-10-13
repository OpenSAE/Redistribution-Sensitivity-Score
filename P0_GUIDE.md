# RSS P0 demos: held-out validation and frozen-readout evaluation

These additions make two main parts of the RSS paper *executable without the
original large-scale experiment pipeline*. They do **not** reproduce the
published numerical results, since the paper's exact 27-SAE study, held-out
variant realizations, frozen safety probes and labeled evaluation data are not
included here. Original project files, including `rss.py`, `demo.py`, the two
READMEs and the tutorial, are left unchanged.

## 1. Held-out validation on a publicly available SAE

```bash
# Install the existing repository's model-demo dependencies (Python 3.12 recommended)
pip install -r requirements_demo.txt

# Quick experiment: 3 prompts, 2 transformations, public GPT-2 Small SAE
python demo_heldout.py

# Longer experiment: 24 prompts, all 6 transformations
python demo_heldout.py --max-prompts 24 --all-settings
```

The first run downloads GPT-2 Small and the public SAE checkpoint. No paper
checkpoints or original research scripts are needed. The demo:

1. Imports the exact five **selection variants** per transformation already
   defined in the repository's original `demo.py`.
2. Independently constructs ten *new*, hand-authored held-out variants from
   `heldout_variants.py`. It checks that the clean, selection and held-out
   prompt strings are distinct and that the target phrase occurs once.
3. Extracts SAE features at the original demo's aligned target token. The
   helper `rss_validation.evaluate_persistence` selects support/direction on
   the **five selection variants only**, then uses `rss.held_out_rate` to
   evaluate the selected directions on the other ten.
4. Prints the number of selected features and their held-out directional
   persistence. As an **illustrative control**, the demo also evaluates an
   equal-sized support chosen by magnitude alone in the same candidate pool.
   Magnitude-only directions use selection variants, with deterministic
   tie-breaking. This control is **not** asserted to match the paper's
   magnitude-only or random control protocols.

**Scientific limitations.** This is a small, public-model example; the
hand-authored held-out variants are NOT the held-out operator realizations
used by the paper. The original paper's 0.952 / 0.758 rates must not be
expected from this demo. It also does not test semantic equivalence with
human annotation or verify the exact paper-scale split. `rss_validation.py`
is independently usable with your own already-extracted SAE activations.

```python
from rss_validation import evaluate_persistence

# Raw SAE activation arrays, one clean prompt and one transformation:
# a0: (F,), a_selection: (5,F), a_heldout: (10,F)
r = evaluate_persistence(a0, a_selection, a_heldout)
print(r.support, r.direction, r.mean_rate)
```

An empty RSS support has an **undefined** held-out rate (`NaN`), never an
artificial zero.

## 2. Frozen linear-readout evaluation

The original paper's trained safety probe coefficients are **not available**.
Instead, `rss_readout.py` implements the evaluation protocol for **any
externally trained and frozen binary linear readout**. It does not fit a probe
on selection or held-out samples.

Try the end-to-end calculation with explicitly **synthetic** data:

```bash
pip install numpy
python demo_readout.py --example

# Inspect an example of the expected data files:
python demo_readout.py --example --export-example ./example_readout_data
python demo_readout.py --data-dir ./example_readout_data
```

The synthetic numbers are *only* integration-test outputs. They are **not**
Gemma / Llama results, real safety evaluation, or paper replication.

### Use your real frozen readout when it becomes available

Export `.npy` arrays into a directory:

| File | Shape | Meaning |
|---|---|---|
| `clean.npy` | `(N, F)` | Raw SAE activations of clean prompts |
| `selection.npy` | `(N, 5, F)` | Five selection variants for each prompt and setting |
| `heldout.npy` | `(N, 10, F)` | Ten disjoint held-out variants |
| `labels.npy` | `(N,)` | Ground-truth class signs `-1` or `+1` |
| `weights.npy` | `(F,)` | Weights of the already trained, **frozen** linear probe |
| `bias.npy` | scalar `()` | Frozen probe bias, even if it is zero |
| `threshold.npy` | scalar `()` | Frozen decision threshold, even if it is zero |
| `train_mean.npy` | optional `(F,)` | Mean computed on the **probe training set only** |
| `train_scale.npy` | optional `(F,)` | Positive scale computed on the **probe training set only** |

Supply `train_mean.npy` and `train_scale.npy` **together** if your probe reads
standardized activations. If you omit both, `weights.npy` must already be
valid for the supplied **raw** SAE activations. Do not recompute the
standardization on test variants. Every row represents one prompt and one
transformation setting; do not pool variants from unrelated prompts inside a
row. Keep each variant split disjoint and constructed *before* feature
selection. SAE features and probe weights must have the same ordering.

```bash
python demo_readout.py --data-dir /path/to/real_readout_arrays \
       --json /path/to/summary.json
```

The loader uses memory-mapped `.npy` arrays and processes one case at a time,
so already-extracted large-SAEs activations need not all be loaded into RAM.
For full paper-scale data, disk usage and execution time may still be large.

### What is computed

For a class sign `y` and a frozen standardized probe
`z(a) = w @ ((a - train_mean) / train_scale) + b`, the clean signed margin is
`m0 = y * (z(clean) - threshold)`.

- **RSS support:** `rss(selection - clean)` on **raw SAE features only**.
- **Selection erosion:** `e_f = -y * w_f * mean(selection_std[:,f] - clean_std[f])`.
- **Full erosion** `E = sum(e_f)`, **support erosion** `E_S = sum(e_f for f in RSS support)`.
- **Selection risk scores:** `-m0`, `E`, `E_S`, `-(m0-E)`, `-(m0-E_S)`.
- **Held-out failure:** at least two **negative** margins among ten held-out variants.
- **AUROC:** ranks that held-out binary outcome by selection-only risk scores;
  score ties receive standard half credit. Returns `NaN` if only one outcome
  class is present (written as `null` in JSON).
- **Held-out logit reconstruction:** actual held-out logit changes versus
  `sum(w_f * standardized_heldout_delta_f for f in RSS support)`, evaluated by
  dataset-level `R²`.
- **Margin-contribution retention:** ratio of absolute held-out *mean feature
  contributions* retained by the support, when denominator is nonzero.
- **Restoration:** restores **only the selected probe inputs to their clean
  values** and checks whether held-out errors become positive-margin outputs.
  It **does not** intervene on the language model or claim causality beyond
  this frozen readout counterfactual.

All score computation is restricted to **clean + selection** values; all
failure labels, reconstruction evaluation and restoration reporting use
**held-out** values. A correct held-out protocol cannot repair selection/test
leakage already present in externally supplied arrays.

## Tests and original-file preservation

```bash
pip install numpy pytest
python -m pytest -q
```

Original `rss.py`, `demo.py`, README files, and all other shipped files are
byte-for-byte unmodified in the full packaged version. Only the following
additions are needed:

```text
rss_validation.py
heldout_variants.py
demo_heldout.py
rss_readout.py
demo_readout.py
tests/test_p0.py
P0_GUIDE.md
```

The pure-NumPy code, splits and synthetic integration tests run without a
model download. A genuine online run of `demo_heldout.py` still requires
access to the public GPT-2 and SAE weights; it was not possible to perform
that particular integration test in the offline development environment.
