# RSS core API

The public [`rss.py`](../rss.py) selector depends only on NumPy. It accepts changes in SAE activations, without performing model forward passes or selecting token positions.

```python
from rss import rss, held_out_rate
support, direction, mean_rss = rss(selection - clean)
rates = held_out_rate(heldout - clean, support, direction)
```

`clean` is a vector of shape `(n_features,)`. `selection` and `heldout` have shape `(n_variants, n_features)`. Extract these from the **same SAE, model, layer and aligned target token position**; use disjoint selection and held-out prompt variants. Subtract the clean SAE activation before calling `rss`, which expects changes `delta[v, f] = a_f(x_v) - a_f(x_0)`.

`rss(delta, rho=0.8, kappa=0.8, tau=1e-5, M=400, K=20)` returns:

- `support`: up to `K` feature indices selected from the top `M` by mean absolute change.
- `direction`: +1 or -1 for each selected feature's dominant response direction.
- `mean_rss`: mean absolute change across selected features, or `NaN` if no feature passes the recurrence and direction checks.

For five selection variants with the default recurrence/consistency thresholds, a selected feature moves in its chosen direction in at least four observations. `held_out_rate` returns a persistence fraction for each selected feature without selecting on the held-out data.

[`evaluate_persistence(clean, selection, heldout)`](../rss_validation.py) also returns a matched magnitude-only control. The optional [`rss_readout.py`](../rss_readout.py) applies a frozen linear readout to previously extracted SAE arrays; it does not rerun the language model.

The [demo guide](../assets/demo/README.md) explains how to reproduce the public GPT-2 demonstration and inspect the readout array format.
