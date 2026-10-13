"""Frozen-readout RSS evaluation on explicit data, or a LABELED SYNTHETIC example.

  python demo_readout.py --example
  python demo_readout.py --example --export-example ./example_readout_data
  python demo_readout.py --data-dir /path/to/real_readout_arrays

Array shapes, labels, frozen weights and the split rules are in the README Demo section.
The example does NOT reproduce the Gemma/Llama safety-probe results in the paper.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rss_readout import FrozenLinearReadout, analyze_dataset, summarize_cases


REQUIRED = ("clean", "selection", "heldout", "labels", "weights", "bias", "threshold")


def synthetic_example(seed=2026, n_cases=90, n_features=32):
    """Deterministic *synthetic* test fixture with both failing/nonfailing cases.

    No LLM, pretrained SAE, real-world labels or published probe coefficients
    are involved. Constructed solely to test the evaluation pipeline.
    """
    rng = np.random.default_rng(seed)
    F, N = int(n_features), int(n_cases)
    if F < 8 or N < 6:
        raise ValueError("synthetic example requires n_features >= 8 and n_cases >= 6")
    w = rng.normal(0, 0.05, F)
    w[:8] = rng.choice([-1, 1], 8) * rng.uniform(0.7, 1.4, 8)
    mean = rng.normal(0, 0.4, F)
    scale = rng.uniform(0.7, 1.6, F)
    bias, threshold = 0.18, 0.10
    labels = rng.choice([-1, 1], size=N)
    clean = np.zeros((N, F))
    selection = np.zeros((N, 5, F))
    heldout = np.zeros((N, 10, F))
    w_norm2 = float(w @ w)
    for i in range(N):
        y = labels[i]
        m0 = rng.uniform(0.6, 1.8)
        x_std = rng.normal(0, 0.18, F)
        # Build a clean example with a known positive signed classification margin.
        x_std += (y * m0 - (x_std @ w + bias - threshold)) * w / w_norm2
        # Some variants move far enough to cause recurrent held-out errors.
        amplitude = m0 + rng.uniform(0.4, 1.0) if i % 3 == 0 else m0 * rng.uniform(0.05, 0.7)
        shift = -y * amplitude * w / w_norm2
        clean[i] = mean + scale * x_std
        selection[i] = mean + scale * (x_std + shift + rng.normal(0, 0.035, (5, F)))
        heldout[i] = mean + scale * (x_std + shift + rng.normal(0, 0.065, (10, F)))

    arrays = {"clean": clean, "selection": selection, "heldout": heldout,
              "labels": labels, "weights": w, "train_mean": mean,
              "train_scale": scale, "bias": np.array(bias),
              "threshold": np.array(threshold)}
    return arrays


def save_arrays(directory, arrays):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(directory / f"{name}.npy", arr, allow_pickle=False)
    return directory


def load_arrays(directory):
    """Memory-map each .npy array; validation processes only one case at a time."""
    directory = Path(directory)
    missing = [name for name in REQUIRED if not (directory / f"{name}.npy").is_file()]
    if missing:
        raise ValueError(f"missing required arrays in {directory}: {', '.join(missing)}")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in REQUIRED}
    for name in ("train_mean", "train_scale"):
        if (directory / f"{name}.npy").is_file():
            arrays[name] = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    if ("train_mean" in arrays) != ("train_scale" in arrays):
        raise ValueError("provide both train_mean.npy and train_scale.npy, or neither")
    return arrays


def get_probe(arrays):
    def scalar(name):
        if name not in arrays:
            return 0.0
        a = np.asarray(arrays[name])
        if a.shape != () or not np.isfinite(float(a)):
            raise ValueError(f"{name}.npy must hold one finite scalar")
        return float(a)
    return FrozenLinearReadout(arrays["weights"],
                               bias=scalar("bias"), threshold=scalar("threshold"),
                               train_mean=arrays.get("train_mean"),
                               train_scale=arrays.get("train_scale"))


def run(arrays, *, min_failures=2, M=400, K=20):
    if len(arrays["selection"].shape) != 3 or len(arrays["heldout"].shape) != 3:
        raise ValueError("selection and heldout arrays must have shape (N, V, F)")
    if arrays["selection"].shape[1] != 5 or arrays["heldout"].shape[1] != 10:
        raise ValueError("this readout protocol requires 5 selection and 10 held-out variants per case")
    probe = get_probe(arrays)
    cases = analyze_dataset(arrays["clean"], arrays["selection"],
                            arrays["heldout"], arrays["labels"], probe,
                            min_failures=min_failures, M=M, K=K)
    return summarize_cases(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--example", action="store_true", help="run the labeled synthetic example")
    src.add_argument("--data-dir", type=Path, help="directory of previously extracted real activation arrays")
    parser.add_argument("--export-example", type=Path,
                        help="when used with --example, write .npy arrays for inspecting the schema")
    parser.add_argument("--min-failures", type=int, default=2)
    parser.add_argument("--M", type=int, default=400)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--json", type=Path, help="save summary JSON")
    args = parser.parse_args()
    if args.export_example and not args.example:
        parser.error("--export-example requires --example")
    arrays = synthetic_example() if args.example else load_arrays(args.data_dir)
    if args.export_example:
        dest = save_arrays(args.export_example, arrays)
        print(f"Exported SYNTHETIC example arrays to {dest}")
    summary = run(arrays, min_failures=args.min_failures, M=args.M, K=args.K)
    print("\n*** SYNTHETIC EXAMPLE — NOT PAPER RESULTS ***" if args.example
          else "\n*** USER-PROVIDED FROZEN READOUT DATA ***")
    print(f"Cases {summary['n_cases']}; recurrent failures {summary['n_recurrent_failures']}; "
          f"median RSS support {summary['median_support_size']:.0f}")
    print("All risk scores use SELECTION data only; outcomes use HELD-OUT data only.")
    print("AUROC (higher = better; NaN if no positive/negative outcome class):")
    for name, value in summary["auroc"].items():
        print(f"  {name:25s} {value:.4f}")
    print(f"Held-out logit-change R^2: {summary['heldout_logit_change_r2']:.4f}")
    print(f"Margin contribution retention: {summary['margin_contribution_retention']:.4f}")
    print(f"Readout-only restoration: {summary['heldout_restoration_rate']:.4f} "
          f"({summary['heldout_restored_count']}/{summary['heldout_error_count']} held-out errors reversed)")
    if args.json:
        # Undefined metrics (e.g., no failures -> undefined AUROC) are JSON null.
        def json_safe(value):
            if isinstance(value, dict):
                return {key: json_safe(v) for key, v in value.items()}
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return None
            return value

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
