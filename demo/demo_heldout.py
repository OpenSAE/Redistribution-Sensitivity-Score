"""GPT-2 Small + public SAE: select RSS features, then test unseen variants.

  python demo/demo_heldout.py                          # 3 prompts x 2 settings
  python demo/demo_heldout.py --max-prompts 24 --all-settings

Downloads the public GPT-2 and SAE checkpoints on the first run. Not a
reproduction of the full paper's 27-SAE, 1000-prompt validation experiment.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from heldout_variants import ensure_disjoint, heldout_variants
from rss_validation import evaluate_persistence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--settings", nargs="+", default=["Irrelevant context", "Format wrapper"])
    parser.add_argument("--all-settings", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_prompts <= 24:
        parser.error("--max-prompts must be between 1 and 24")

    # Import only when the user actually executes the model demo, keeping the
    # validation math importable and testable without GPU/model dependencies.
    import torch
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    from demo.demo import PROMPTS, SETTINGS, sae_features

    names = list(SETTINGS) if args.all_settings else args.settings
    if not names or any(n not in SETTINGS for n in names):
        parser.error(f"--settings must be chosen from {list(SETTINGS)}")
    if len(names) != len(set(names)):
        parser.error("--settings contains duplicates")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hook = f"blocks.{args.layer}.hook_resid_pre"
    print(f"Loading GPT-2 Small and SAE {hook} on {device}...")
    model = HookedTransformer.from_pretrained_no_processing(
        "gpt2", center_writing_weights=True, device=device)
    sae = SAE.from_pretrained("gpt2-small-res-jb", hook, device=device)
    sae = sae[0] if isinstance(sae, tuple) else sae

    results = {name: [] for name in names}
    with torch.no_grad():
        for n, (opening, verb, target) in enumerate(PROMPTS[:args.max_prompts], start=1):
            clean_text = f"{opening} {verb} {target}"
            clean = sae_features(model, sae, hook, clean_text, target)
            for name in names:
                selection_texts = SETTINGS[name](opening, verb, target)
                heldout_texts = heldout_variants(opening, verb, target, name)
                ensure_disjoint(clean_text, selection_texts, heldout_texts, target)
                selection = np.stack([sae_features(model, sae, hook, text, target)
                                      for text in selection_texts])
                heldout = np.stack([sae_features(model, sae, hook, text, target)
                                    for text in heldout_texts])
                r = evaluate_persistence(clean, selection, heldout)
                results[name].append(r)
                print(f"Prompt {n:>2} | {name:24s} | features={len(r.support):>2} "
                      f"held-out rate={r.mean_rate:.3f} "
                      f"magnitude-only={r.magnitude_mean_rate:.3f}", flush=True)

    print("\nHELD-OUT VALIDATION (illustrative public-model sample, not paper estimates)")
    print("Selection = 5 variants; evaluation = 10 different variants per prompt/setting.")
    print(f"{'Setting':25s} {'N':>3s} {'mean support':>13s} {'RSS rate':>11s} {'magnitude':>11s}")
    for name in names:
        r = results[name]
        # Pooled feature-level rate; ignore empty supports rather than misreporting zero.
        selected_rates = [v for item in r for v in item.heldout_rates]
        control_rates = [v for item in r for v in item.magnitude_heldout_rates]
        print(f"{name:25s} {len(r):3d} {np.mean([len(item.support) for item in r]):13.2f} "
              f"{(np.mean(selected_rates) if selected_rates else float('nan')):11.3f} "
              f"{(np.mean(control_rates) if control_rates else float('nan')):11.3f}")
    print("\nThe magnitude-only control is illustrative; it is not a replication of the paper baseline.")
    print("No data from the held-out variants was used to select supports or directions.")


if __name__ == "__main__":
    main()
