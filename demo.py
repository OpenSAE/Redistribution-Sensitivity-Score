"""RSS demo: GPT-2 Small + a public SAE, 24 prompts, 6 prompt transformations, 5 variants each.

python demo.py                 # layer 8, about 1 minute on a CPU
python demo.py --layer 4       # paper layers for GPT-2 Small: 4, 6, 8, 10
"""
import argparse
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from rss import rss

warnings.filterwarnings("ignore", category=UserWarning)
# (opening, verb, target phrase). The target phrase is never edited; we read the SAE at its last token.
PROMPTS = [
    ("Can you", "explain", "why the sky looks blue during the day?"),
    ("Please", "describe", "how a rainbow forms after a storm."),
    ("Could you", "explain", "how to make a simple tomato sauce?"),
    ("Please", "outline", "the steps for baking banana bread."),
    ("Can you", "describe", "the benefits of drinking enough water?"),
    ("Please", "explain", "why sleep matters for memory."),
    ("Could you", "summarize", "what to pack for a week in Tokyo?"),
    ("Please", "describe", "a relaxing weekend trip to the coast."),
    ("Can you", "summarize", "why the Roman Empire declined?"),
    ("Please", "explain", "how the printing press changed Europe."),
    ("Could you", "explain", "how a password manager keeps data safe?"),
    ("Please", "describe", "how solar panels turn light into power."),
    ("Can you", "explain", "how compound interest grows savings?"),
    ("Please", "outline", "a simple monthly budget for a student."),
    ("Could you", "describe", "the basic rules of volleyball?"),
    ("Please", "explain", "how marathon runners pace themselves."),
    ("Can you", "explain", "how bees make honey?"),
    ("Please", "describe", "why leaves change color in autumn."),
    ("Could you", "describe", "what makes a painting impressionist?"),
    ("Please", "summarize", "the history of jazz music."),
    ("Can you", "outline", "good ways to study for an exam?"),
    ("Please", "explain", "how children learn to read."),
    ("Could you", "describe", "how to remove a coffee stain from a shirt?"),
    ("Please", "explain", "how to keep houseplants healthy."),
]
VERBS = ["explain", "describe", "summarize", "outline", "clarify", "discuss"]
FACTS = ["Octopuses have three hearts.", "The Eiffel Tower was finished in 1889.", "Honey never spoils.",
         "A group of flamingos is called a flamboyance.", "Mount Everest grows a little every year."]
TASKS = ["Also think of a name for a goldfish, but do not write it.", "Someone asked about train times; that can wait.",
         "Keep the shopping list for later.", "There is a spelling quiz to do afterwards.",
         "Ignore the weather report for now."]
MODES = ["Answer in one sentence:", "Explain it to a ten-year-old:", "Reply as a formal letter:",
         "Respond with a bulleted list:", "Answer as briefly as possible:"]
WRAPS = ["Q: {}\nA:", "### Request\n{}\n### Response", '{{"question": "{}"}}', "<user>{}</user>", "Input: {}\nOutput:"]
MARKS = [",", "...", " -", ";", "!"]

# Each setting maps (opening, verb, target) to 5 variants. Context and wrappers: first four; local edits: last two.
SETTINGS = {
    "Irrelevant context": lambda o, v, t: [f"{x} {o} {v} {t}" for x in FACTS],
    "Distractor context": lambda o, v, t: [f"{x} {o} {v} {t}" for x in TASKS],
    "Metalinguistic wrapper": lambda o, v, t: [f"{x} {o} {v} {t}" for x in MODES],
    "Format wrapper": lambda o, v, t: [w.format(f"{o} {v} {t}") for w in WRAPS],
    "Punctuation noise": lambda o, v, t: [f"{o}{m} {v} {t}" for m in MARKS],
    "Typed substitution": lambda o, v, t: [f"{o} {w} {t}" for w in [w for w in VERBS if w != v][:5]],
}
CONTEXT, LOCAL = list(SETTINGS)[:4], list(SETTINGS)[4:]


def sae_features(model, sae, hook, text, target):
    """SAE activations at the last token overlapping the (only) occurrence of the target phrase."""
    start = text.index(target)
    enc = model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    last = max(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < start + len(target) and b > start)
    tokens = torch.tensor([[model.tokenizer.bos_token_id] + enc["input_ids"]], device=model.cfg.device)
    _, cache = model.run_with_cache(tokens, names_filter=hook)
    return sae.encode(cache[hook][0, last + 1]).cpu().numpy()  # +1 for BOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hook = f"blocks.{args.layer}.hook_resid_pre"
    model = HookedTransformer.from_pretrained_no_processing("gpt2", center_writing_weights=True, device=device)
    sae = SAE.from_pretrained("gpt2-small-res-jb", hook, device=device)
    sae = sae[0] if isinstance(sae, tuple) else sae

    mean_rss = {s: [] for s in SETTINGS}
    supports = {s: [] for s in SETTINGS}
    example = None
    with torch.no_grad():
        for o, v, t in PROMPTS:
            a0 = sae_features(model, sae, hook, f"{o} {v} {t}", t)
            for name, make in SETTINGS.items():
                delta = np.stack([sae_features(model, sae, hook, x, t) - a0 for x in make(o, v, t)])
                S, direction, m = rss(delta)
                mean_rss[name].append(m)
                supports[name].append(set(S.tolist()))
                if example is None:
                    example = (f"{o} {v} {t}", name, S[:5], direction[:5], np.abs(delta).mean(0)[S[:5]])

    # 1. Normalized MeanRSS per transformation (the paper's Figure 2, for one SAE).
    means = np.array([np.nanmean(mean_rss[s]) for s in SETTINGS])
    norm = means / means.mean()
    print(f"\nGPT-2 Small, layer {args.layer}, {len(PROMPTS)} prompts\n")
    print(f"{'Transformation':24s} {'MeanRSS':>8s} {'normalized':>11s}")
    for s, m, z in zip(SETTINGS, means, norm):
        print(f"{s:24s} {m:8.3f} {z:11.3f}")
    gap = norm[:4].mean() - norm[4:].mean()
    per_prompt = np.array([[mean_rss[s][i] for s in SETTINGS] for i in range(len(PROMPTS))]) / means.mean()
    wins = int(np.sum(np.nanmean(per_prompt[:, :4], 1) > np.nanmean(per_prompt[:, 4:], 1)))
    print(f"\nContext and wrappers minus local edits: {gap:.3f}")
    print(f"Prompts where context and wrappers move more: {wins} / {len(PROMPTS)}")

    # 2. Which features respond: support overlap between transformations (the paper's Figure S2).
    names = list(SETTINGS)
    jac = np.eye(len(names))
    for i, j in itertools.combinations(range(len(names)), 2):
        jac[i, j] = jac[j, i] = np.mean([len(a & b) / len(a | b) for a, b in zip(supports[names[i]], supports[names[j]])
                                         if a | b])

    text, name, S, d, h = example
    print(f'\nExample: "{text}" under {name.lower()}')
    for f, sign, size in zip(S, d, h):
        print(f"  feature {f:5d}  {'up  ' if sign > 0 else 'down'}  mean |change| {size:.2f}  "
              f"https://www.neuronpedia.org/gpt2-small/{args.layer}-res-jb/{f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    bars = ax[0].bar(range(6), norm, color=["#2c7fb8"] * 4 + ["#f4a259"] * 2)
    ax[0].legend([bars[0], bars[4]], ["Context and wrappers", "Local edits"], frameon=False, fontsize=8)
    ax[0].axhline(1, color="grey", lw=0.8, ls="--")
    ax[0].set_xticks(range(6), [n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax[0].set_ylabel("Normalized MeanRSS")
    ax[0].set_title("A. How much the representation moves")
    im = ax[1].imshow(jac, cmap="Blues", vmin=0, vmax=1)
    short = ["IC", "DC", "MW", "FW", "PN", "TL"]
    ax[1].set_xticks(range(6), short)
    ax[1].set_yticks(range(6), short)
    ax[1].set_title("B. Which features respond (Jaccard)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.suptitle(f"RSS demo: GPT-2 Small layer {args.layer}, {len(PROMPTS)} prompts")
    fig.tight_layout()
    fig.savefig(f"demo_layer{args.layer}.png", dpi=150)
    print(f"\nSaved demo_layer{args.layer}.png")


if __name__ == "__main__":
    main()
