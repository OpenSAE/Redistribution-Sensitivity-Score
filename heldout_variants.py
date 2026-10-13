"""Fresh, hand-authored demo variants, excluded from RSS selection.

`demo.py` retains its original five selection variants for each setting.
This file provides ten different texts for each setting. These are small,
illustrative public examples, NOT the exact paper's held-out operator draws.
"""

EXTRA_FACTS = (
    "Saturn has an extensive ring system.",
    "The Pacific Ocean is larger than the Atlantic Ocean.",
    "A leap year ordinarily contains 366 days.",
    "Venus rotates in the opposite direction to Earth.",
    "Antarctica is the coldest continent.",
    "The chemical symbol for gold is Au.",
    "The Amazon River flows through South America.",
    "A hexagon has six sides.",
    "Most human adults have 206 bones.",
    "The speed of light in a vacuum is constant.",
)
EXTRA_DISTRACTIONS = (
    "A calendar reminder about a dentist appointment can be handled later.",
    "There is an unrelated question about bike routes; leave it for later.",
    "A note about changing a light bulb is unrelated to this request.",
    "A shopping reminder about bread is for a different conversation.",
    "There is also a request to name a pet; do not address it.",
    "Someone asked about train tickets, but that is a separate task.",
    "A reminder to water the plants should not affect this answer.",
    "Please leave the unrelated grocery list for another time.",
    "A separate question about a movie can be ignored for now.",
    "There is a note about a library book due date; it is not relevant.",
)
EXTRA_MODES = (
    "Give a concise one-sentence answer:",
    "Explain this with simple everyday vocabulary:",
    "Give a polite and formal explanation:",
    "Use a short numbered list:",
    "Keep your response very short:",
    "Answer in two brief sentences:",
    "Use wording that a middle-school student can follow:",
    "Please respond in a professional tone:",
    "Present the explanation as bullet points:",
    "Keep the explanation compact and direct:",
)
EXTRA_WRAPS = (
    "Question: {}\nAnswer:",
    "### User question\n{}\n### Assistant answer",
    '{{"user_request": "{}"}}',
    "<request>{}</request>",
    "User input: {}\nAssistant output:",
    "User asks: {}\nAssistant replies:",
    "## Prompt\n{}\n## Completion",
    '{{"prompt": "{}"}}',
    "<instruction>{}</instruction>",
    "Human says: {}\nAI says:",
)
EXTRA_MARKS = ("..", "?!", " --", ",,", "!!", "??", ";;", "...!", " ::", "!?",)
EXTRA_VERBS = (
    "go over", "break down", "walk through", "provide an overview of", "help me understand",
    "elaborate on", "talk about", "explore", "detail", "review",
)

SETTINGS = (
    "Irrelevant context", "Distractor context", "Metalinguistic wrapper",
    "Format wrapper", "Punctuation noise", "Typed substitution",
)


def heldout_variants(opening, verb, target, setting):
    """Ten variants for a setting, distinct from the original demo's five.

    The target phrase is preserved verbatim, so demo.sae_features can align the
    final overlapping token. Validity and disjointness are checked by the caller.
    """
    core = f"{opening} {verb} {target}"
    if setting == "Irrelevant context":
        return [f"{fact} {core}" for fact in EXTRA_FACTS]
    if setting == "Distractor context":
        return [f"{task} {core}" for task in EXTRA_DISTRACTIONS]
    if setting == "Metalinguistic wrapper":
        return [f"{mode} {core}" for mode in EXTRA_MODES]
    if setting == "Format wrapper":
        return [wrap.format(core) for wrap in EXTRA_WRAPS]
    if setting == "Punctuation noise":
        return [f"{opening}{mark} {verb} {target}" for mark in EXTRA_MARKS]
    if setting == "Typed substitution":
        return [f"{opening} {alternate} {target}" for alternate in EXTRA_VERBS]
    raise ValueError(f"unknown transformation {setting!r}")


def ensure_disjoint(clean_text, selection_texts, heldout_texts, target):
    """Fail rather than silently leak variants or change the alignment target."""
    s, h = list(selection_texts), list(heldout_texts)
    if len(s) != 5 or len(h) != 10:
        raise ValueError(f"expected 5 selection and 10 held-out variants, got {len(s)} and {len(h)}")
    if len(set(s)) != len(s) or len(set(h)) != len(h):
        raise ValueError("duplicate variants within a split")
    if set(s) & set(h) or clean_text in set(s) | set(h):
        raise ValueError("overlap between clean, selection, and held-out prompt texts")
    if not target or any(x.count(target) != 1 for x in [clean_text, *s, *h]):
        raise ValueError("target phrase must occur exactly once in every prompt")
