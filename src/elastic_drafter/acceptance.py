"""Measure acceptance offline, on saved target responses.

Real speculative decoding with greedy verification works step by step: the drafter proposes k
tokens, the target keeps them up to the first one it disagrees with, and adds one token of its own.
Because the target is greedy, its choices are exactly the saved response, so we can replay that
loop without running the target: at each step the drafter sees the true text so far, we count how
many of its k tokens match the response in a row (a), and move forward a + 1 tokens. The acceptance
length is the average number of tokens moved forward per step, as PARD reports it.
"""
import numpy as np


def walk(drafter, prompt_ids, response_ids, k):
    """One response. Returns one (accepted, compared, advanced) triple per step: how many draft tokens
    matched in a row, how many could be compared (fewer than k at the end of the response), and how
    many tokens the step moved forward (accepted + 1, or fewer at the end)."""
    steps, pos = [], 0
    while pos < len(response_ids):
        draft = drafter.draft(list(prompt_ids) + list(response_ids[:pos]), k)
        truth = response_ids[pos:pos + k]
        a = 0
        while a < len(truth) and draft[a] == truth[a]:
            a += 1
        advanced = min(a + 1, len(response_ids) - pos)
        steps.append((a, len(truth), advanced))
        pos += advanced
    return steps


def summarize(steps, k):
    """Acceptance length (mean tokens advanced per step) and the acceptance at each draft position:
    of the steps that reach position i and can compare it, the fraction whose i-th token is kept."""
    a, n, adv = (np.array(x) for x in zip(*steps))
    per_position = []
    for i in range(k):
        reached = (a >= i) & (n > i)
        per_position.append(round(float((a[reached] >= i + 1).mean()), 4) if reached.any() else None)
    return {"acceptance_length": float(adv.mean()), "per_position": per_position, "steps": len(a)}


def bootstrap(per_item_steps, n=2000, seed=0):
    """95% interval of the acceptance length, resampling items (all turns of an item together)."""
    rng = np.random.default_rng(seed)
    groups = [np.array([s[2] for s in steps]) for steps in per_item_steps]
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
