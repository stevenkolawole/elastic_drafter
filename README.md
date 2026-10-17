# elastic_drafter

In speculative decoding, a small model (the *drafter*) proposes the next few tokens, and the large
model we want answers from (the *target*) checks all of them in one forward pass. The number of
tokens proposed per check is the *draft length*, for example 5. The target keeps the proposed
tokens up to the first one it disagrees with, so the output is exactly what the target would have
written alone. The speedup depends on the *acceptance length*: the average number of tokens the
target keeps per check.

The right size of drafter depends on how busy the server is. When many requests share the GPUs,
the target's checks dominate the time and a cheap drafter pays off; when few requests are running,
a larger and more accurate drafter is worth its cost (the scope doc gives the evidence). We want
one drafter that can be made cheaper at deployment. Any weight matrix can be written as a sum of
rank-one components (its singular value decomposition, SVD), and keeping only *r* of them gives a
cheaper matrix. The open question is which *r* components to keep. Existing methods keep the
largest components, or the ones the drafter's own training loss depends on most. We will test a
third choice: keep the components that the target's acceptance depends on most.

## Setup

On uk-p5, our GPU machine, in a clone of this repo:

```
uv venv
source .venv/bin/activate
uv pip install -e ".[gen]"      # gen adds vLLM, used once per target to generate its responses
```

Run E1 for each target (the first run generates and saves the target's responses under `data/`):

```
python -m elastic_drafter.e1 --target llama
python -m elastic_drafter.e1 --target qwen3
python -m elastic_drafter.e1 --target llama --limit 5    # a quick try on 5 prompts
```

Each run prints the acceptance lengths and writes a folder under `outputs/` with the per-step
records and a `manifest.json` (commit, models, settings, results).

## Choices shared by every experiment (ours)

- **Targets:** [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  and [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
- **Drafters:** the released [PARD-2](https://arxiv.org/abs/2605.08632) drafters for these two
  targets, [PARD2-Llama-3.1-8B](https://huggingface.co/amd/PARD2-Llama-3.1-8B) (1B parameters)
  and [PARD2-Qwen3-8B](https://huggingface.co/amd/PARD2-Qwen3-8B) (0.6B). They are parallel
  drafters: they propose all the draft tokens in one forward pass, by filling a block of
  placeholder ("mask") tokens appended to the text.
- **Benchmarks:** [HumanEval](https://arxiv.org/abs/2107.03374) (164 Python programming problems)
  and [MT-Bench](https://arxiv.org/abs/2306.05685) (80 two-turn conversations), all prompts.
- **Target responses:** generated once with greedy decoding in bf16 and saved; every experiment
  reuses the same saved prompts and responses, so the numbers are comparable.
- **Measuring acceptance:** offline, on the saved responses. Starting at the beginning of a
  response, the drafter proposes the next tokens given the true text so far; we count how many
  match the target's tokens in a row, move past them and the target's next token, and repeat, as
  real decoding would. Responses are scored over their first 1,024 generated tokens.
- **Calibration set:** 32 conversations from the drafters' training-data sources, never from the
  benchmarks, each with mask tokens appended as at drafting time.
- **Cost:** the multiply-adds of the linear layers inside the drafter's transformer blocks. The
  embedding table and the output layer stay as they are. No experiment trains the drafter after
  shrinking it.

## Experiments

E1 measures the starting point. E2 to E5 shrink the drafter in the same way and differ only in
which components they keep. E6 shrinks it by another route, at the same cost as E2 to E5.

**E1. Measure the starting drafters.**
Generate and save the target responses, then measure each drafter's acceptance length at draft
lengths 5 and 8, and its acceptance at each position of the draft (how often the 1st, 2nd, ...
proposed token is kept). Before trusting the numbers, check on 20 prompts that `drafter.py`
proposes exactly the tokens AMD's own code does (`pard/pard2_infer.py` in the
[PARD repo](https://github.com/AMD-AIG-AIMA/PARD), target-independent mode).

- **Why:** every later experiment asks how much of this acceptance survives in a cheaper drafter.
- **Useful output:** acceptance length per target, benchmark and draft length, compared with what the
PARD-2 paper reports for the same drafters; the saved responses that the other experiments reuse.

**E2. Plain SVD.**
Replace every linear layer's weight matrix with its *r* largest SVD components, stored as two
thin matrices, and measure acceptance length as in E1. An *m*-by-*n* matrix kept at rank *r*
costs *r(m+n)* multiply-adds instead of *mn*, so an *n*-by-*n* matrix is cheaper only for *r* below
*n*/2. We choose *r* for each matrix so that it keeps 75%, 50% or 25% of its original cost.

- **Why:** this shows how much acceptance depends on the number of components kept. If acceptance
barely drops at 25%, the choice of components matters little; if it collapses already at 75%, the
range worth studying is small.
- **Useful output:** a plot of acceptance length against the fraction of cost kept, one line per
target. E3 to E6 add to this plot.

**E3. Whitened SVD.**
Before the SVD, multiply each matrix by a transform computed from the inputs it receives on the
calibration set, so that directions the inputs use often count for more; truncate; then undo the
transform. This is [SVD-LLM](https://arxiv.org/abs/2403.07378).

- **Why:** plain truncation works poorly on language models (see SVD-LLM), so beating it would
prove little. Whitened SVD is the fair baseline.
- **Useful output:** a second line on the plot.

**E4. Loss-weighted SVD.**
The same procedure as E3, with a different scaling: before the SVD, scale each row of a matrix by
how strongly the drafter's training loss depends on that row's weights (the summed squared
gradients of the loss on the calibration set); truncate; then undo the scaling. This is
[FWSVD](https://arxiv.org/abs/2207.00112). The drafter's training loss is the cross-entropy of its
predictions at the mask positions against the target's tokens.

- **Why:** this is the strongest existing way to choose components, and the one a new criterion has to
beat.
- **Useful output:** a third line on the plot.

**E5. Acceptance-weighted SVD.**
The same as E4, with the gradients of the expected acceptance length in place of the training
loss. The expected acceptance length counts, for each draft position, the probability that the
drafter's token there and at every earlier position matches the target's; it is differentiable,
and [D-PACE](https://arxiv.org/abs/2605.18810) gives it in closed form for parallel drafters.

- **Why:** we predict that E5 keeps more acceptance than E4, and that its lead grows as the drafter
gets cheaper: with many components kept, both criteria keep the useful ones, and with few kept,
the criterion decides which ones go.
- **Useful output:** a fourth line on the plot, and the gap between E5 and E4 at each cost.

**E6. Dropping layers.**
Remove whole transformer blocks from the drafter, choosing the number of blocks whose cost is
closest to each point of E2, and measure acceptance length.

- **Why:** dropping layers is the obvious other way to make a drafter cheaper; keeping fewer components
is only worth using if it beats it at the same cost.
- **Useful output:** three points on the plot.

If E5 leads, the next step is training one drafter whose every rank works at once
([Nested Subspace Networks](https://arxiv.org/abs/2509.17874)). If it does not, we decide with the
numbers in hand.

## Code

| File | What it does |
|---|---|
| `data.py` | HumanEval and MT-Bench prompts with stable IDs, in the chat format PARD evaluates with |
| `targets.py` | generates each target's greedy responses once (vLLM) and saves them under `data/targets/` |
| `drafter.py` | loads a PARD-2 drafter and proposes *k* tokens in one forward pass |
| `acceptance.py` | replays speculative decoding on the saved responses: acceptance length, acceptance per position, bootstrap interval |
| `lowrank.py` | replaces linear layers with low-rank factors at a given fraction of cost, with the scalings of E2 to E5; `python -m elastic_drafter.lowrank` runs its checks |
| `e1.py` | E1 end to end |

Files for E2 onward are added when those experiments start. Runs write to `outputs/` and data to
`data/`; neither is in git.
