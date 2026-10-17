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

The code runs on any Linux or macOS machine with Python 3.12; the experiments need an NVIDIA GPU
(sizes below). Every package version is fixed in `uv.lock`, so every machine gets the same ones.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repo and install the exact versions: `git clone git@github.com:stevenkolawole/elastic_drafter.git`,
   then, in the folder, `uv sync`.
3. Hugging Face: log in once with `uv run hf auth login` (a read token from your Hugging Face
   settings), and request access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   on its page; Qwen3 and the PARD-2 drafters need no request. Models download to `$HF_HOME`
   (default `~/.cache/huggingface`); point it at a disk with ~60 GB free.
4. Check that it runs, on 5 prompts (Qwen3 needs no access request):
   `uv run python -m elastic_drafter.measure --target qwen3 --limit 5`

Then run E1 in full:

```
uv run python -m elastic_drafter.measure --target qwen3
uv run python -m elastic_drafter.measure --target llama
```

The first run per target generates and saves the target's responses under `data/`; later runs
reuse them. Each run prints the acceptance lengths and writes a folder under `outputs/` with the
per-step records and a `manifest.json` (commit, models, settings, results).

### What each step needs

| Step | GPU memory | Example cards | Time, roughly |
|---|---|---|---|
| Code checks (`python -m elastic_drafter.lowrank`, `--limit 5` tries) | none (CPU works) | any laptop | minutes |
| Generating one target's responses (8B model in bf16) | 24 GB (lower `batch_size` in `targets.py` if memory runs out) | L4, A10, RTX 4090, L40S, A6000, A100 | 1 to 3 hours per target |
| Measuring and shrinking the drafter (E1 to E6; 0.6B and 1B models) | 8 to 16 GB | any of the above, or a T4 | under an hour per run |

Generating the responses is the only step that needs a 24 GB card. Once they are saved (and shared,
see Decisions), every experiment up to E6 runs on a 16 GB card.

### On Babel (CMU)

Babel runs everything as batch jobs. GPU jobs go to the `general` partition with `sbatch` (it
rejects `srun` and CPU-only jobs); pin the card type, since `general` mixes several; and keep large
files under `/data/user_data/$USER`, which exists only on compute nodes, not on the login node:

```
sbatch --partition=general --gres=gpu:1 --constraint=L40S --time=08:00:00 --mem=64G \
  --job-name=measure-qwen3 --output=measure-%j.log \
  --wrap="cd ~/elastic_drafter && export HF_HOME=/data/user_data/$USER/hf && \
          uv run python -c 'import torch; assert torch.cuda.is_available()' && \
          uv run python -m elastic_drafter.measure --target qwen3"
```

Check a job with `squeue -u $USER` and read its log with `tail measure-<jobid>.log`. On uk-p5 the
same commands run directly in a terminal.

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

## How E1 works, and what to watch

E1 is built; read the code in this order before running it, since every later experiment reuses it.

1. `data.py`: the prompts. Watch that IDs stay the same between runs (`items_hash`).
2. `targets.py`: the target's responses, generated once and saved. Every comparison uses these
   same saved responses; never regenerate them in the middle of a set of experiments.
3. `drafter.py`: how a PARD drafter proposes *k* tokens in one pass. Watch that its tokens match
   AMD's reference code (the check in E1 below).
4. `acceptance.py`: how a saved response is replayed step by step. Watch the per-position
   acceptance: it should fall from the 1st draft position to the *k*-th.
5. `measure.py`: ties them together and writes the run folder.

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
| `targets.py` | generates each target's greedy responses once (Hugging Face Transformers, bf16) and saves them under `data/targets/` |
| `drafter.py` | loads a PARD-2 drafter and proposes *k* tokens in one forward pass |
| `acceptance.py` | replays speculative decoding on the saved responses: acceptance length, acceptance per position, bootstrap interval |
| `lowrank.py` | replaces linear layers with low-rank factors at a given fraction of cost, with the scalings of E2 to E5; `python -m elastic_drafter.lowrank` runs its checks |
| `measure.py` | measures a drafter's acceptance on the saved responses and writes a run folder (E1; later experiments reuse it on shrunk drafters) |

Files for E2 onward are added when those experiments start. Runs write to `outputs/` and data to
`data/`; neither is in git.

## Decisions

| Question | Options | Now | Status |
|---|---|---|---|
| How to generate the target responses | Hugging Face Transformers, or vLLM | Transformers: the responses are generated once, and nothing before end-to-end speed measurements needs vLLM | decided for now; vLLM returns for speedups at the end |
| Which mode of the PARD-2 drafter | target-independent (tokens only), or target-dependent (adds a projection of the target's hidden states) | target-independent: the cleanest test of which components to keep, since the target's states are not shrunk | decided for now; the headline comparison is repeated target-dependent later |
| The loss behind E4 | plain cross-entropy at the mask positions, or the drafter's own training loss | plain cross-entropy, as FWSVD uses the drafting loss | decided |
| Whether cost counts the output layer and embeddings | only the linear layers in the transformer blocks, or also the output layer (about a quarter of a 1B drafter with a 128K vocabulary) | blocks only | open |
| HumanEval set | all 164 problems, or the 80 in PARD's evaluation file (comparable with their numbers) | all 164 | open |
| Calibration set | which conversations, and how many | 32 from the drafters' training-data sources | open: the exact source |
| Training data for the nested drafter | self-distilled conversations: ShareGPT and UltraChat-200K prompts with each target's own regenerated responses | reuse the conversations already generated by Llama-3.1-8B-Instruct and Qwen3-8B for an earlier project (on uk-p5), shared through the Hugging Face repo | open |
| Where saved target responses and run outputs live | local `data/` and `outputs/` only, or a private Hugging Face repo both of us can pull from | local for now | to do: Steven sets up a private Hugging Face repo for our data |
| Changing the rank per request in a serving engine | | | deferred (mixed ranks in one batch give irregular shapes) |
