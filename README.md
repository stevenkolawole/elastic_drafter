# elastic_drafter

One speculative drafter, many sizes. The drafter is a standalone parallel drafter (it fills a
block of mask tokens in one forward pass); we factor its weights so that keeping the first *r*
components gives a smaller drafter inside the larger one, and we choose which components come
first by what the target accepts. What we test and why is in the scope doc (ask Steven for the
link). This README is how to run things and what is left to do.

## TL;DR

1. **Get access** to the GPU machine (uk-p5): send Steven your SSH public key (the one line from
   `cat ~/.ssh/id_ed25519.pub`) and your Andrew ID. Your GPUs and data folder are in the task board below.
2. **Install** (on uk-p5): `git clone <repo>`, then in the folder `uv venv && source .venv/bin/activate && uv pip install -e .`
3. **Start with T1** below, then take tasks in order unless Steven says otherwise.

## Task board

Status: `todo`, `doing`, `done`, `blocked`. Every task has a check that passes or fails; a task is
done when its check passes and its output is where the table says. Items marked (proposed) wait
for Steven's decision.

| ID | Task | Owner | Done when | Output |
|---|---|---|---|---|
| T0 | Access and environment on uk-p5 | student | `python -c "import elastic_drafter, torch; print(torch.cuda.device_count())"` prints your GPU count | |
| T1 | Pick the drafter for each target (proposed: a released PARD-family drafter if one exists for Llama-3.1-8B and Qwen3-8B, otherwise train one with the Janus data pipeline) | Steven decides; student surveys | a table of candidate drafters with links, sizes and reported acceptance | `docs/drafters.md` |
| T2 | Port the data and evaluation pieces from Janus: target-generated responses, mask-token blocks, the teacher-forced acceptance-length walk (scored over the first 1,024 generated tokens), the paired bootstrap | student, Steven reviews | the chosen drafter's acceptance length on HumanEval and MT-Bench at draft length 5 is stable across two runs to 0.01, and matches its published value where one exists | `src/elastic_drafter/eval.py`, a run folder |
| T3 | Whitened SVD of every linear layer at a given rank fraction ([SVD-LLM](https://arxiv.org/abs/2403.07378)) | student | at full rank the factored drafter reproduces the original logits to 1e-3; at fractions 0.75, 0.5, 0.25 it loads and runs | `src/elastic_drafter/decompose.py` |
| T4 | Loss-weighted importance: weight the decomposition by squared gradients of the drafting loss ([FWSVD](https://arxiv.org/abs/2207.00112)) on a small calibration set | student | weights computed for every linear layer; the full-rank check of T3 still passes | same |
| T5 | Acceptance-weighted importance: the same construction with gradients of D-PACE's expected-accepted-length surrogate | Steven and student | as T4 | same |
| T6 | The rank ladder: 3 weightings (T3, T4, T5) by 3 rank fractions, plus the uncompressed drafter, on frozen item lists and frozen target references | student | 10 run folders; one table and one plot of acceptance length against rank fraction | `outputs/runs/`, `docs/results/` |
| T7 | Matched-compute baseline: the drafter with layers dropped to the same FLOPs as each rank fraction | student | as T6, 3 more points on the same plot | same |
| T8 | Literature cards for the 11 papers in the scope doc: what each reads, where it acts, its numbers, and how we differ; every claim with a page or section | student | 11 cards, checked by Steven | `docs/literature/` |

**Before T6 runs:** write the analysis plan into T6 (the prediction: acceptance weighting beats
loss weighting by more as the rank shrinks; the test; what result would falsify it), and Steven signs it.

**Checkpoint after T6 and T7:** if the prediction holds, the next step is nested multi-rank
training ([Nested Subspace Networks](https://arxiv.org/abs/2509.17874) as the substrate); if it
does not, we decide with the numbers in hand.

## Layout

```
src/elastic_drafter/   the package (data, eval, decompose, ...)
configs/               one YAML per experiment
scripts/               entry points that write run folders
docs/                  drafters, literature cards, results notes
outputs/, data/        git-ignored: runs, downloads, checkpoints
```

Each run writes `outputs/runs/<date>_<kind>_<short hash>/` with `manifest.json` (commit, config,
seed, model IDs and revisions, item-list hash, start and end time), per-item records, and `derived/`
tables that scripts regenerate.
