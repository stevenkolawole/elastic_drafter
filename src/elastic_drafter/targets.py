"""Generate each target's greedy responses once and save them; every experiment reuses them.

`MODELS` maps a target name to (target model, its PARD-2 drafter). Responses are capped at
`MAX_NEW_TOKENS`, the length they are scored over."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data

MODELS = {
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "amd/PARD2-Llama-3.1-8B"),
    "qwen3": ("Qwen/Qwen3-8B", "amd/PARD2-Qwen3-8B"),
}
MAX_NEW_TOKENS = 1024


def path(target, benchmark):
    return Path("data/targets") / target / f"{benchmark}.jsonl"


def load(target, benchmark):
    """Saved records, one per (item, turn): id, turn, prompt_ids, response_ids."""
    return [json.loads(line) for line in path(target, benchmark).open()]


@torch.no_grad()
def _greedy(model, tok, prompts, max_new_tokens, batch_size):
    """Greedy continuations of token-ID prompts, left-padded into batches and each cut after its first
    end-of-sequence token."""
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    out = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        width = max(len(p) for p in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(batch):
            ids[i, width - len(p):] = torch.tensor(p)
            mask[i, width - len(p):] = 1
        gen = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                             do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        for row in gen[:, width:].tolist():
            cut = next((j + 1 for j, t in enumerate(row) if t in eos), len(row))
            out.append(row[:cut])
    return out


def generate(target, benchmark, max_new_tokens=MAX_NEW_TOKENS, batch_size=16, limit=None):
    """Greedy responses of the target to every turn of every item, in bf16. Later turns of a
    conversation see the target's own earlier replies. Each batch is appended to a partial file as
    soon as it finishes, so a run that is killed (for example by preemption) resumes where it stopped
    when started again. Skips the work if the finished file exists."""
    out = path(target, benchmark)
    if out.exists():
        return out
    part = out.with_suffix(".partial.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if part.exists():
        for r in data.read_complete_lines(part):
            done[(r["id"], r["turn"])] = r
    name = MODELS[target][0]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto").eval()
    items = data.load(benchmark)[:limit]
    with part.open("a") as f:
        for turn in range(max(len(item.turns) for item in items)):
            todo = [item for item in items if turn < len(item.turns) and (item.id, turn) not in done]
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                prompts = [data.chat_ids(tok, item.turns[:turn + 1],
                                         [done[(item.id, t)]["response_text"] for t in range(turn)])
                           for item in batch]
                for item, prompt, resp in zip(batch, prompts, _greedy(model, tok, prompts, max_new_tokens, len(batch))):
                    r = {"id": item.id, "turn": turn, "prompt_ids": prompt, "response_ids": resp,
                         "response_text": tok.decode(resp, skip_special_tokens=True)}
                    done[(item.id, turn)] = r
                    f.write(json.dumps(r) + "\n")
                f.flush()
    order = {item.id: n for n, item in enumerate(items)}
    with out.open("w") as f:
        for r in sorted(done.values(), key=lambda r: (r["turn"], order[r["id"]])):
            f.write(json.dumps(r) + "\n")
    part.unlink()
    meta = {"target": name, "benchmark": benchmark, "items_hash": data.items_hash(items),
            "max_new_tokens": max_new_tokens, "decoding": "greedy", "dtype": "bfloat16"}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return out
