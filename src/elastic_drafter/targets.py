"""Generate each target's greedy responses once and save them; every experiment reuses them."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data

# target name -> (target model, its PARD-2 drafter)
MODELS = {
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "amd/PARD2-Llama-3.1-8B"),
    "qwen3": ("Qwen/Qwen3-8B", "amd/PARD2-Qwen3-8B"),
}
MAX_NEW_TOKENS = 1024  # responses are scored over their first 1,024 generated tokens


def path(target, benchmark):
    return Path("data/targets") / target / f"{benchmark}.jsonl"


def load(target, benchmark):
    """Saved records, one per (item, turn): id, turn, prompt_ids, response_ids."""
    return [json.loads(line) for line in path(target, benchmark).open()]


@torch.no_grad()
def _greedy(model, tok, prompts, max_new_tokens, batch_size):
    """Greedy continuations of token-ID prompts, each cut after its first end-of-sequence token."""
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    out = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        width = max(len(p) for p in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(batch):  # left padding, so every prompt ends at the same column
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
    conversation see the target's own earlier replies. Skips the work if the file exists."""
    out = path(target, benchmark)
    if out.exists():
        return out
    name = MODELS[target][0]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto").eval()
    items = data.load(benchmark)[:limit]
    replies = {item.id: [] for item in items}
    records = []
    for turn in range(max(len(item.turns) for item in items)):
        todo = [item for item in items if turn < len(item.turns)]
        prompts = [data.chat_ids(tok, item.turns[:turn + 1], replies[item.id]) for item in todo]
        for item, prompt, resp in zip(todo, prompts, _greedy(model, tok, prompts, max_new_tokens, batch_size)):
            replies[item.id].append(tok.decode(resp, skip_special_tokens=True))
            records.append({"id": item.id, "turn": turn, "prompt_ids": prompt, "response_ids": resp})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    meta = {"target": name, "benchmark": benchmark, "items_hash": data.items_hash(items),
            "max_new_tokens": max_new_tokens, "decoding": "greedy", "dtype": "bfloat16"}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return out
