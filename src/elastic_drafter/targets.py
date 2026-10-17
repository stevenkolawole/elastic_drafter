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


def path(target, benchmark, limit=None):
    return Path("data/targets") / target / (benchmark + (f"_limit{limit}" if limit else "") + ".jsonl")


def load(target, benchmark, limit=None):
    """Saved records, one per (item, turn): id, turn, prompt_ids, response_ids, response_text."""
    return data.read_jsonl(path(target, benchmark, limit))


def generate(target, benchmark, limit=None, max_new_tokens=MAX_NEW_TOKENS, batch_size=16):
    """Greedy responses of the target, in bf16, to every turn of the first `limit` items (all if
    None). Later turns see the target's own earlier replies. Records are appended batch by batch, so
    a stopped run continues where it left off; the `.meta.json` file marks a finished one."""
    out = path(target, benchmark, limit)
    meta = out.with_suffix(".meta.json")
    if meta.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["turn"]): r for r in data.read_jsonl(out)} if out.exists() else {}
    name = MODELS[target][0]
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto").eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    items = data.load(benchmark)[:limit]
    with out.open("a") as f:
        f.write("\n")
        for turn in range(max(len(item.turns) for item in items)):
            todo = [item for item in items if turn < len(item.turns) and (item.id, turn) not in done]
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                prompts = [data.chat_ids(tok, item.turns[:turn + 1],
                                         [done[(item.id, t)]["response_text"] for t in range(turn)])
                           for item in batch]
                enc = tok.pad({"input_ids": prompts}, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens)
                for item, prompt, row in zip(batch, prompts, gen[:, enc["input_ids"].shape[1]:].tolist()):
                    resp = row[:next((j + 1 for j, t in enumerate(row) if t in eos), len(row))]
                    r = {"id": item.id, "turn": turn, "prompt_ids": prompt, "response_ids": resp,
                         "response_text": tok.decode(resp, skip_special_tokens=True)}
                    done[(item.id, turn)] = r
                    f.write(json.dumps(r) + "\n")
                f.flush()
    meta.write_text(json.dumps({"target": name, "items_hash": data.items_hash(items),
                                "max_new_tokens": max_new_tokens, "decoding": "greedy",
                                "dtype": "bfloat16"}, indent=2))
