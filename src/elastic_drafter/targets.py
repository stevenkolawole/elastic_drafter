"""Generate each target's greedy responses once and save them; every experiment reuses them."""
import json
from pathlib import Path

from transformers import AutoTokenizer

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


def generate(target, benchmark):
    """Greedy responses of the target to every turn of every item, in bf16 with vLLM. Later turns of
    a conversation see the target's own earlier replies. Skips the work if the file exists."""
    out = path(target, benchmark)
    if out.exists():
        return out
    from vllm import LLM, SamplingParams  # only this step needs vLLM

    name = MODELS[target][0]
    tok = AutoTokenizer.from_pretrained(name)
    llm = LLM(name, dtype="bfloat16", max_model_len=8192)
    greedy = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)
    items = data.load(benchmark)
    replies = {item.id: [] for item in items}
    records = []
    for turn in range(max(len(item.turns) for item in items)):
        todo = [item for item in items if turn < len(item.turns)]
        prompts = [data.chat_ids(tok, item.turns[:turn + 1], replies[item.id]) for item in todo]
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], greedy)
        for item, prompt, o in zip(todo, prompts, outputs):
            reply = o.outputs[0]
            replies[item.id].append(reply.text)
            records.append({"id": item.id, "turn": turn, "prompt_ids": prompt,
                            "response_ids": list(reply.token_ids)})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    meta = {"target": name, "benchmark": benchmark, "items_hash": data.items_hash(items),
            "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy", "dtype": "bfloat16"}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return out
