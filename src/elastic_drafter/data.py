"""Benchmark prompts with stable IDs, and the chat format the drafters were evaluated with."""
import hashlib
import json
from dataclasses import dataclass

from datasets import load_dataset

# PARD evaluates with this system prompt and thinking disabled
# (https://github.com/AMD-AIG-AIMA/PARD, pard/pard2_infer.py, get_input).
SYSTEM = "You are a helpful assistant."


@dataclass(frozen=True)
class Item:
    id: str        # "<benchmark>:<original id>", the same in every run
    turns: tuple   # the user's messages, in order


def humaneval():
    """All 164 problems of HumanEval (https://arxiv.org/abs/2107.03374), with the instruction PARD
    puts before each one (PARD datas/bmk/humaneval.jsonl, which holds 80 of the 164)."""
    rows = load_dataset("openai/openai_humaneval", split="test")
    return [Item(f"humaneval:{r['task_id']}", ("Complete the code I provided.\n\n" + r["prompt"],))
            for r in rows]


def mtbench():
    """The 80 two-turn questions of MT-Bench (https://arxiv.org/abs/2306.05685)."""
    rows = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    return [Item(f"mtbench:{r['prompt_id']}", tuple(r["prompt"])) for r in rows]


BENCHMARKS = {"humaneval": humaneval, "mtbench": mtbench}


def load(benchmark):
    return BENCHMARKS[benchmark]()


def items_hash(items):
    """A short hash of the IDs and prompts; two runs are comparable only if their hashes match."""
    blob = json.dumps([[i.id, list(i.turns)] for i in items]).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def chat_ids(tokenizer, turns, responses=()):
    """Token IDs of a conversation up to the start of the next assistant reply.

    `turns` are the user messages so far and `responses` the assistant's earlier replies, one fewer
    than `turns`: the second turn of an MT-Bench question is chat_ids(tok, item.turns, [first_reply]).
    """
    assert len(responses) == len(turns) - 1
    messages = [{"role": "system", "content": SYSTEM}]
    for i, turn in enumerate(turns):
        messages.append({"role": "user", "content": turn})
        if i < len(responses):
            messages.append({"role": "assistant", "content": responses[i]})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return tokenizer(text, add_special_tokens=False).input_ids
