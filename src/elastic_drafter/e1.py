"""E1: measure the starting drafter.

    python -m elastic_drafter.e1 --target llama
    python -m elastic_drafter.e1 --target qwen3

Generates the target's responses if they are not saved yet, then measures the drafter's acceptance
length at draft lengths 5 and 8 on HumanEval and MT-Bench, and writes one folder under outputs/.
"""
import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from . import acceptance, targets
from .drafter import Drafter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=targets.MODELS, required=True)
    ap.add_argument("--benchmarks", nargs="+", default=["humaneval", "mtbench"])
    ap.add_argument("--k", nargs="+", type=int, default=[5, 8])
    ap.add_argument("--limit", type=int, default=None, help="first N records only, for a quick try")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    run = Path("outputs") / f"{time.strftime('%Y%m%d-%H%M%S')}_e1_{args.target}_{commit}"
    run.mkdir(parents=True)
    drafter_name = targets.MODELS[args.target][1]
    drafter = Drafter(drafter_name)
    results = {}
    for bench in args.benchmarks:
        targets.generate(args.target, bench)
        records = targets.load(args.target, bench)[:args.limit]
        for k in args.k:
            by_item = defaultdict(list)
            with (run / f"steps_{bench}_k{k}.jsonl").open("w") as f:
                for r in records:
                    steps = acceptance.walk(drafter, r["prompt_ids"], r["response_ids"], k)
                    by_item[r["id"]].extend(steps)
                    f.write(json.dumps({"id": r["id"], "turn": r["turn"], "steps": steps}) + "\n")
            summary = acceptance.summarize([s for steps in by_item.values() for s in steps], k)
            summary["ci95"] = acceptance.bootstrap(list(by_item.values()))
            results[f"{bench}_k{k}"] = summary
            print(f"{bench} k={k}: acceptance length {summary['acceptance_length']:.3f} "
                  f"{summary['ci95']}, per position {summary['per_position']}")

    meta = json.loads(targets.path(args.target, args.benchmarks[0]).with_suffix(".meta.json").read_text())
    manifest = {"experiment": "e1", "commit": commit, "target": meta["target"],
                "drafter": drafter_name, "mode": "target-independent", "args": vars(args),
                "results": results}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
