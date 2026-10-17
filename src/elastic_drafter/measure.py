"""Measure a drafter's acceptance on the saved target responses.

    python -m elastic_drafter.measure --target llama
    python -m elastic_drafter.measure --target qwen3

Generates the target's responses if they are not saved yet, then measures the drafter's acceptance
length at draft lengths 5 and 8 on HumanEval and MT-Bench, and writes one folder under outputs/. Started again after it is stopped, it continues in the same
folder and skips the responses already measured.
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from . import acceptance, data, targets
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
    run = Path("outputs") / (f"measure_{args.target}_{commit}" + (f"_limit{args.limit}" if args.limit else ""))
    run.mkdir(parents=True, exist_ok=True)
    drafter_name = targets.MODELS[args.target][1]
    drafter = Drafter(drafter_name)
    results = {}
    for bench in args.benchmarks:
        targets.generate(args.target, bench)
        records = targets.load(args.target, bench)[:args.limit]
        for k in args.k:
            by_item = defaultdict(list)
            steps_file = run / f"steps_{bench}_k{k}.jsonl"
            measured = set()
            if steps_file.exists():
                for row in data.read_complete_lines(steps_file):
                    measured.add((row["id"], row["turn"]))
                    by_item[row["id"]].extend(tuple(s) for s in row["steps"])
            with steps_file.open("a") as f:
                for r in records:
                    if (r["id"], r["turn"]) in measured:
                        continue
                    steps = acceptance.walk(drafter, r["prompt_ids"], r["response_ids"], k)
                    by_item[r["id"]].extend(steps)
                    f.write(json.dumps({"id": r["id"], "turn": r["turn"], "steps": steps}) + "\n")
                    f.flush()
            summary = acceptance.summarize([s for steps in by_item.values() for s in steps], k)
            summary["ci95"] = acceptance.bootstrap(list(by_item.values()))
            results[f"{bench}_k{k}"] = summary
            print(f"{bench} k={k}: acceptance length {summary['acceptance_length']:.3f} "
                  f"{summary['ci95']}, per position {summary['per_position']}")

    meta = json.loads(targets.path(args.target, args.benchmarks[0]).with_suffix(".meta.json").read_text())
    manifest = {"commit": commit, "target": meta["target"],
                "drafter": drafter_name, "mode": "target-independent", "args": vars(args),
                "results": results}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
