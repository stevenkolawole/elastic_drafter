"""Measure a drafter's acceptance on the saved target responses.

    python -m elastic_drafter.measure --target qwen3
    python -m elastic_drafter.measure --target llama --limit 5

Generates the target's responses if they are not saved yet, then measures the drafter's acceptance
length at draft lengths 5 and 8 on HumanEval and MT-Bench, and writes one folder under outputs/.
A stopped run, started again, continues in the same folder.
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
    ap.add_argument("--limit", type=int, default=None, help="first N items only, for a quick try")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    run = Path("outputs") / (f"measure_{args.target}_{commit}" + (f"_limit{args.limit}" if args.limit else ""))
    run.mkdir(parents=True, exist_ok=True)
    target_name, drafter_name = targets.MODELS[args.target]
    drafter = Drafter(drafter_name)
    results = {}
    for bench in args.benchmarks:
        targets.generate(args.target, bench, args.limit)
        records = targets.load(args.target, bench, args.limit)
        for k in args.k:
            steps_file = run / f"steps_{bench}_k{k}.jsonl"
            rows = data.read_jsonl(steps_file) if steps_file.exists() else []
            measured = {(row["id"], row["turn"]) for row in rows}
            with steps_file.open("a") as f:
                f.write("\n")
                for r in records:
                    if (r["id"], r["turn"]) not in measured:
                        row = {"id": r["id"], "turn": r["turn"],
                               "steps": acceptance.walk(drafter, r["prompt_ids"], r["response_ids"], k)}
                        rows.append(row)
                        f.write(json.dumps(row) + "\n")
                        f.flush()
            by_item = defaultdict(list)
            for row in rows:
                by_item[row["id"]].extend(tuple(s) for s in row["steps"])
            summary = acceptance.summarize([s for steps in by_item.values() for s in steps], k)
            summary["ci95"] = acceptance.bootstrap(list(by_item.values()))
            results[f"{bench}_k{k}"] = summary
            print(f"{bench} k={k}: acceptance length {summary['acceptance_length']:.3f} "
                  f"{summary['ci95']}, per position {summary['per_position']}")

    manifest = {"commit": commit, "target": target_name, "drafter": drafter_name,
                "mode": "target-independent", "args": vars(args), "results": results}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
