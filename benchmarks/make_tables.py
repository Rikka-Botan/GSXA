#!/usr/bin/env python
"""benchmarks/results/*.json -> the README's speed table.

    python benchmarks/make_tables.py benchmarks/results/cuda.json
"""

from __future__ import annotations

import json
import sys

NAMES = {"gsxa": "GSXA", "gated": "Gated Attention", "xsa": "XSA",
         "scalable_gqa": "Scalable GQA", "gqa": "GQA"}


def main(path):
    data = json.loads(open(path, encoding="utf-8").read())
    env, rows = data["env"], {r["type"]: r for r in data["rows"]}
    types = [t for t in NAMES if t in rows]
    print(f"_{env['device']}, torch {env['torch']} (CUDA {env['cuda']}), {env['dtype']}, "
          f"{env['mode']}, {env['shape']}; {env['rounds']} rounds x median of "
          f"{env['iters']} forwards; mean ± 95% CI over rounds._\n")
    print("| | " + " | ".join(NAMES[t] for t in types) + " |")
    print("|:--|" + "--:|" * len(types))
    print("| latency (ms) | " + " | ".join(
        f"{rows[t]['ms']:.2f} ± {rows[t]['ms_ci95']:.2f}" for t in types) + " |")
    print("| vs GQA | " + " | ".join(
        f"{rows[t]['vs_gqa_pct']:+.2f} ± {rows[t]['vs_gqa_pct_ci95']:.2f} %"
        if "vs_gqa_pct" in rows[t] else "—" for t in types) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
