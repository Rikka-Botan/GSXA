#!/usr/bin/env python
"""benchmarks/results/*.json -> the README's latency tables (one per mode).

    python benchmarks/make_tables.py benchmarks/results/cuda.json
"""

from __future__ import annotations

import json
import sys

NAMES = {"gsxa": "GSXA", "gated": "Gated Attention", "xsa": "XSA",
         "scalable_gqa": "Scalable GQA", "gqa": "GQA"}


def main(path):
    data = json.loads(open(path, encoding="utf-8").read())
    env, rows = data["env"], data["rows"]
    print(f"_{env['device']}, torch {env['torch']} (CUDA {env['cuda']}), {env['dtype']}, "
          f"median of {env['iters']} forwards after {env['warmup']} warm-up._\n")
    cell = {(r["type"], r["mode"], r["shape"]): r for r in rows}
    types = [t for t in NAMES if any(r["type"] == t for r in rows)]
    shapes = list(dict.fromkeys(r["shape"] for r in rows))
    for mode in dict.fromkeys(r["mode"] for r in rows):
        title = "`torch.compile`" if mode == "compile" else "eager"
        print(f"Latency (ms) and throughput (k tokens/s), {title}. Last column: GSXA vs GQA.\n")
        print("| batch x length | " + " | ".join(NAMES[t] for t in types) + " | GSXA overhead |")
        print("|---|" + "---:|" * (len(types) + 1))
        for s in shapes:
            cells = []
            for t in types:
                r = cell.get((t, mode, s))
                cells.append(f"{r['ms']:.2f} ({r['ktok_s']:.0f}k)" if r else "--")
            g, b = cell.get(("gsxa", mode, s)), cell.get(("gqa", mode, s))
            over = f"{100 * (g['ms'] / b['ms'] - 1):+.1f}%" if g and b else "--"
            print(f"| {s.replace('x', ' x ')} | " + " | ".join(cells) + f" | {over} |")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
