#!/usr/bin/env python
"""benchmarks/results/*.json -> the README's speed and training tables.

    python benchmarks/make_tables.py benchmarks/results/<inference result>.json
    python benchmarks/make_tables.py benchmarks/results/train_<...>.json
"""

from __future__ import annotations

import json
import sys

NAMES = {"gsxa": "GSXA", "gated": "Gated Attention", "xsa": "XSA",
         "scalable_gqa": "Scalable GQA", "gqa": "GQA"}


def _ms(r):
    return f"{r['ms']:.2f} ± {r['ms_ci95']:.2f}"


def _rel(r):
    return (f"{r['vs_gqa_pct']:+.2f} ± {r['vs_gqa_pct_ci95']:.2f} %"
            if "vs_gqa_pct" in r else "—")


def _table(types, lines):
    print("| | " + " | ".join(NAMES[t] for t in types) + " |")
    print("|:--|" + "--:|" * len(types))
    for label, cells in lines:
        print(f"| {label} | " + " | ".join(cells) + " |")


def inference(env, rows, types):
    print(f"_{env['device']}, torch {env['torch']} (CUDA {env['cuda']}), {env['dtype']}, "
          f"{env['mode']}, {env['shape']}; {env['rounds']} rounds x median of "
          f"{env['iters']} forwards; mean ± 95% CI over rounds._\n")
    _table(types, [("latency (ms)", [_ms(rows[t]) for t in types]),
                   ("vs GQA", [_rel(rows[t]) for t in types])])


def training(env, rows, types):
    print(f"_{env['device']}, torch {env['torch']} (CUDA {env['cuda']}), "
          f"{env['precision']}, {env['mode']}, {env['shape']}; {env['rounds']} rounds x "
          f"median of {env['iters']} steps; mean ± 95% CI over rounds._\n")
    lines = []
    if "fwd" in rows[types[0]]:
        for part, label in (("fwd", "forward"), ("bwd", "backward"),
                            ("step", "forward + backward")):
            lines += [(f"{label} (ms)", [_ms(rows[t][part]) for t in types]),
                      (f"{label} vs GQA", [_rel(rows[t][part]) for t in types])]
    if "memory" in rows[types[0]]:
        gib = lambda t, k: f"{rows[t]['memory'][k] / 1024:.2f}"  # noqa: E731
        ref = rows.get("gqa", {}).get("memory")
        lines += [("peak memory, forward (GiB)", [gib(t, "fwd_alloc_mib") for t in types]),
                  ("peak memory, backward (GiB)", [gib(t, "bwd_alloc_mib") for t in types]),
                  ("peak reserved (GiB)", [gib(t, "peak_reserved_mib") for t in types])]
        if ref:
            lines.append(("peak memory vs GQA", [
                f"{100 * (rows[t]['memory']['peak_alloc_mib'] / ref['peak_alloc_mib'] - 1):+.1f} %"
                if t != "gqa" else "—" for t in types]))
    _table(types, lines)


def main(path):
    data = json.loads(open(path, encoding="utf-8").read())
    env, rows = data["env"], {r["type"]: r for r in data["rows"]}
    types = [t for t in NAMES if t in rows]
    (training if env.get("benchmark") == "training" else inference)(env, rows, types)


if __name__ == "__main__":
    main(sys.argv[1])
