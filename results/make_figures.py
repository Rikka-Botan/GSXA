#!/usr/bin/env python
"""Figures and tables of the attention ablation (GSXA and the four compared models).

Reads the five scoring JSONs in ``results/attention_ablation/`` (masked-LM perplexity,
stored under the key "pppl", and MLM accuracy per sequence length, each the mean and
95% Student-t CI over 20 disjoint
measurements) and writes, for sequence lengths 64 ... 4,096, into ``docs/figures/``:

* ``fig1_bars_summary.{png,pdf}`` -- gains over GQA averaged over the lengths,
* ``fig2_bars_by_length.{png,pdf}`` -- the same gains at each length,
* ``tab_attention_ablation.tex`` -- a booktabs table for a paper,

and with ``--tables`` prints the README's markdown tables.

    python results/make_figures.py            # figures
    python results/make_figures.py --tables   # + tables

The CIs of relative quantities combine the two models' CIs as if they were
independent (delta method). Both models are scored on the same windows and masks,
so this is conservative; the JSONs keep no per-measurement values, so a paired
interval cannot be computed from them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "attention_ablation"
FIGURES = ROOT / "docs" / "figures"
TRAINED_LEN = 1024
LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]

# Categorical slots 1-4 of the reference palette (validated on white, light mode:
# worst adjacent CVD dE 9.1, normal-vision dE 22.9); GQA is the reference.
ARMS = [
    ("at_sf_xsa_gate", "GSXA", "#2a78d6"),
    ("at_sf_gate", "Gated Attention", "#eb6834"),
    ("at_sf_xsa", "XSA", "#1baf7a"),
    ("at_sf", "Scalable GQA", "#eda100"),
    ("at_plain", "GQA", "#52514e"),
]
BASELINE, BASE_NAME = "at_plain", "GQA"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
TEXT_WIDTH = 6.3   # inches: ACL / NeurIPS-style \textwidth


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=str(RESULTS))
    p.add_argument("--out_dir", default=str(FIGURES))
    p.add_argument("--tables", action="store_true", help="print the README tables")
    return p.parse_args(argv)


def load(results_dir: Path):
    data = {}
    for key, *_ in ARMS:
        d = json.loads((results_dir / f"{key}.json").read_text(encoding="utf-8"))
        data[key] = {int(L): r for L, r in d["results"].items() if r}
    return data


def series(data, key, metric):
    scale = 100.0 if metric == "mlm_acc" else 1.0
    rows = data[key]
    return ([rows[L][metric] * scale for L in LENGTHS],
            [rows[L][metric + "_ci95"] * scale for L in LENGTHS])


def relative(data, key, metric):
    """PPL ratio or accuracy difference vs GQA per length, with the conservative CI."""
    a, ca = series(data, key, metric)
    p, cp = series(data, BASELINE, metric)
    if metric == "pppl":   # ratio, CI by the delta method
        r = [x / y for x, y in zip(a, p)]
        e = [v * math.hypot(cx / x, cy / y) for v, x, y, cx, cy in zip(r, a, p, ca, cp)]
    else:                  # accuracy points
        r = [x - y for x, y in zip(a, p)]
        e = [math.hypot(cx, cy) for cx, cy in zip(ca, cp)]
    return r, e


def averages(data, key):
    """Geometric-mean PPL reduction (%) and mean accuracy gain (points) vs GQA."""
    r, _ = relative(data, key, "pppl")
    a, _ = relative(data, key, "mlm_acc")
    return 100.0 * (1.0 - math.exp(sum(math.log(x) for x in r) / len(r))), sum(a) / len(a)


def style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def panel_label(ax, text):
    ax.text(-0.14, 1.03, text, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", color=INK)


def figure_bar_summary(plt, data):
    """Single column: the averaged gains, one horizontal bar per model, GSXA on top."""
    arms = [a for a in ARMS if a[0] != BASELINE]
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH / 2, 2.55), constrained_layout=True)
    rows = list(range(len(arms)))[::-1]
    for ax, which, xlabel, fmt, xmax in (
            (axes[0], 0, f"PPL reduction vs {BASE_NAME} (%)", "{:.1f}%", 14.5),
            (axes[1], 1, f"MLM accuracy gain vs {BASE_NAME} (pt)", "+{:.2f}", 1.6)):
        for row, (key, _label, color) in zip(rows, arms):
            v = averages(data, key)[which]
            ax.barh(row, v, height=0.62, color=color, edgecolor="white", linewidth=0.6,
                    zorder=3)
            ax.text(v + xmax * 0.015, row, fmt.format(v), va="center", ha="left",
                    fontsize=7, color=INK, fontweight="bold" if key == ARMS[0][0] else None)
        ax.set_yticks(rows)
        ax.set_yticklabels([label for _k, label, _c in arms])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_title(f"Averaged over lengths {LENGTHS[0]:,}–{LENGTHS[-1]:,}",
                      fontsize=7.5, color=INK_2, loc="left")
    return fig


def figure_bars(plt, data):
    """Grouped bars: PPL reduction and accuracy gain over GQA at each length."""
    arms = [a for a in ARMS if a[0] != BASELINE]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)
    n, width = len(arms), 0.2
    x = list(range(len(LENGTHS)))
    for ax, metric, ylabel in ((axes[0], "pppl", f"PPL reduction vs {BASE_NAME} (%)"),
                               (axes[1], "mlm_acc", f"MLM accuracy gain vs {BASE_NAME} (points)")):
        for i, (key, label, color) in enumerate(arms):
            r, e = relative(data, key, metric)
            vals, ci = (([100.0 * (1.0 - v) for v in r], [100.0 * c for c in e])
                        if metric == "pppl" else (r, e))
            xs = [p + (i - (n - 1) / 2) * width for p in x]
            # white edge = the surface gap between touching bars
            ax.bar(xs, vals, width=width, color=color, edgecolor="white", linewidth=0.6,
                   label=label, zorder=3)
            ax.errorbar(xs, vals, yerr=ci, fmt="none", ecolor=INK_2, elinewidth=0.6,
                        capsize=1.2, capthick=0.6, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{L:,}" + ("$^\\dagger$" if L > TRAINED_LEN else "")
                            for L in LENGTHS])
        ax.set_xlabel("Sequence length (tokens)")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=5)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(x) - 0.45)
        ax.text(0.985, 0.97, "$^\\dagger$beyond training length", transform=ax.transAxes,
                fontsize=6.5, color=INK_2, va="top", ha="right")
    axes[0].set_ylim(0, 38)    # headroom: the 4,096 whisker stays clear of the note
    axes[1].set_ylim(-0.5, 5.6)
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.2,
                   labelspacing=0.3)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return fig


def best(vals, i, lower):
    return sorted(vals, key=lambda k: vals[k][0][i], reverse=not lower)[0]


def latex_table(data):
    lines = [
        "% Generated by results/make_figures.py -- do not edit by hand.",
        "% Requires \\usepackage{booktabs}. Bold: best in the column (the 95% CIs of",
        "% the top models overlap). \\dag: longer than the 1,024 training length.",
        "\\begin{tabular}{l" + "r" * len(LENGTHS) + "}",
        "\\toprule",
        "Model & " + " & ".join(f"{L:,}" + ("$^\\dag$" if L > TRAINED_LEN else "")
                                for L in LENGTHS) + " \\\\",
    ]
    for metric, title, lower in (("pppl", "Perplexity (PPL) $\\downarrow$", True),
                                 ("mlm_acc", "MLM accuracy (\\%) $\\uparrow$", False)):
        lines += ["\\midrule",
                  f"\\multicolumn{{{len(LENGTHS) + 1}}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        vals = {key: series(data, key, metric) for key, *_ in ARMS}
        for key, label, _c in ARMS:
            cells = [f"{vals[key][0][i]:.2f}" for i in range(len(LENGTHS))]
            cells = [f"\\textbf{{{c}}}" if best(vals, i, lower) == key else c
                     for i, c in enumerate(cells)]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def markdown_tables(data):
    """The README's tables: per-length PPL and accuracy (mean +/- CI; bold = best in
    the column) and the averages of Figure 1."""
    out = []
    head = " | ".join(f"{L:,}" + ("†" if L > TRAINED_LEN else "") for L in LENGTHS)
    for metric, title, lower in (("pppl", "Perplexity (PPL) ↓", True),
                                 ("mlm_acc", "MLM accuracy (%) ↑", False)):
        vals = {k: series(data, k, metric) for k, *_ in ARMS}
        out += [f"*{title}*", "", f"| Model | {head} |", "|:--|" + "--:|" * len(LENGTHS)]
        for key, label, _c in ARMS:
            cells = [f"{vals[key][0][i]:.2f} ± {vals[key][1][i]:.2f}"
                     for i in range(len(LENGTHS))]
            cells = [f"**{c}**" if best(vals, i, lower) == key else c
                     for i, c in enumerate(cells)]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["| Model | PPL reduction vs GQA (geo. mean) | MLM accuracy gain vs GQA (mean) |",
            "|:--|--:|--:|"]
    for key, label, _c in ARMS:
        if key != BASELINE:
            g, a = averages(data, key)
            out.append(f"| {label} | {g:.2f} % | +{a:.2f} |")
    return "\n".join(out)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(Path(args.results_dir))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style(plt)
    for name, build in (("fig1_bars_summary", figure_bar_summary),
                        ("fig2_bars_by_length", figure_bars)):
        fig = build(plt, data)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"wrote {out / name}.pdf / .png")
    (out / "tab_attention_ablation.tex").write_text(latex_table(data), encoding="utf-8")
    print(f"wrote {out / 'tab_attention_ablation.tex'}")
    if args.tables:
        print()
        print(markdown_tables(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
