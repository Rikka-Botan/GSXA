#!/usr/bin/env python
"""Figures and tables of the attention ablation (GSXA and the four compared models).

Reads the five scoring JSONs in ``results/attention_ablation/`` (pseudo-perplexity and
MLM accuracy per sequence length, each the mean and 95% Student-t CI over 20 disjoint
measurements) and writes, into ``docs/figures/``:

* ``fig1_bars_summary.{png,pdf}`` -- gains over GQA averaged over 64 ... 4,096 tokens,
* ``fig2_bars_by_length.{png,pdf}`` -- the same gains at each length,
* ``fig3_quality_vs_length.{png,pdf}`` -- PPPL and MLM accuracy, 64 ... 4,096 tokens,
* ``figA1_full_sweep.{png,pdf}`` -- the same at every measured length, 8 ... 65,536,
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

# Categorical slots 1-4 of the reference palette (validated on white, light mode:
# worst adjacent CVD dE 9.1, normal-vision dE 22.9); plain attention is the
# reference, drawn in neutral ink. Marker shape is the print / CVD backup channel.
ARMS = [
#   GSXA (Gated Scalable Exclusive Self Attention, the proposed method)
#       = scalable factor (SSSMax) + XSA projection + per-head gate (attn_gate "exclusive")
#   Gated Attention = scalable factor + per-head gate (attn_gate "head")
#   XSA             = scalable factor + parameter-free XSA (attn_gate "xsa")
#   Scalable GQA    = scalable factor alone
#   GQA             = plain RoPE + GQA attention, the reference
# Gated Attention and XSA carry the scalable factor too: say so in the caption.
    ("at_sf_xsa_gate", "GSXA", "#2a78d6", "o"),
    ("at_sf_gate", "Gated Attention", "#eb6834", "s"),
    ("at_sf_xsa", "XSA", "#1baf7a", "^"),
    ("at_sf", "Scalable GQA", "#eda100", "D"),
    ("at_plain", "GQA", "#52514e", "v"),
]
BASELINE = "at_plain"
BASE_NAME = "GQA"   # how the reference is named in axis labels and notes
INK, INK_2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#f1f0ec"
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
    lengths = sorted(set.intersection(*(set(rows) for rows in data.values())))
    return data, lengths


def series(data, key, metric, lengths):
    scale = 100.0 if metric == "mlm_acc" else 1.0
    rows = data[key]
    return ([rows[L][metric] * scale for L in lengths],
            [rows[L][metric + "_ci95"] * scale for L in lengths])


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


def frame(ax, lengths):
    ax.set_xscale("log", base=2)
    ax.set_xlim(lengths[0] / 1.4, lengths[-1] * 1.4)
    if len(lengths) <= 8:      # a short range: label every length
        ax.set_xticks(lengths)
        ax.set_xticklabels([f"{L:,}" for L in lengths])
    else:
        ax.set_xticks([2 ** k for k in range(3, 17, 2)])
        ax.set_xticklabels([f"$2^{{{k}}}$" for k in range(3, 17, 2)])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    # lengths beyond training: shaded
    ax.axvspan(TRAINED_LEN, lengths[-1] * 1.4, color=BAND, zorder=0, lw=0)
    ax.set_xlabel("Sequence length (tokens)")


def band_label(ax, x, y, va="bottom"):
    ax.text(x, y, "beyond\ntraining length", transform=ax.transAxes, fontsize=6.5,
            color=INK_2, va=va, ha="right", linespacing=1.1)


def panel_label(ax, text):
    ax.text(-0.14, 1.03, text, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", color=INK)


def draw(ax, x, y, ci, color, marker, label, z=3):
    lo = [a - b for a, b in zip(y, ci)]
    hi = [a + b for a, b in zip(y, ci)]
    ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0, zorder=z - 1)
    ax.plot(x, y, color=color, marker=marker, markersize=3.6, linewidth=1.3,
            markeredgecolor="white", markeredgewidth=0.5, label=label, zorder=z)


def figure_quality(plt, data, lengths):
    """PPPL and MLM accuracy vs length over ``BAR_LENGTHS`` (64 ... 4,096): the main
    figure. ``figure_full_sweep`` draws every measured length."""
    lengths = [L for L in BAR_LENGTHS if L in lengths]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), constrained_layout=True)
    for ax, metric in ((axes[0], "pppl"), (axes[1], "mlm_acc")):
        frame(ax, lengths)
        for z, (key, label, color, marker) in enumerate(reversed(ARMS)):
            y, ci = series(data, key, metric, lengths)
            draw(ax, lengths, y, ci, color, marker, label, z=3 + z)
        if metric == "pppl":
            ax.set_ylim(9, 22.5)
            ax.set_yticks(range(10, 23, 2))
            ax.set_ylabel("Pseudo-perplexity")
            band_label(ax, 0.985, 0.97, va="top")
        else:
            ax.set_ylim(46, 58.5)
            ax.set_ylabel("MLM accuracy (%)")
            band_label(ax, 0.985, 0.03)
    handles, labels = axes[0].get_legend_handles_labels()
    order = list(reversed(range(len(handles))))   # ARMS order, GSXA first
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="upper center", frameon=False, handlelength=2.0,
                   borderaxespad=0.2, bbox_to_anchor=(0.5, 1.0), labelspacing=0.3)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return fig


def figure_full_sweep(plt, data, lengths):
    """Appendix: PPPL and MLM accuracy at every measured length, 8 ... 65,536."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), constrained_layout=True)
    for ax, metric in ((axes[0], "pppl"), (axes[1], "mlm_acc")):
        frame(ax, lengths)
        for z, (key, label, color, marker) in enumerate(reversed(ARMS)):
            y, ci = series(data, key, metric, lengths)
            draw(ax, lengths, y, ci, color, marker, label, z=3 + z)
        if metric == "pppl":
            ax.set_yscale("log")
            ax.set_ylim(7, 5000)
            ax.set_yticks([10, 30, 100, 300, 1000, 3000])
            ax.set_yticklabels(["10", "30", "100", "300", "1,000", "3,000"])
            ax.minorticks_off()
            ax.set_ylabel("Pseudo-perplexity (log scale)")
            band_label(ax, 0.985, 0.03)
        else:
            ax.set_ylim(0, 62)
            ax.set_ylabel("MLM accuracy (%)")
            band_label(ax, 0.985, 0.97, va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    order = list(reversed(range(len(handles))))   # ARMS order, full first
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="upper center", frameon=False, handlelength=2.0,
                   borderaxespad=0.2, bbox_to_anchor=(0.42, 1.0), labelspacing=0.3)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return fig


def relative(data, key, metric, lengths):
    a, ca = series(data, key, metric, lengths)
    p, cp = series(data, BASELINE, metric, lengths)
    if metric == "pppl":   # ratio, CI by the delta method
        r = [x / y for x, y in zip(a, p)]
        e = [v * math.hypot(cx / x, cy / y) for v, x, y, cx, cy in zip(r, a, p, ca, cp)]
    else:                  # accuracy points
        r = [x - y for x, y in zip(a, p)]
        e = [math.hypot(cx, cy) for cx, cy in zip(ca, cp)]
    return r, e


def figure_relative(plt, data, lengths):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), constrained_layout=True)
    arms = [a for a in ARMS if a[0] != BASELINE]
    for ax, metric in ((axes[0], "pppl"), (axes[1], "mlm_acc")):
        frame(ax, lengths)
        ref = 1.0 if metric == "pppl" else 0.0
        ax.axhline(ref, color=INK_2, linewidth=0.9, zorder=2)
        for z, (key, label, color, marker) in enumerate(reversed(arms)):
            r, e = relative(data, key, metric, lengths)
            draw(ax, lengths, r, e, color, marker, label, z=3 + z)
        if metric == "pppl":
            ax.set_yscale("log")
            ax.set_ylim(0.025, 2.2)
            ticks = [0.03, 0.1, 0.3, 0.5, 1, 2]
            ax.set_yticks(ticks)
            ax.set_yticklabels(["0.03", "0.1", "0.3", "0.5", "1", "2"])
            ax.minorticks_off()
            ax.set_ylabel(f"PPPL ratio to {BASE_NAME}")
            ax.text(0.02, 0.03, f"below 1: better than {BASE_NAME}", transform=ax.transAxes,
                    fontsize=6.5, color=INK_2, va="bottom", ha="left")
            # above the reference where every curve is below it (32 ... 512)
            ax.text(2 ** 5.3, 1.06, BASE_NAME, fontsize=6.5, color=INK_2,
                    va="bottom", ha="left")
        else:
            ax.set_ylim(-10, 42)
            ax.set_ylabel(f"Δ MLM accuracy vs {BASE_NAME} (points)")
            ax.text(0.02, 0.97, f"above 0: better than {BASE_NAME}", transform=ax.transAxes,
                    fontsize=6.5, color=INK_2, va="top", ha="left")
        band_label(ax, 0.985, 0.97, va="top")
    handles, labels = axes[1].get_legend_handles_labels()
    order = list(reversed(range(len(handles))))
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="upper left", frameon=False, handlelength=2.0, borderaxespad=0.2,
                   bbox_to_anchor=(0.0, 0.88), labelspacing=0.3)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return fig


def figure_in_distribution(plt, data, lengths):
    """Single column: PPPL change vs plain (%) at the trained lengths (<= 1,024),
    where the differences are a few percent and vanish on the full-range axes."""
    short = [L for L in lengths if L <= TRAINED_LEN]
    lim = (-40, 20)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH / 2, 2.5), constrained_layout=True)
    ax.set_xscale("log", base=2)
    ax.set_xlim(short[0] / 1.3, short[-1] * 1.3)
    ax.set_xticks(short)
    ax.set_xticklabels([f"{L:,}" for L in short])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Sequence length (tokens)")
    ax.axhline(0, color=INK_2, linewidth=0.9, zorder=2)
    arms = [a for a in ARMS if a[0] != BASELINE]
    for z, (key, label, color, marker) in enumerate(reversed(arms)):
        r, e = relative(data, key, "pppl", short)
        d = [100.0 * (v - 1.0) for v in r]
        e = [100.0 * x for x in e]
        draw(ax, short, d, e, color, marker, label, z=3 + z)
        for L, v in zip(short, d):        # points clipped by the axis range: their value
            if v > lim[1]:
                ax.annotate(f"↑ {v:+.0f}%", xy=(L, lim[1]), xytext=(0, -2),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=6.5, color=INK_2)
    ax.set_ylim(*lim)
    ax.set_ylabel(f"Δ pseudo-perplexity vs {BASE_NAME} (%)")
    ax.text(0.98, 0.04, f"below 0: better than {BASE_NAME}", transform=ax.transAxes,
            fontsize=6.5, color=INK_2, va="bottom", ha="right")
    handles, labels = ax.get_legend_handles_labels()
    order = list(reversed(range(len(handles))))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right",
              frameon=False, handlelength=2.0, borderaxespad=0.2, labelspacing=0.25,
              bbox_to_anchor=(1.0, 0.1))
    return fig


# The lengths over which GSXA (SF + XSA + Gate) has the lowest PPPL at
# every single length: 64 ... 4,096 (below, Gated Attention / XSA lead; from 8,192
# up, Scalable GQA does). The bar figures cover exactly this range and say so.
BAR_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]


def gains(data, key, lengths):
    """PPPL reduction (%) and MLM-accuracy gain (points) vs plain, with the
    conservative (independent) 95% CI half-widths."""
    r, er = relative(data, key, "pppl", lengths)
    a, ea = relative(data, key, "mlm_acc", lengths)
    return ([100.0 * (1.0 - v) for v in r], [100.0 * x for x in er], a, ea)


def figure_bars(plt, data):
    """Grouped bars: PPPL reduction and MLM-accuracy gain over plain attention at
    each length of BAR_LENGTHS, one bar per scalable-factor variant."""
    arms = [a for a in ARMS if a[0] != BASELINE]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)
    n, width = len(arms), 0.2
    x = list(range(len(BAR_LENGTHS)))
    for ax, which, ylabel in ((axes[0], 0, f"PPPL reduction vs {BASE_NAME} (%)"),
                              (axes[1], 2, f"MLM accuracy gain vs {BASE_NAME} (points)")):
        for i, (key, label, color, _marker) in enumerate(arms):
            g = gains(data, key, BAR_LENGTHS)
            vals, ci = g[which], g[which + 1]
            xs = [p + (i - (n - 1) / 2) * width for p in x]
            # white edge = the surface gap between touching bars
            ax.bar(xs, vals, width=width, color=color, edgecolor="white", linewidth=0.6,
                   label=label, zorder=3)
            ax.errorbar(xs, vals, yerr=ci, fmt="none", ecolor=INK_2, elinewidth=0.6,
                        capsize=1.2, capthick=0.6, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{L:,}" + ("$^\\dagger$" if L > TRAINED_LEN else "")
                            for L in BAR_LENGTHS])
        ax.set_xlabel("Sequence length (tokens)")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=5)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(x) - 0.45)
    axes[0].set_ylim(0, 38)    # headroom: the 4,096 whisker stays clear of the note
    axes[1].set_ylim(-0.5, 5.6)
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.2,
                   labelspacing=0.3)
    for ax in axes:
        ax.text(0.985, 0.97, "$^\\dagger$beyond training length", transform=ax.transAxes,
                fontsize=6.5, color=INK_2, va="top", ha="right")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    return fig


def figure_bar_summary(plt, data):
    """Single column: the gains averaged over BAR_LENGTHS -- geometric-mean PPPL
    reduction and mean accuracy gain -- one horizontal bar per variant, full on top."""
    arms = [a for a in ARMS if a[0] != BASELINE]
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH / 2, 2.55), constrained_layout=True)
    rows = list(range(len(arms)))[::-1]          # full at the top
    for ax, which, xlabel, fmt, xmax in (
            (axes[0], "pppl", f"PPPL reduction vs {BASE_NAME} (%)", "{:.1f}%", 14.5),
            (axes[1], "acc", f"MLM accuracy gain vs {BASE_NAME} (pt)", "+{:.2f}", 1.6)):
        for row, (key, _label, color, _m) in zip(rows, arms):
            if which == "pppl":
                r, _ = relative(data, key, "pppl", BAR_LENGTHS)
                v = 100.0 * (1.0 - math.exp(sum(math.log(t) for t in r) / len(r)))
            else:
                a, _ = relative(data, key, "mlm_acc", BAR_LENGTHS)
                v = sum(a) / len(a)
            ax.barh(row, v, height=0.62, color=color, edgecolor="white", linewidth=0.6,
                    zorder=3)
            ax.text(v + xmax * 0.015, row, fmt.format(v), va="center", ha="left",
                    fontsize=7, color=INK, fontweight="bold" if key == ARMS[0][0] else None)
        ax.set_yticks(rows)
        ax.set_yticklabels([label for _k, label, *_ in arms])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_title(f"Averaged over lengths {BAR_LENGTHS[0]:,}–{BAR_LENGTHS[-1]:,}",
                      fontsize=7.5, color=INK_2, loc="left")
    return fig


def latex_table(data, lengths):
    show = [L for L in BAR_LENGTHS if L in lengths]
    lines = [
        "% Generated by results/make_figures.py -- do not edit by hand.",
        "% Requires \\usepackage{booktabs}. Bold: best in the column (the 95% CIs of",
        "% the top models overlap). \\dag: longer than the 1,024 training length.",
        "\\begin{tabular}{l" + "r" * len(show) + "}",
        "\\toprule",
        "Model & " + " & ".join(f"{L:,}" + ("$^\\dag$" if L > TRAINED_LEN else "")
                                for L in show) + " \\\\",
    ]
    for metric, title, digits, lower_better in (
            ("pppl", "Pseudo-perplexity $\\downarrow$", 2, True),
            ("mlm_acc", "MLM accuracy (\\%) $\\uparrow$", 2, False)):
        lines += ["\\midrule", f"\\multicolumn{{{len(show) + 1}}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        vals = {key: series(data, key, metric, show) for key, *_ in ARMS}
        bold = {}
        for i, L in enumerate(show):
            bold[i] = sorted(vals, key=lambda k: vals[k][0][i], reverse=not lower_better)[0]
        for key, label, *_ in ARMS:
            cells = []
            for i in range(len(show)):
                v = vals[key][0][i]
                text = f"{v:,.{digits}f}"
                cells.append(f"\\textbf{{{text}}}" if bold[i] == key else text)
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def markdown_tables(data, lengths):
    """Table 3 of the README (64 ... 4,096, mean +/- CI; bold = best in the column,
    whether or not its CI overlaps the runner-up's) and the all-lengths appendix."""
    show = [L for L in BAR_LENGTHS if L in lengths]
    out = []
    for metric, title, digits, lower in (("pppl", "Pseudo-perplexity ↓", 2, True),
                                         ("mlm_acc", "MLM accuracy (%) ↑", 2, False)):
        vals = {k: series(data, k, metric, show) for k, *_ in ARMS}
        bold = {}
        for i in range(len(show)):
            ranked = sorted(vals, key=lambda k: vals[k][0][i], reverse=not lower)
            (b, cb), (s, cs) = ((vals[r][0][i], vals[r][1][i]) for r in ranked[:2])
            bold[i] = ranked[0]
        head = " | ".join(f"{L:,}" + ("†" if L > TRAINED_LEN else "") for L in show)
        out += [f"*{title}*", "", f"| Model | {head} |", "|:--|" + "--:|" * len(show)]
        for key, label, *_ in ARMS:
            cells = [f"{vals[key][0][i]:.{digits}f} ± {vals[key][1][i]:.{digits}f}"
                     for i in range(len(show))]
            cells = [f"**{c}**" if bold[i] == key else c for i, c in enumerate(cells)]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["Appendix: the full sweep, 8 – 65,536 tokens", ""]
    for metric, title, digits in (("pppl", "Pseudo-perplexity ↓", 3),
                                  ("mlm_acc", "MLM accuracy (%) ↑", 2)):
        out += [f"*{title}*", "", "| Length | " + " | ".join(l for _k, l, *_ in ARMS) + " |",
                "|--:|" + "--:|" * len(ARMS)]
        cols = {k: series(data, k, metric, lengths) for k, *_ in ARMS}
        for i, L in enumerate(lengths):
            out.append(f"| {L:,} | " + " | ".join(
                f"{cols[k][0][i]:.{digits}f} ± {cols[k][1][i]:.{digits}f}" for k, *_ in ARMS) + " |")
        out.append("")
    out += [""]
    # the averages behind Figure 1
    out += ["| Model | PPPL reduction vs GQA, 64–4,096 (geo. mean) | MLM accuracy gain vs GQA, 64–4,096 (mean) |",
            "|:--|--:|--:|"]
    for key, label, *_ in ARMS:
        if key == BASELINE:
            continue
        r, _ = relative(data, key, "pppl", BAR_LENGTHS)
        a, _ = relative(data, key, "mlm_acc", BAR_LENGTHS)
        g = 100.0 * (1.0 - math.exp(sum(math.log(x) for x in r) / len(r)))
        out.append(f"| {label} | {g:.2f} % | +{sum(a) / len(a):.2f} |")
    return "\n".join(out)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, lengths = load(Path(args.results_dir))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style(plt)
    for name, build in (("fig1_bars_summary", lambda p, d, _l: figure_bar_summary(p, d)),
                        ("fig2_bars_by_length", lambda p, d, _l: figure_bars(p, d)),
                        ("fig3_quality_vs_length", figure_quality),
                        ("figA1_full_sweep", figure_full_sweep)):
        fig = build(plt, data, lengths)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"wrote {out / name}.pdf / .png")
    (out / "tab_attention_ablation.tex").write_text(latex_table(data, lengths), encoding="utf-8")
    print(f"wrote {out / 'tab_attention_ablation.tex'}")
    if args.tables:
        print()
        print(markdown_tables(data, lengths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
