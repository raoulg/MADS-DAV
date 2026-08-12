#!/usr/bin/env python3
"""Build notebooks/lesson5/05.1-relationships.ipynb (PTT-45).

Written as a builder so the notebook is reproducible and reviewable as source
rather than as a diff of JSON. Run from the repo root:

    uv run python build_nb051.py
"""

import json
from pathlib import Path

OUT = Path("/Users/rgrouls/code/courses/MADS-DAV/notebooks/lesson5/05.1-relationships.ipynb")

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


# ---------------------------------------------------------------- intro
md("""
# 5. Relationships, and what makes one believable

**The question: does this pattern mean anything?**

Lesson 2 asked what one row is. Lesson 3 subtracted the boring part. This lesson is about
the step everybody skips — deciding whether the thing you found is real before writing it
down.

Four moves, in order:

1. **Look before you summarise.** Summary statistics agree far more often than pictures do.
2. **Walk into the trap.** Hunt for a difference between two groups that are not different.
   You will find one. So will everybody else.
3. **Rebuild.** A grid for deciding what a finding is worth, and seven claims run through it.
4. **A finding that survives all of it** — how eight people type, which turns out to be more
   identifying than what they talk about.
""")

code("""
import sys

sys.path.append("../../")

import numpy as np
import pandas as pd
import seaborn as sns
from goad_toolkit.visualizer import (
    CorrelationHeatmap,
    PlotSettings,
    RegPlot,
    ScatterPlot,
)
from scipy import stats

from scripts.pipelines import build_irc_pipeline
from wa_analyzer.data import load_own_chat, load_showcase
""")

# ---------------------------------------------------------------- 5.1 anscombe
md("""
## 5.1 Four datasets, one summary

Anscombe's quartet. Four small datasets, built in 1973 to make exactly one point.

Compute the things you would normally report.
""")

code("""
anscombe = load_showcase("anscombe")

summary = anscombe.groupby("dataset").agg(
    n=("x", "size"),
    mean_x=("x", "mean"),
    mean_y=("y", "mean"),
    std_x=("x", "std"),
    std_y=("y", "std"),
)
summary["correlation"] = anscombe.groupby("dataset").apply(
    lambda g: g.x.corr(g.y), include_groups=False
)
summary.round(2)
""")

md("""
Identical to two decimal places, on every statistic. Same means, same spreads, same
correlation. If you reported these four datasets in a table, they would be the same dataset.

Now look at them.
""")

code("""
settings = PlotSettings(
    figsize=(12, 3.2),
    title="Anscombe's quartet: same statistics, four different stories",
    subplot_titles=[f"dataset {name}" for name in sorted(anscombe.dataset.unique())],
    xlabel="x",
    ylabel="y",
)

host = ScatterPlot(settings)
fig, axes = host.create_figure(n_plots=4)

for ax, (name, group) in zip(axes, anscombe.groupby("dataset")):
    host.plot_on_axes(RegPlot(settings), ax, data=group, x="x", y="y",
                      ci=None, color="#c44e52", scatter_kws={"color": "#4c72b0"})
fig.tight_layout()
""")

md("""
Four completely different situations:

- **I** — a genuine linear relationship with noise. The only one where the line means what
  you think it means.
- **II** — a curve. The line is the best straight answer to a question whose answer is not
  straight.
- **III** — a perfect line plus one outlier, which drags the fit off the line everything
  else sits on.
- **IV** — no relationship at all. One point at x=19 creates the entire slope. Delete it and
  there is nothing left.

`linregress` returns a slope and an r-value for all four, and never mentions which situation
you are in.

**The habit this buys you:** plot it before you summarise it, and plot it again after. A
scatter costs one line and rules out four different ways of being wrong.
""")

code("""
datasaurus = load_showcase("datasaurus")

dino_summary = datasaurus.groupby("dataset").agg(
    mean_x=("x", "mean"), mean_y=("y", "mean"),
    std_x=("x", "std"), std_y=("y", "std"),
)
dino_summary["correlation"] = datasaurus.groupby("dataset").apply(
    lambda g: g.x.corr(g.y), include_groups=False
)
print(f"{len(dino_summary)} datasets, and their summaries agree to one decimal:")
dino_summary.round(1).head()
""")

md("""
Thirteen this time, same trick, and one of them is a dinosaur.
""")

code("""
names = sorted(datasaurus.dataset.unique())
zoo = PlotSettings(
    figsize=(13, 8),
    title="Thirteen datasets with the same mean, spread and correlation",
    subplot_titles=names,
    xlabel="",
    ylabel="",
)

host = ScatterPlot(zoo)
fig, axes = host.create_figure(n_plots=len(names))

for ax, name in zip(axes, names):
    host.plot_on_axes(ScatterPlot(zoo), ax,
                      data=datasaurus[datasaurus.dataset == name], x="x", y="y",
                      s=8, color="#4c72b0")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
""")

# ---------------------------------------------------------------- 5.2 the toolkit
md("""
## 5.2 The line you draw is a claim

A scatter shows the relationship. A fitted line says what *kind* it is — and that is a
claim you are making, not a formatting choice.

`RegPlot` takes the three that matter: `fit_reg` for whether to draw one at all, `order`
for a polynomial, `lowess` to let the data pick the shape.

Fuel efficiency against weight, which is famously not a straight line.
""")

code("""
mpg = load_showcase("mpg").dropna(subset=["weight", "mpg"])

shapes = PlotSettings(
    figsize=(13, 3.6),
    title="Three claims about the same scatter",
    subplot_titles=["order=1: a straight line", "order=2: a curve", "lowess: no shape assumed"],
    xlabel="weight (lbs)",
    ylabel="miles per gallon",
)

host = ScatterPlot(shapes)
fig, axes = host.create_figure(n_plots=3)

host.plot_on_axes(RegPlot(shapes), axes[0], data=mpg, x="weight", y="mpg", ci=None)
host.plot_on_axes(RegPlot(shapes), axes[1], data=mpg, x="weight", y="mpg", order=2, ci=None)
host.plot_on_axes(RegPlot(shapes), axes[2], data=mpg, x="weight", y="mpg", lowess=True)
fig.tight_layout()
""")

md("""
The straight line is wrong in a specific, readable way: it over-predicts economy for the
heaviest cars and under-predicts it for the lightest, because it is averaging a curve. The
quadratic follows the bend. Lowess agrees with the quadratic without being told there was a
bend to find.

**Which to use.** `lowess` first, when you do not know the shape — it is a description.
Then a polynomial once you have decided what the shape *is* — that is a model, it has
parameters, and you can extrapolate from it and be wrong in an informative way.

Numbers behind the lines. `scipy.stats.linregress` for the straight one:
""")

code("""
fit = stats.linregress(mpg.weight, mpg.mpg)
print(f"slope     {fit.slope:.5f} mpg per lb")
print(f"intercept {fit.intercept:.2f}")
print(f"r         {fit.rvalue:.3f}   (r^2 = {fit.rvalue**2:.3f})")
print(f"p         {fit.pvalue:.2e}")
""")

md("""
`r² = 0.69`, and a p-value with a lot of zeros in it. Both are true and neither tells you
the relationship is curved — which the picture said immediately.

A log transform is often the honest fix, because "each extra pound costs proportionally
less" is a claim about *ratios* rather than differences:
""")

code("""
log_fit = stats.linregress(np.log(mpg.weight), np.log(mpg.mpg))
quad = np.polyfit(mpg.weight, mpg.mpg, 2)

print(f"log-log slope {log_fit.slope:.3f}, r^2 {log_fit.rvalue**2:.3f}")
print(f"quadratic     {quad[0]:.3e} x^2 + {quad[1]:.3f} x + {quad[2]:.1f}")
""")

md("""
The log-log fit says mpg falls with roughly the **1.2th power** of weight, and its r² beats
the straight-line fit on the raw scale. That is a sentence about mechanism — doubling the
weight costs you more than half the economy — rather than a slope in units nobody thinks in.

> **Your turn, briefly.** Fit `order=3` to the same data. Does it follow the points better?
> Does it predict better? Those are different questions, and lesson 6 is where the second one
> gets its own machinery.
""")


def write() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": kind,
                "id": f"c{i:02d}",
                "metadata": {},
                **({"source": text.splitlines(keepends=True), "outputs": [], "execution_count": None}
                   if kind == "code" else {"source": text.splitlines(keepends=True)}),
            }
            for i, (kind, text) in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} with {len(cells)} cells")


if __name__ == "__main__":
    write()
