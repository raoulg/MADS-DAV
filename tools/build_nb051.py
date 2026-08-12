#!/usr/bin/env python3
"""Build notebooks/lesson5/05.1-relationships.ipynb (PTT-45).

Written as a builder so the notebook is reproducible and reviewable as source
rather than as a diff of JSON. Run from the repo root:

    uv run python tools/build_nb051.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/lesson5/05.1-relationships.ipynb"

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
    figsize=(13, 3.2),
    title="Anscombe's quartet: same statistics, four different stories",
    subplot_titles=[f"dataset {name}" for name in sorted(anscombe.dataset.unique())],
    xlabel="x",
    ylabel="y",
    max_cols=4,
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

# ---------------------------------------------------------------- 5.3 forking paths
md("""
## 5.3 Find me a difference

Here is the IRC corpus, one row per author, with a `cohort` label attached to each one.

**Your job: find the most interesting difference between cohort A and cohort B.**

Fifteen metrics are computed below. Pick whichever difference looks most striking, and
write the sentence you would put in a report. Do it before reading on — the section does not
work if you skip this part.
""")

code("""
msgs = build_irc_pipeline().apply(load_showcase("ubuntu_irc"))
uk = msgs[msgs.channel == "#ubuntu-uk"].copy()

uk["length"] = uk.message.str.len()
uk["words"] = uk.message.str.split().str.len()
uk["is_night"] = uk.hh.between(0, 5)
uk["starts_upper"] = uk.message.str.match(r"^[A-Z]").astype(float)
uk["exclaims"] = uk.message.str.count("!")
uk["is_weekend"] = uk.date.dt.dayofweek >= 5

authors = uk.groupby("author").agg(
    n=("message", "size"),
    mean_length=("length", "mean"),
    median_length=("length", "median"),
    mean_words=("words", "mean"),
    night_share=("is_night", "mean"),
    upper_share=("starts_upper", "mean"),
    question_rate=("n_question", "mean"),
    exclaim_rate=("exclaims", "mean"),
    url_share=("has_url", "mean"),
    address_share=("addressed_to", lambda s: s.notna().mean()),
    active_days=("date", "nunique"),
    weekend_share=("is_weekend", "mean"),
)
authors = authors[authors.n >= 30]
authors["msgs_per_day"] = authors.n / authors.active_days
authors["unique_share"] = uk.groupby("author").message.nunique().reindex(authors.index) / authors.n
authors["hour_spread"] = uk.groupby("author").hh.std().reindex(authors.index)

METRICS = [
    "mean_length", "median_length", "mean_words", "night_share", "upper_share",
    "question_rate", "exclaim_rate", "url_share", "address_share", "active_days",
    "msgs_per_day", "unique_share", "hour_spread", "weekend_share", "n",
]

rng = np.random.default_rng(60)
authors["cohort"] = rng.permutation(["A", "B"] * (len(authors) // 2 + 1))[: len(authors)]

print(f"{len(authors)} authors, {len(METRICS)} metrics to choose from")
authors.groupby("cohort")[METRICS].mean().round(3).T
""")

md("""
Something in that table looks like a finding. Here is the one that stands out most.
""")

code("""
comparison = []
for metric in METRICS:
    a = authors.loc[authors.cohort == "A", metric].dropna()
    b = authors.loc[authors.cohort == "B", metric].dropna()
    comparison.append({
        "metric": metric,
        "cohort A": a.mean(),
        "cohort B": b.mean(),
        "difference": a.mean() / b.mean() - 1,
        "p": stats.ttest_ind(a, b, equal_var=False).pvalue,
    })

comparison = pd.DataFrame(comparison).sort_values("p").reset_index(drop=True)
comparison.head(5).round(4)
""")

md("""
Cohort A does **twice as much of its posting between midnight and 05:00** as cohort B —
13.5% of their messages against 6.7% — at `p = 0.0026`, on 221 authors per group.

That is a publishable-sounding sentence, and a mechanism suggests itself immediately:
cohort A must be night owls, or in another timezone. Write the sentence down. Notice how
quickly the explanation arrived.

### The reveal

The cohort labels were assigned by `np.random.permutation`. There is no cohort. The two
groups are the same population, split at random, and every difference in that table is
produced by nothing at all.
""")

code("""
print(authors.cohort.value_counts().to_string())
print("\\nthe line that made them, from the cell above:")
print('rng.permutation(["A", "B"] * (len(authors) // 2 + 1))[: len(authors)]')
""")

# ---------------------------------------------------------------- the maths
md("""
### Why that was going to happen

This is worth doing slowly, because the arithmetic is the whole defence.

**One test.** A p-value is the probability of seeing a difference at least this big *if
nothing is going on*. Use the conventional threshold `p < 0.05` and you have chosen to
accept a 5% false-alarm rate. That is not a flaw — it is the price, agreed in advance:

    P(this metric falsely looks significant) = 0.05

**So the chance it behaves is:**

    P(this metric stays quiet) = 1 - 0.05 = 0.95

**Fifteen tests.** If the metrics were independent, all fifteen staying quiet is `0.95`
multiplied by itself fifteen times — that is what independence means: each one's outcome
tells you nothing about the next:

    P(all fifteen stay quiet) = 0.95^15 = 0.46

**And at least one going off is the complement of that:**

    P(at least one false finding) = 1 - 0.95^15 = 0.54

The complement is the trick worth remembering. "At least one" is awkward to count directly —
it means exactly one, or exactly two, or exactly three... — but "none of them" is a single
product, and everything else is what is left.
""")

code("""
p_threshold = 0.05
n_metrics = len(METRICS)

quiet_one = 1 - p_threshold
quiet_all = quiet_one ** n_metrics

print(f"P(one metric stays quiet)     = 1 - {p_threshold}            = {quiet_one}")
print(f"P(all {n_metrics} stay quiet)         = {quiet_one}^{n_metrics}          = {quiet_all:.3f}")
print(f"P(at least one false finding) = 1 - {quiet_all:.3f}        = {1 - quiet_all:.3f}")
""")

md("""
**That number is a sanity baseline**, and computing one is the habit this section is really
teaching. Before asking "is my finding real?", ask "how often would I have found *something*
if there were nothing to find?" If the answer is "half the time", then finding something is
not evidence of anything.

It is the same move as lesson 4's shuffle test, done with arithmetic instead of simulation:
work out what nothing looks like, then compare.

### Except 0.54 is wrong here, and the reason matters more than the number

That formula assumed the fifteen metrics were independent. Look at them.
""")

code("""
correlations = authors[METRICS].corr(method="spearman")

heat = PlotSettings(
    figsize=(9, 7),
    title="The fifteen 'independent' metrics",
    xlabel="",
    ylabel="",
)
fig, ax = CorrelationHeatmap(heat).plot(data=authors[METRICS], method="spearman",
                                        annot=False)
""")

code("""
pairs = (
    correlations.where(~np.eye(len(METRICS), dtype=bool))
    .abs().stack().sort_values(ascending=False)
)
print("the least independent pairs:")
print(pairs[::2].head(5).round(2).to_string())
""")

md("""
`mean_length`, `median_length` and `mean_words` are three names for message size.
`active_days`, `n` and `msgs_per_day` are three views of how much someone posts. Asking
fifteen questions is not the same as having fifteen chances, because several of the
questions are the same question.

**So how many chances were there really?** Two ways to estimate it, and they bracket the
answer.
""")

code("""
eigenvalues = np.linalg.eigvalsh(authors[METRICS].corr().fillna(0))
effective_dims = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

print(f"metrics counted naively      : {len(METRICS)}")
print(f"effective dimensions         : {effective_dims:.1f}")
print(f"1 - 0.95^{effective_dims:.1f}                : {1 - 0.95 ** effective_dims:.3f}")
""")

md("""
The second method is to stop deriving and go and measure: shuffle the labels many times and
count how often a hunt over all fifteen turns up *anything*.
""")

code("""
def hunt_once(seed: int) -> bool:
    \"\"\"Assign cohorts at random, then look for any metric with p < 0.05.\"\"\"
    rng = np.random.default_rng(seed)
    labels = np.array(["A", "B"] * (len(authors) // 2 + 1))[: len(authors)]
    cohort = pd.Series(rng.permutation(labels), index=authors.index)
    for metric in METRICS:
        a = authors.loc[cohort == "A", metric].dropna()
        b = authors.loc[cohort == "B", metric].dropna()
        if stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05:
            return True
    return False


runs = 300
hits = sum(hunt_once(seed) for seed in range(runs))
print(f"a 'finding' appeared in {hits}/{runs} random assignments = {hits / runs:.0%}")
print(f"the independence formula predicted {1 - 0.95 ** len(METRICS):.0%}")
""")

md("""
About **43%** — measured over 2000 runs it settles at 0.428, and 300 runs here will land
near it. The independence formula said 54%. The gap is the correlation: solve
`1 − 0.95ⁿ = 0.43` for n and you get about **11**, against an effective-dimensions estimate
of **9**. Somewhere between 9 and 11 real chances, not 15.

**Both directions matter.**

- Assuming independence when there is none makes you *over*-estimate how much you hunted,
  and a baseline that is too high lets a real finding be dismissed.
- Forgetting the baseline entirely makes you believe the first striking thing you see.

The honest version of this section's finding: *"the largest of fifteen differences I looked
at, on labels I did not choose in advance."* That sentence is publishable. "Cohort A is more
active at weekends" is not.
""")

md("""
### The defence: can you predict the future?

Every fix for this is a version of the same thing — commit before you look.

1. **Write the question down first.** Then there is one test, `0.05` means `0.05`, and the
   arithmetic above never starts.
2. **Say how many things you looked at.** "The best of fifteen" is honest and still
   reportable.
3. **Split the data.** Hunt on one half, confirm on the other.

The third is the strongest, because it is the only one a reader can check. It also turns a
claim into a **prediction**: if the pattern is real, it is in data you have not looked at
yet. If it is a fluke, it is not.

Take the night-owl finding to data the hunt never touched: split the corpus by period and
re-measure within each half.
""")

code("""
early = uk[uk.date.dt.year <= 2015]
late = uk[uk.date.dt.year >= 2016]


def night_share(frame: pd.DataFrame) -> pd.Series:
    return frame.assign(night=frame.hh.between(0, 5)).groupby("author").night.mean()


held_out = pd.DataFrame({
    "early": night_share(early),
    "late": night_share(late),
}).join(authors[["cohort"]], how="inner").dropna()

print(f"{len(held_out)} authors appear in both periods")
print()

for period in ("early", "late"):
    a = held_out.loc[held_out.cohort == "A", period]
    b = held_out.loc[held_out.cohort == "B", period]
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    print(f"{period:6s}: A {a.mean():.3f} vs B {b.mean():.3f}   p = {p:.3f}")
""")

md("""
`p = 0.77` and `p = 0.39`. The gap does not survive either half. It was never there — the
labels are random, so nothing about them could predict anything about a period the hunt did
not touch.

Two details worth noticing. The split costs you authors: only those active in both periods
can be compared, so 442 becomes 133. And the full-period difference was not a rounding
error — `p = 0.0026` is the kind of number people quote. Size and significance both survived
right up until the moment the finding had to predict something.

That is what a held-out check buys you, and it is why "can you predict the future?" is the
best single question to ask a finding. A pattern that only exists in the data you searched
is a pattern you made.

> **The rule to carry:** a finding and the search that produced it are one object. Report
> them together or you have not reported the finding.
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
