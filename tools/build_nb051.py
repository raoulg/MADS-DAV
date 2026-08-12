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

Then one model, because two of those seven claims turn out to be the trend and the residual
of the same series.
""")

code("""
import numpy as np
import pandas as pd
import seaborn as sns
from goad_toolkit.visualizer import (
    BarbellPlot,
    CorrelationHeatmap,
    DecomposePlot,
    HeatmapPlot,
    LinePlot,
    PlotSettings,
    RegPlot,
    ScatterPlot,
)
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


# ---------------------------------------------------------------- 5.4 credibility grid
md("""
## 5.4 What would make you believe it

The previous section is only half a lesson. Run it alone and the conclusion is "nothing is
ever real", which is worse than the credulity it was meant to cure — a student who believes
everything and a student who believes nothing both stop thinking.

So: what *would* make a finding believable?

A claim rests on three legs, and statistics supplies exactly one of them.

1. **Evidence** — how many observations, *at the unit the claim is about*, how big is the
   effect, and does it survive a null test.
2. **Mechanism** — is there a reason it would be true, and would you have predicted the
   direction before looking? **No amount of data supplies this leg.**
3. **Replication** — does it appear where you did not look? Another period, another channel,
   a second way of measuring the same thing. Three mediocre independent confirmations beat
   one excellent p-value.

Two of those are cheap to check. Cross them:

|  | **survives testing** | **fails testing** |
| -- | -- | -- |
| **mechanism** | Report it — then go and check leg three. | **"Plausible, unproven."** Say exactly that, and say what would settle it. |
| **no mechanism** | **Do not conclude. Go looking.** A fluke, or a confounder you have not named — and the confounder is usually the more interesting finding. | Nothing here. Drop it. |

The diagonal is obvious. The off-diagonal cells are the lesson, and both are things students
almost never write down: a null result stated as a null result, and a significant result
treated as a question rather than an answer.

Seven claims, run through the grid. **The right answer is different every time** — which is
the point. If scepticism were the lesson, one example would do.
""")

md("""
### Claim 1 · "There are far fewer messages at 4am"

Mechanism: overwhelming. Evidence: enormous. Both legs, no argument.
""")

code("""
hourly = uk.hh.value_counts(normalize=True).sort_index()
peak_hour = hourly.idxmax()

print(f"04:00      {hourly[4]:.1%} of all messages")
print(f"{peak_hour}:00 (peak) {hourly[peak_hour]:.1%}")
print(f"ratio      {hourly[peak_hour] / hourly[4]:.0f}x")
""")

md("""
Fifteen to one, on half a million messages. Unarguable, and **worthless**.

The grid has a fifth verdict it does not draw, because it sits outside the grid entirely:
**true, well-evidenced, and not worth reporting.** You have shown that people sleep. No
reader's beliefs move.

The question that rescues it: *what did you expect, and where does the data differ from
that?* "Fewest messages at 4am" is the expectation. "The 4am dip is 15× on `#ubuntu-uk` and
1.7× on `#ubuntu`" is a finding, because now something varies and the variation needs
explaining. This is lesson 3's move — the daily cycle is the boring part you subtract, not
the result.
""")

md("""
### Claim 2 · "People stay up late for an Ubuntu release"

A prediction, written before looking, which is what makes it worth testing. Mechanism:
plausible — a release is an event, people wait for it, downloads and problems arrive at once.

Ubuntu releases on a Thursday, so ordinary **Thursdays** are the fair comparison. Anything
else confounds the release with the weekly cycle.
""")

code("""
RELEASES = pd.to_datetime([
    "2013-04-25", "2013-10-17", "2014-04-17", "2014-10-23", "2015-04-23",
    "2015-10-22", "2016-04-21", "2016-10-13", "2017-04-13", "2017-10-19",
])

thursdays = uk[uk.date.dt.dayofweek == 3].copy()
thursdays["is_release"] = thursdays.date.isin(RELEASES)

ordinary = thursdays[~thursdays.is_release]
release = thursdays[thursdays.is_release]

print(f"{ordinary.date.nunique()} ordinary Thursdays, {release.date.nunique()} release days")
print(f"night share (00-05)  ordinary {ordinary.hh.between(0, 5).mean():.1%}"
      f"   release {release.hh.between(0, 5).mean():.1%}")
""")

md("""
Refuted, and firmly: the night share **falls** on release days, 6.8% to 1.0%. People do not
stay up. A prediction that dies is worth more than three that survive, because it is the only
kind that could have gone either way.

But look at that comparison for a second longer, because it is doing something dishonest.
Those are pooled means — every message from 251 Thursdays thrown into one bucket. Ask the
same question per day.
""")

code("""
per_day = thursdays.groupby(["date", "is_release"]).agg(
    n=("hh", "size"),
    night=("hh", lambda h: h.between(0, 5).mean()),
    afternoon=("hh", lambda h: h.between(13, 16).mean()),
).reset_index()

typical = per_day[~per_day.is_release]
print(f"ordinary Thursday, night share:      mean {typical.night.mean():.1%}"
      f"   median {typical.night.median():.1%}")
print(f"ordinary Thursday, afternoon share:  mean {typical.afternoon.mean():.1%}"
      f"   median {typical.afternoon.median():.1%}")
print()
print(per_day[per_day.is_release][["date", "n", "night", "afternoon"]].round(3).to_string(index=False))
""")

md("""
The typical ordinary Thursday has a night share of **1.2%**, not 6.8%. The pooled mean was
inflated by a handful of unusual nights — Anscombe again, one section later and on a
statistic nobody thinks of as fragile. Against the median, the release days are unremarkable.
The prediction is still refuted, but the evidence for refuting it is much weaker than the
first cell implied.

What *does* survive is a different pattern, visible in the same table: the afternoon.
""")

code("""
above = (per_day.loc[per_day.is_release, "afternoon"] > typical.afternoon.median()).sum()
n_releases = per_day.is_release.sum()
sign_p = stats.binomtest(int(above), int(n_releases), 0.5).pvalue

print(f"{above}/{n_releases} release days are above the ordinary-Thursday median "
      f"afternoon share ({typical.afternoon.median():.1%})")
print(f"sign test p = {sign_p:.3f}")
""")

code("""
profile = pd.concat([
    ordinary.hh.value_counts(normalize=True).rename("share").reset_index().assign(day="ordinary Thursday"),
    release.hh.value_counts(normalize=True).rename("share").reset_index().assign(day="release day"),
])

shape = PlotSettings(
    figsize=(10, 4),
    title="A release does not lengthen the day, it moves it",
    xlabel="hour (UTC)",
    ylabel="share of the day's messages",
)
fig, ax = LinePlot(shape).plot(data=profile.sort_values("hh"), x="hh", y="share",
                               hue="day", marker="o")
ax.legend(title="")
""")

md("""
Nine of ten release days sit above the ordinary median, `p = 0.021` by a sign test — and a
sign test is the right instrument here, because it asks only *which side of typical* each day
falls on, which is a question ten noisy days can actually answer.

**Verdict: report it, restated.** Not "people stay up for a release" but *"a release moves
the channel's activity into the afternoon."* And do not go further than that: the pooled
profile peaks at 15:00, but only two of the ten individual release days do. The aggregate
shape is real; the sentence "the peak moves to 15:00" is about a day that does not exist.
""")

md("""
### Claim 3 · "People write longer messages on weekdays"

Mechanism: real, and **ambiguous in sign**. Weekday chat happens at work, in short bursts
between other things — that predicts shorter. Weekday chat is also about work, technical and
detailed — that predicts longer. You genuinely cannot call it in advance.

This is the case where a test earns its keep. Without a claim like this, students conclude
statistics is a formality, because every other example was decidable by thinking.
""")

code("""
weekday = uk.loc[~uk.is_weekend, "length"]
weekend = uk.loc[uk.is_weekend, "length"]

print("one row per MESSAGE")
print(f"  weekday {weekday.mean():.2f} chars (n={len(weekday):,})")
print(f"  weekend {weekend.mean():.2f} chars (n={len(weekend):,})")
print(f"  p = {stats.ttest_ind(weekday, weekend, equal_var=False).pvalue:.1e}")
""")

md("""
`p = 8e-10`. A 1.05-character difference, on half a million messages, in the direction
opposite to the claim.

That p-value is meaningless, and lesson 2 said why: the claim is about **people**, and
425,286 messages are not 425,286 independent observations of people. Ask it at the unit the
claim is about.
""")

code("""
by_author = uk.groupby(["author", "is_weekend"]).length.mean().unstack().dropna()
by_author = by_author[uk.groupby("author").size().reindex(by_author.index) >= 30]
by_author.columns = ["weekday", "weekend"]

paired = stats.ttest_rel(by_author.weekday, by_author.weekend)
longer_on_weekdays = (by_author.weekday > by_author.weekend).sum()

print(f"one row per AUTHOR (n={len(by_author)} with >=30 messages, active on both)")
print(f"  weekday {by_author.weekday.mean():.2f} chars   weekend {by_author.weekend.mean():.2f}")
print(f"  mean within-author difference {(by_author.weekday - by_author.weekend).mean():+.2f} chars")
print(f"  paired t-test p = {paired.pvalue:.3f}")
print(f"  authors longer on weekdays: {longer_on_weekdays}/{len(by_author)} "
      f"({longer_on_weekdays / len(by_author):.0%})")
""")

md("""
259 people, `p = 0.06`, and **53% of them go one way while 47% go the other**. That last
number is the one to look at: it is a coin flip. Whatever is happening is not something
individual people do.

**Verdict: plausible, unproven.** Write that sentence down, in a report, as the result:

> *"Message length does not differ meaningfully between weekdays and weekends at the author
> level (n = 259, p = 0.06, and the direction splits 53/47). The per-message difference is
> significant but reflects message counts, not people. Distinguishing them would need a
> dataset with more authors, or a within-person design across more weeks."*

That is a finished piece of work. Students almost never produce it, because a null reads like
a failed assignment — so they keep slicing until something turns up, which is exactly the
hunt from §5.3. **A null result, honestly bounded, is a pass.**
""")

md("""
### Claim 4 · "Nicknames starting A–M write longer messages"

No mechanism. Nobody has a theory about the alphabet, which is precisely why this one is
safe to use — nobody gets defensive defending it.

Hunt it properly: two channels, five years, ten places to look.
""")

code("""
def alphabet_split(frame: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Mean message length per author, tagged by whether the nick starts A-M.\"\"\"
    frame = frame.assign(length=frame.message.str.len())
    out = frame.groupby("author").agg(length=("length", "mean"), n=("length", "size"))
    out = out[out.n >= 30]
    out["early"] = pd.Series(out.index, index=out.index).str.upper().str[0].between("A", "M")
    return out


hunt = []
for (channel, year), cell in msgs.groupby([msgs.channel, msgs.date.dt.year]):
    table = alphabet_split(cell)
    a, b = table.loc[table.early, "length"], table.loc[~table.early, "length"]
    hunt.append({"channel": channel, "year": year, "n_AM": len(a), "n_NZ": len(b),
                 "A-M": a.mean(), "N-Z": b.mean(),
                 "p": stats.ttest_ind(a, b, equal_var=False).pvalue})

hunt = pd.DataFrame(hunt)
print(f"{len(hunt)} places looked, {(hunt.p < 0.05).sum()} significant at 0.05")
print(f"N-Z longer in {(hunt['N-Z'] > hunt['A-M']).sum()}/{len(hunt)} of them")
hunt.round(3)
""")

md("""
Nothing clears the line — the closest is `p = 0.053` on `#ubuntu-uk` in 2016, where the gap
looks big enough to write up: 52.1 against 61.2 characters, 17%. Ten tests should hand you a
false positive about half the time; this time it did not, which is a useful thing to see once.

The interesting part is the last line. **N–Z looks longer in eight of the ten cells.** Eight
out of ten, in the same direction, on labels nobody assigned — that has the shape of
replication, and a student who has just learned about leg three will reach for it.

It is not replication, for the same reason the fifteen metrics were not fifteen chances.
""")

code("""
per_year = {year: set(alphabet_split(cell).index)
            for year, cell in msgs.groupby(msgs.date.dt.year)}

print("share of each year's authors who were already there the year before:")
for earlier, later in zip(sorted(per_year), sorted(per_year)[1:]):
    overlap = len(per_year[earlier] & per_year[later]) / len(per_year[later])
    print(f"  {earlier} -> {later}: {overlap:.0%}")
""")

md("""
Half the authors in each year are the same people as the year before, and a person's typing
habits do not reset in January. The ten cells are not ten independent looks; they are one
look, re-photographed. If `daftykins` writes long messages and starts with a D, that fact is
in every cell.

**Verdict: drop it.** No mechanism, no evidence, and the thing that looked like confirmation
was the same observation counted ten times.

> **Independence is the assumption that fails quietly.** It failed in §5.3 (correlated
> metrics), it failed in claim 3 (messages within a person), and it failed here (people
> across years). Every time, the effect is the same: you think you have more information than
> you have.
""")

md("""
### Claim 5 · "People on the 4th floor have worse sentiment"

Not from this corpus — a real submission from a previous cohort, and the most useful of the
seven.

**Evidence:** every set of numbers has a maximum, and a minimum. Attach a sentiment score to
five floors and one of them comes last. That is arithmetic, not a finding.

**Mechanism for *floor → mood*:** none. Nothing about being 12 metres up makes a person
unhappy.

Bottom-left cell, then: **do not conclude — go looking.** And here is why that cell is not a
polite way of saying "drop it". There is no mechanism from the floor, but there is an obvious
one from what the floor *stands for*: floors hold departments, departments have different
work, deadlines, managers and hours. The floor number is a proxy for something real, and
nobody wrote it down.

**No mechanism does not mean no finding. It means you have not found it yet.** The confounder
you are hunting is usually more interesting than the claim you started with — "sales is
having a bad quarter" is a better result than "the 4th floor is grumpy", and it is the same
data.

The follow-up question is always the same shape: *what else is true of the 4th floor?*
""")

md("""
### Claim 6 · "The channel got quieter after 2016"

Mechanism: strong, and predictable in advance — Slack, Discord and Matrix happened to IRC
everywhere, not just here. Direction callable before looking, which is what separates this
from claim 4.
""")

code("""
per_year_counts = msgs.groupby([msgs.date.dt.year.rename("year"), "channel"]).size().unstack()

print("messages per year")
print(per_year_counts.to_string())
print()
print("relative to 2013")
print((per_year_counts / per_year_counts.loc[2013]).round(2).to_string())
""")

md("""
`#ubuntu-uk` ends at **16%** of its 2013 volume, `#ubuntu-nl` at **8%**, and both fall in
every single year after 2014. Two channels, different countries, different sizes, same
direction, no exceptions.

**All three legs.** Mechanism predicted before looking, effect enormous and monotone,
replicated on an independent channel. Nothing here needs a p-value, and asking for one would
be a category error — you are not distinguishing this from noise, you are looking at it.

The honest caveat belongs in the same paragraph: this is *this corpus* getting quieter, which
is not automatically *IRC* getting quieter, and definitely not *the Ubuntu community*
shrinking. People moved. The claim is about a channel.
""")

md("""
### Claim 7 · "Release days are busier"

Mechanism: strong. Dates known **before** looking, from an external calendar this dataset had
no say in. Ten instances, spread over five years.

The obvious test is the release day against a typical Thursday.
""")

code("""
thursday_counts = uk[uk.date.dt.dayofweek == 3].groupby("date").size().sort_index()
global_median = thursday_counts[~thursday_counts.index.isin(RELEASES)].median()
release_counts = thursday_counts[thursday_counts.index.isin(RELEASES)]

print(f"median ordinary Thursday, whole corpus: {global_median:.0f} messages")
print(f"release days above it: {(release_counts > global_median).sum()}/{len(release_counts)}")
""")

md("""
Six out of ten. That is a coin flip, and on its own it kills the claim.

It should not. Claim 6 is the reason: the channel lost 84% of its traffic across the window,
so "a typical Thursday" in 2013 and in 2017 are different quantities, and the 2017 releases
are being compared against a median that mostly comes from 2013. **The trend is a confounder
for the event.**

The fix is to compare each release with the Thursdays around it.
""")

code("""
def local_baseline(frame: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    \"\"\"Each release against the median of the Thursdays within `weeks` either side.\"\"\"
    counts = frame[frame.date.dt.dayofweek == 3].groupby("date").size().sort_index()
    rows = []
    for release_date in RELEASES:
        if release_date not in counts.index:
            continue
        window = counts[(counts.index >= release_date - pd.Timedelta(weeks=weeks))
                        & (counts.index <= release_date + pd.Timedelta(weeks=weeks))]
        rows.append({"release": str(release_date.date()),
                     "nearby Thursday": window.drop(release_date).median(),
                     "on the day": counts[release_date]})
    out = pd.DataFrame(rows)
    out["lift"] = out["on the day"] / out["nearby Thursday"]
    return out


for channel in sorted(msgs.channel.unique()):
    table = local_baseline(msgs[msgs.channel == channel])
    above = int((table.lift > 1).sum())
    p = stats.binomtest(above, len(table), 0.5).pvalue
    print(f"{channel}: {above}/{len(table)} releases above their local baseline, "
          f"median lift {table.lift.median():.2f}x, sign test p = {p:.3f}")

local = local_baseline(uk)
local.round(2)
""")

code("""
gap = PlotSettings(
    figsize=(9, 5),
    title="Each release against the Thursdays around it",
    xlabel="messages on the day",
    ylabel="",
)
fig, ax = BarbellPlot(gap).plot(
    data=local.sort_values("release", ascending=False), category="release",
    start="nearby Thursday", end="on the day",
    start_label="nearby Thursdays (median)", end_label="release day",
)
""")

md("""
**Nine of ten**, median lift 1.7×, `p = 0.02`. Same data, same claim, and the only thing that
changed was what "typical" means.

**Verdict: the gold standard, and the closest this notebook gets to a finding you could
publish.** Mechanism strong, dates fixed in advance by someone else, ten near-independent
instances.

Leg three is partial, and say so: `#ubuntu-nl` moves the same way — 6 of 9, median lift
1.5× — but on its own that is `p = 0.51`, which is no evidence at all. A tenth of the traffic
buys a tenth of the resolution. "The direction agrees on a second channel, which is too small
to test" is the accurate sentence, and it is worth more than either "replicated" or
"not replicated".

And the detour is the real lesson. The first version of this test said *six out of ten* and
would have been reported as "no effect". A finding can be destroyed by a baseline as easily as
it can be invented by one, which is why §5.3's rule needs its mirror image: **the comparison
you chose is part of the claim.**
""")

md("""
### The seven, side by side

| claim | mechanism | evidence | verdict |
| -- | -- | -- | -- |
| 1 · fewer messages at 4am | overwhelming | 15× | **true, not worth reporting** — what did you expect? |
| 2 · people stay up for a release | plausible | refuted | **restate it**: the day moves to the afternoon |
| 3 · longer messages on weekdays | real, ambiguous | p = 0.06, 53/47 | **plausible, unproven** — and that is the report |
| 4 · A–M nicks write longer | none | nothing, in ten places | **drop it** |
| 5 · the 4th floor is grumpier | none *for the floor* | a minimum exists | **go looking** — the floor is a proxy |
| 6 · quieter after 2016 | strong, predicted | 0.16× and 0.08×, monotone | **report it** — all three legs |
| 7 · release days are busier | strong, dates external | 9/10, p = 0.02 | **report it** — after fixing the baseline |

Seven claims, six different answers. "Be sceptical" would have got one of them right.

> **What to carry out of this section.** Before you write a finding down, say out loud: what
> is the mechanism, what would have surprised me, and where could I check this that I have
> not already looked? If the answer to the first is "none", you are not finished — you are at
> the beginning of a more interesting question.
""")


# ---------------------------------------------------------------- 5.5 fingerprints
md("""
## 5.5 Eight people, and how they type

One claim, taken all the way through the grid.

> **People are more identifiable by *how* they write than by *what* they write about.**

Mechanism: plausible, and it predicts something specific — habits should be stable while
subject matter drifts. Evidence: this is what the rest of the section builds. Replication:
there is a four-year gap in this corpus to check it on.

The eight busiest people on `#ubuntu-uk`. Action lines (`/me`) are dropped: they have a
different grammar and would be a free giveaway.
""")

code("""
people = uk[~uk.is_action].copy()
top8 = people.author.value_counts().head(8)
people = people[people.author.isin(top8.index)]

print(top8.to_string())
print(f"\\n{len(people):,} messages, {people.author.nunique()} authors")
print(f"guessing the most frequent author every time = {top8.iloc[0] / top8.sum():.1%}")
print(f"guessing at random                           = {1 / len(top8):.1%}")
""")

md("""
### The classifier is an instrument, not the result

Fit a model that reads one message and names its author. **The accuracy is a gate, not a
finding** — it tells you whether there is anything to look at. What the model *learned* is
the finding, and that lives in the weights.
""")

code("""
train_msgs, test_msgs = train_test_split(
    people, test_size=0.25, random_state=42, stratify=people.author)

words = TfidfVectorizer(min_df=5, sublinear_tf=True)
one_liner = LogisticRegression(max_iter=1000, C=5)
one_liner.fit(words.fit_transform(train_msgs.message), train_msgs.author)

accuracy = one_liner.score(words.transform(test_msgs.message), test_msgs.author)
print(f"{len(words.vocabulary_):,} features, one message at a time")
print(f"accuracy {accuracy:.1%}  against a {top8.iloc[0] / top8.sum():.1%} baseline")
""")

md("""
47% on eight-way classification from a single IRC line, against a 22% baseline. Twice the
baseline is a gate comfortably passed and nowhere near a destination — half the messages are
still attributed to the wrong person.

Now the actual question: **what is it using?**
""")

code("""
vocabulary = np.array(words.get_feature_names_out())
for i, author in enumerate(one_liner.classes_):
    strongest = vocabulary[np.argsort(one_liner.coef_[i])[::-1][:8]]
    print(f"{author:12s} {', '.join(strongest)}")
""")

md("""
Read those rows and they are not all the same kind of thing:

- `ali1234` — `n900`, `qml`, `pidgin`, `mythtv`. **Subject matter.** The model knows what he
  works on.
- `MartijnVdS` — `mungbean`, `dimpy`, `neuro`, `paladine`. **Other people's nicknames.** Not
  a writing habit at all; a social position.
- `zmoylan-pi` — `ireland`, `dublin`. **Location.**
- `daftykins` — `0o`, `xd`, `hrmm`, `8d`. **Typing.**
- `popey` — `dont`, `didnt`, `thats`, `wont`. **Typing** — specifically, missing apostrophes.

Only the last two are the claim. The others are a topic detector wearing a stylometry
costume, and a topic detector fails the moment somebody changes subject — which is exactly
the situation an authorship model exists for.

> **The verification step, and it is the habit worth stealing from this section:** when a
> model works, read its weights and ask *what each one actually is*. A model can be right for
> a reason that does not generalise, and the accuracy will never tell you.

So stop using the model to find the fingerprint. Measure the habits directly, and let the
model come back later as a check.
""")

md("""
### Four habits, measured

Nothing here needs machine learning. Regular expressions and a `groupby` — the same tools as
lesson 1.

Start with smileys, where there are three ways to write the same thing.
""")

code("""
text = people.message
habits = people.assign(
    nosed=text.str.count(r"[:;=]-[)DPp(\\]]"),
    noseless=text.str.count(r"[:;=][)DPp(\\]]|\\bXD\\b"),
    unicode_smiley=text.str.count(r"[☺☻☹㋛]"),
    apos_drop=text.str.count(r"(?i)\\b(?:dont|doesnt|didnt|cant|wont|isnt|im|thats|its|ive|youre)\\b"),
    apos_keep=text.str.count(r"(?i)\\b(?:don't|doesn't|didn't|can't|won't|isn't|i'm|that's|it's|i've|you're)\\b"),
    starts_upper=text.str.match(r"^[A-Z]").astype(float),
    addresses=text.str.match(r"^\\S+[:,]\\s").astype(float),
    night=people.hh.between(0, 5).astype(float),
    length=text.str.len(),
)

per_author = habits.groupby("author")
smileys = per_author[["nosed", "noseless", "unicode_smiley"]].sum()
dialect = smileys.div(smileys.sum(axis=1), axis=0) * 100

print("share of that author's smileys, by dialect (%)")
dialect.round(1)
""")

md("""
Three dialects, and **nobody mixes them**. `zmoylan-pi` writes the nose 99.4% of the time,
`diddledan` 97.4%, and the other six essentially never — the highest nosed share among them
is 0.2%. There is nobody between 0.2% and 97.4%.

That is not what a habit usually looks like. Most measurable differences between people are
matters of degree — one person is somewhat more likely to do something. This is a **discrete
choice**, made once and then held for five years, by people sitting in the same channel
reading each other's messages every day.

`popey` is the third dialect on his own: 57.6% of his smileys are ☺, a character the others
never type once.
""")

code("""
summary = pd.DataFrame({
    "apostrophes dropped %": per_author.apos_drop.sum()
    / (per_author.apos_drop.sum() + per_author.apos_keep.sum()) * 100,
    "starts with a capital %": per_author.starts_upper.mean() * 100,
    "addresses someone %": per_author.addresses.mean() * 100,
    "messages at 00-05h %": per_author.night.mean() * 100,
    "median length": per_author.message.apply(lambda s: s.str.len().median()),
})
summary.round(1)
""")

md("""
Every column has the same shape as the smileys: a couple of people at one extreme and
everybody else clustered at the other.

- **Apostrophes.** `popey` drops 38% of his, `foobarry` 36%. Everyone else is between 0.6%
  and 3.7%. Again nothing in the middle.
- **Clocks.** `daftykins` does 15% of his talking between midnight and 05:00. `foobarry` has
  posted in that window exactly zero times in five years. Not "rarely" — zero.
- **Addressing.** `MartijnVdS` opens 49% of his messages with someone's nick, `zmoylan-pi`
  2.9%. That is the thing the classifier found, seen properly: it is not that he *writes*
  differently, it is that his role in the channel is answering people.

The channel average hides all of this. There is no such thing as a typical `#ubuntu-uk`
author.
""")

code("""
fingerprint = pd.DataFrame({
    "nosed :-)": dialect.nosed,
    "unicode ☺": dialect.unicode_smiley,
    "no apostrophe": summary["apostrophes dropped %"],
    "starts capital": summary["starts with a capital %"],
    "addresses": summary["addresses someone %"],
    "00-05h": summary["messages at 00-05h %"],
})

marks = PlotSettings(
    figsize=(10, 5),
    title="Eight fingerprints (% of that author's messages)",
    xlabel="",
    ylabel="",
)
fig, ax = HeatmapPlot(marks).plot(data=fingerprint, annot=True, fmt=".0f", cmap="rocket_r")
""")

md("""
### Why the classifier only reached 47%, when the habits look this decisive

Both of those are true at once, and the reason is the same unit-of-analysis question that has
run through the whole notebook.
""")

code("""
has_smiley = habits[["nosed", "noseless", "unicode_smiley"]].sum(axis=1) > 0
has_contraction = habits[["apos_drop", "apos_keep"]].sum(axis=1) > 0

print(f"messages containing a smiley       {has_smiley.mean():.1%}")
print(f"messages containing a contraction  {has_contraction.mean():.1%}")
print(f"messages containing neither        {(~has_smiley & ~has_contraction).mean():.1%}")
""")

md("""
**Three quarters of messages carry no fingerprint at all.** The habits are near-deterministic
*when they appear*, and most single lines are `ok`, `thanks`, `brb`. A model reading one line
usually has nothing to go on, so 47% is what a strong signal looks like when it is sparse.

The fix is to stop asking about one message. Pool fifty of them per person and ask again.
""")

code("""
FEATURES = ["nosed", "noseless", "unicode_smiley", "apos_drop", "apos_keep",
            "starts_upper", "addresses", "night", "length", "n_question"]
BLOCK = 50

shuffled = habits.sample(frac=1, random_state=0)
shuffled["block"] = shuffled.groupby("author").cumcount() // BLOCK
grouped = shuffled.groupby(["author", "block"])

blocks = grouped[FEATURES].mean()
blocks["text"] = grouped.message.apply(" ".join)
blocks = blocks[grouped.size() == BLOCK].reset_index()

print(f"{len(blocks):,} blocks of {BLOCK} messages")
print(blocks.author.value_counts().to_string())
""")

code("""
train, test = train_test_split(blocks, test_size=0.25, random_state=42, stratify=blocks.author)

vec = TfidfVectorizer(min_df=3, sublinear_tf=True)
vocabulary_model = LogisticRegression(max_iter=1000, C=5)
vocabulary_model.fit(vec.fit_transform(train.text), train.author)

scaler = StandardScaler().fit(train[FEATURES])
habit_model = LogisticRegression(max_iter=2000)
habit_model.fit(scaler.transform(train[FEATURES]), train.author)

print(f"every word they used : {len(vec.vocabulary_):>6,} features   "
      f"accuracy {vocabulary_model.score(vec.transform(test.text), test.author):.3f}")
print(f"ten habits           : {len(FEATURES):>6} features   "
      f"accuracy {habit_model.score(scaler.transform(test[FEATURES]), test.author):.3f}")
print(f"baseline             : {' ' * 6}            "
      f"          {top8.iloc[0] / top8.sum():.3f}")
""")

md("""
**Ten numbers do what seventeen thousand words do.** Both models are at or near ceiling on a
block of fifty messages, and the one that knows nothing about vocabulary — no topics, no
nicknames, no place names — is within half a point of the one that knows everything.

That is the claim, demonstrated: identity is in the habits, not the subject matter. It took
volume to see, which is the honest caveat that goes in the same sentence.

And because there are ten features rather than seventeen thousand, the weights are readable.
""")

code("""
weights = pd.DataFrame(habit_model.coef_, index=habit_model.classes_, columns=FEATURES)
weights.round(1)
""")

md("""
Each row is a person, in the model's own words:

- `MartijnVdS` — **addresses** `+3.4`, everything else near zero. The role, again.
- `daftykins` — **noseless** `+4.4`, **night** `+2.3`. Types `:)` and is awake at 3am.
- `foobarry` — **night** `−4.0`. Identified largely by *never* being there.
- `popey` — **unicode** `+1.9`, **apostrophes dropped** `+1.4`, **length** `−2.2`.
- `bigcalm` — **starts with a capital** `+3.8`, alone among the eight.

Compare that with the tf-idf model's rows. Same task, same data, and one of them can be read
aloud.
""")

md("""
### Leg three: does it hold when nobody is looking?

The corpus spans five years. Fit nothing — just measure the same habits in 2013–2014 and
again in 2016–2017, and see whether people are still themselves.
""")

code("""
early = habits[habits.date.dt.year <= 2014]
late = habits[habits.date.dt.year >= 2016]
print(f"early {len(early):,} messages (2013-2014), late {len(late):,} (2016-2017)")


def stability(column: str, how: str = "mean") -> float:
    \"\"\"Spearman across the eight authors between their early and late value.\"\"\"
    pair = pd.concat([early.groupby("author")[column].agg(how),
                      late.groupby("author")[column].agg(how)], axis=1).dropna()
    return pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")


for label, column, how in [("addresses by nick", "addresses", "mean"),
                           ("starts with a capital", "starts_upper", "mean"),
                           ("median message length", "length", "median"),
                           ("asks questions", "n_question", "mean")]:
    print(f"{label:24s} spearman {stability(column, how):.2f}")
""")

md("""
0.98, 0.95, 0.93 — and then **0.55**.

The first three are as close to "the same people" as this kind of measurement gets: the
ordering of eight people on a habit is unchanged after a four-year gap, on messages nobody
was thinking about when the habit formed. Leg three, cleanly.

The fourth is the useful one. **Question rate is not a trait.** Whether you ask questions
depends on whether you currently have a problem, and that changes — so it moves people around
the ranking while the typing habits hold them still.

A finding that came with its own counter-example is more believable than one that did not,
because it shows the measurement could have said no.
""")

md("""
### And now the objection

Run it through the grid honestly and it lands top-left: mechanism, evidence, replication. But
there is a sentence you cannot write.

**There are eight people here.**

The evidence is 218,670 messages, and every one of the tests above is at the author level for
exactly that reason. But `n = 8`, and the claim — *"people are identified by how they type"* —
is about people in general. Eight is the sample size, and a spearman of 0.98 across eight
points is eight points.

This is the most seductive finding in the course and therefore the best place to say it:
**large data does not fix a small n at the unit your claim is about.** It is lesson 2's
pseudoreplication, arriving at the end of the section that spent its whole length being
careful.

What it would take to fix: more authors. The habits are cheap to measure — the cells above run
on anybody with thirty messages — so the honest next step is to recompute the stability
correlations across every author with enough history in both periods, and report *that*
number instead of this one.

> **What survives, stated the way it should be reported.** *"Among the eight most active
> `#ubuntu-uk` authors, hand-picked typing habits classify blocks of 50 messages at 99.5%,
> matching a 17,918-feature bag of words, and the habit rankings are preserved across a
> four-year gap (spearman 0.93–0.98) while question rate is not (0.55). Whether this holds for
> less active authors is untested."*

Every clause in that sentence is doing work, and none of it is the word "significant".
""")


# ---------------------------------------------------------------- 5.6 a model of a day
md("""
## 5.6 A model of an ordinary day

Two of §5.4's seven claims were about the same series. "The channel got quieter after 2016"
is a statement about its **trend**. "Release days are busier" is a statement about its
**residual**. Both were tested by picking a comparison by hand — a ratio to 2013, the
Thursdays either side.

Write the model down instead, and the comparisons stop being choices.

    messages on a day = trend  x  what day of the week it is  x  whatever is left

Three lines of arithmetic, and the third line is the one you keep.
""")

code("""
daily = uk.groupby("date").size().rename("messages").to_frame()
span = pd.date_range(daily.index.min(), daily.index.max(), freq="D")

print(f"{len(daily):,} days with messages, {len(span):,} days in the span")
print(f"quietest day {daily.messages.min()}, busiest {daily.messages.max():,}")
""")

md("""
No gaps and no empty days, which is worth checking before modelling a series rather than
after: a missing day and a zero day mean different things, and both would be silently wrong
here.

**Work in logs.** The three effects above multiply — a quiet Sunday in 2013 and a quiet
Sunday in 2017 are both "about half a normal day", not both "about 200 messages fewer".
Taking logs turns multiplication into addition, which is what lets the three lines be
*subtracted* one at a time.
""")

code("""
daily["day_number"] = np.arange(len(daily))
daily["weekday"] = daily.index.dayofweek
daily["log_messages"] = np.log(daily.messages)

levels = PlotSettings(
    figsize=(13, 3.4),
    title="The same series, twice",
    subplot_titles=["messages per day", "log(messages per day)"],
    xlabel="",
    ylabel="",
)
host = LinePlot(levels)
fig, axes = host.create_figure(n_plots=2)
for ax, column in zip(axes, ["messages", "log_messages"]):
    host.plot_on_axes(LinePlot(levels), ax, data=daily.reset_index(),
                      x="date", y=column, lw=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("")
fig.tight_layout()
""")

md("""
On the raw scale the first year shouts and the last year is a flat line near zero — you
cannot see whether 2017 has any structure at all. In logs, every year gets the same vertical
space, and the decline turns into something a straight line can describe.

**Step one: the trend.**
""")

code("""
trend_fit = stats.linregress(daily.day_number, daily.log_messages)
daily["trend"] = trend_fit.intercept + trend_fit.slope * daily.day_number

print(f"slope {trend_fit.slope:+.5f} log-units per day   (r^2 = {trend_fit.rvalue ** 2:.2f})")
print(f"per year        x{np.exp(trend_fit.slope * 365):.2f}  "
      f"({(np.exp(trend_fit.slope * 365) - 1) * 100:+.0f}%)")
print(f"over five years x{np.exp(trend_fit.slope * len(daily)):.2f}")
""")

md("""
**Down 40% a year, every year.** That is §5.4's claim 6 again, and notice what changed by
writing it as a model: "16% of its 2013 volume" was a ratio between two years somebody chose,
and this is a rate that every day in the corpus contributed to. The second one survives
somebody asking "why 2013?".

**Step two: the week.** Subtract the trend, then ask what each weekday does to what is left.
""")

code("""
detrended = daily.log_messages - daily.trend
weekly = detrended.groupby(daily.weekday).mean()
weekly = weekly - weekly.mean()
daily["seasonal"] = daily.weekday.map(weekly)

names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for day, effect in weekly.items():
    print(f"{names[day]}  x{np.exp(effect):.2f}")
""")

md("""
Five weekdays within a hair of each other at about ×1.3, and then Saturday ×0.48, Sunday
×0.55. **The weekend is not a dip, it is a different channel** — less than half the traffic,
and lesson 3 found the shape behind it: `#ubuntu-uk` peaks at the start of UK working hours
on weekdays and in the evening at weekends. People are chatting from work.

**Step three: whatever is left.**
""")

code("""
daily["residual"] = daily.log_messages - daily.trend - daily.seasonal

variance = pd.Series({
    "log messages": daily.log_messages.var(),
    "minus trend": detrended.var(),
    "minus trend and weekday": daily.residual.var(),
})
print(variance.round(3).to_string())
print(f"\\ntrend accounts for  {1 - variance.iloc[1] / variance.iloc[0]:.0%}")
print(f"weekday adds        {(variance.iloc[1] - variance.iloc[2]) / variance.iloc[0]:.0%}")
print(f"residual sd {daily.residual.std():.2f} log-units "
      f"— a typical day lands within x{np.exp(daily.residual.std()):.1f} of the model")
""")

code("""
parts = PlotSettings(
    figsize=(13, 6),
    title="messages per day = trend x weekday x residual",
    subplot_titles=["observed (log)", "trend + weekday", "residual"],
    xlabel="",
    ylabel="",
    max_cols=1,
)
host = LinePlot(parts)
fig, axes = host.create_figure(n_plots=3)

frame = daily.assign(fitted=daily.trend + daily.seasonal).reset_index()
for ax, column in zip(axes, ["log_messages", "fitted", "residual"]):
    host.plot_on_axes(LinePlot(parts), ax, data=frame, x="date", y=column, lw=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("log messages")
axes[2].axhline(0, color="grey", lw=1)
for release_date in RELEASES:
    axes[2].axvline(release_date, color="crimson", lw=1, alpha=0.6)
fig.tight_layout()
""")

md("""
Trend and weekday together account for a bit over half the variance, so nearly half of it is
still in that bottom panel: the model is not a good predictor of any particular day, and it
was never meant to be. Its job was to remove the part you could have written down in advance.

The red lines are the ten release dates. Ask the residual about them.
""")

code("""
daily["is_release"] = daily.index.isin(RELEASES)
percentile = daily.residual.rank(pct=True)

for release_date in RELEASES:
    print(f"{release_date.date()}  x{np.exp(daily.residual[release_date]):.2f}  "
          f"({percentile[release_date]:.0%} of days are below it)")

on, off = daily.residual[daily.is_release], daily.residual[~daily.is_release]
print(f"\\nmean residual x{np.exp(on.mean() - off.mean()):.2f} on release days, "
      f"p = {stats.ttest_ind(on, off, equal_var=False).pvalue:.3f}")
""")

md("""
Nine of the ten sit above the median day, the average release day runs **1.7× the model**,
and `p = 0.014`. Same answer as §5.4's local-baseline test, which is the point: the model
replaces the hand-picked comparison, and gets there without anyone choosing which Thursdays
count.

Now the part that matters more than the result.
""")

code("""
biggest = daily.residual.nlargest(15)
for date, value in biggest.items():
    marker = "  <- release" if date in RELEASES else ""
    print(f"{date.date()} {names[date.dayofweek]}  x{np.exp(value):.1f}  "
          f"({daily.messages[date]:>4} messages){marker}")

weekend_days = sum(date.dayofweek >= 5 for date in biggest.index)
print(f"\\n{weekend_days}/15 of the biggest residuals are weekend days, "
      f"against {(daily.weekday >= 5).mean():.0%} of all days")
""")

md("""
**Not one release is in the top fifteen.** The release effect is real — it just is not large
compared with an ordinary weekend that happened to go well.

And thirteen of those fifteen days are Saturdays and Sundays, which is not a discovery about
weekends. It is the model admitting something. A single number per weekday says every Sunday
is 0.55 of a normal day; if Sundays are instead *unpredictable* — sometimes dead, sometimes a
marathon conversation — then a constant cannot fit them and the misfit lands in the residual.

Checkable in one line.
""")

code("""
print("residual sd by weekday")
for day, spread in daily.groupby("weekday").residual.std().items():
    print(f"  {names[day]}  {spread:.2f}")

print("\\nresidual sd by year, against the median day that year")
report = pd.DataFrame({
    "residual sd": daily.groupby(daily.index.year).residual.std(),
    "median messages": daily.groupby(daily.index.year).messages.median(),
})
print(report.round(2).to_string())
""")

md("""
Sunday's residual is nearly twice as spread out as Monday's — 1.01 against 0.62. The weekend
term is not wrong about the average, it is wrong about the *shape*: it moves the mean and
leaves the variance alone, and the variance was the interesting half.

The second table shows the same thing over time: as the channel emptied, the residual grew
from 0.62 to 0.97. The reflex explanation is counting noise — smaller counts are relatively
noisier — and it is worth checking rather than asserting, because it is the kind of
explanation that sounds right.
""")

code("""
quietest_year = report["median messages"].min()
print(f"if days were pure counting noise at {quietest_year:.0f} messages,")
print(f"  expected residual sd = 1/sqrt({quietest_year:.0f}) = {1 / np.sqrt(quietest_year):.2f}")
print(f"  observed in 2017                              = {report['residual sd'].iloc[-1]:.2f}")
""")

md("""
Counting noise would produce 0.13 of the 0.97 observed. The late years are genuinely more
erratic —
a handful of regulars, and whether they show up is the whole story of a 2017 day.

> **What a residual is.** Everything your model does not explain, which is *both* the
> discoveries and the mistakes, mixed together and impossible to tell apart by size. The ten
> release days are a discovery. The thirteen weekend days are a mistake. A residual is a
> to-do list, not a result.

> **Your turn.** Take the biggest residual, `2014-11-16` at ten times the model, and go and
> read that day's messages — the text is in the corpus. Then decide which of the three it is:
> an event worth reporting, a property of weekends the model should have had, or one
> conversation that ran long. Whichever you pick, the check is the same one this whole lesson
> has been about: what would you expect to see if it were the other two?
""")

md("""
### The packaged version

Everything above is `statsmodels`' `seasonal_decompose`, done by hand. Now that you know what
it does, use the short form — and notice it makes exactly the same choices you just made
explicitly, including the one about additive-versus-multiplicative that becomes a `log` on
the way in.
""")

code("""
decomposition = PlotSettings(
    figsize=(12, 8),
    title="seasonal_decompose, period=7",
    xlabel="",
    ylabel="",
    max_cols=1,
)
fig = DecomposePlot(decomposition).plot(data=daily[["log_messages"]], column="log_messages",
                                        period=7)
""")

md("""
Its trend is a 7-day rolling mean rather than a straight line, so it absorbs the slow wobbles
this section left in the residual — a different, equally defensible choice, and one you can
now argue about rather than accept.

---

## What this lesson was

Four moves, and the order was the argument:

1. **Look before you summarise.** Four datasets with identical statistics, and one of them is
   a dinosaur.
2. **Walk into the trap.** Fifteen metrics, random labels, a `p = 0.0026` finding about
   nothing — and the arithmetic that says how often that has to happen.
3. **Rebuild.** Mechanism × evidence, and seven claims that between them produce six different
   verdicts. "Be sceptical" would have got one of them right.
4. **Do it properly once.** Eight people, ten habits, a four-year gap — and an objection
   about `n = 8` that the evidence cannot answer.

One question runs through all four, and it is the one to ask your own work:

> **What would I expect to see if there were nothing here?**

Lesson 4 answered it by shuffling. §5.3 answered it with `1 − 0.95ⁿ`. §5.4 answered it by
choosing a baseline, twice, and getting opposite verdicts. §5.6 answered it by writing the
expectation down as a model and keeping what was left.

They are the same question, and a finding is only worth as much as the answer to it.
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
