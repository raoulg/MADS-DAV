"""Figures for the lesson 4 slides, on the Nocturne palette.

Run from the repo root:  uv run python scripts/plots_lesson4.py
Writes PNGs to presentations/img/lesson4/ and prints the numbers the slides quote.
Data: the taxis, penguins and ubuntu_irc_release_hourly showcases, the Dutch COVID
figures that goad's CovidDataProcessor downloads once to ~/.cache/goad/covid/, and the SiP
effort-estimation dataset, downloaded once to ~/.cache/mads-dav/.
Fits use DistributionFitter(seed=42), as the notebooks do, so the numbers match 04.2–04.5.
"""

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import special, stats

from goad_toolkit.analytics import (
    DistributionFitter,
    FitResult,
    NullDistribution,
    fit_table,
)
from goad_toolkit.config import DataConfig, FileConfig
from goad_toolkit.dataprocessor import CovidDataProcessor
from goad_toolkit.datatransforms import Filter, GroupAgg, Pipeline
from goad_toolkit.distributions import DistributionRegistry
from goad_toolkit.models import linear_model, mse, train_model
from wa_analyzer.data import load_showcase

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentations" / "img" / "lesson4"
OUT.mkdir(parents=True, exist_ok=True)

# pttrn tokens, Paper theme (~/code/packages/pttrn-huisstijl/tokens.json): petrol for the main
# series, vermilion only as a line or a dot, warm greys for everything else
BG = "#f4f1ec"
TEXT = "#111111"
ACCENT = "#1d5a73"
ACCENT_LIGHT = "#5e93a8"
SECOND = "#c43a14"  # vermilion: the family on trial and the observed value, as lines and dots
THIRD = "#8a6d3b"  # a third series hue, for three groups
GREY = "#2e2a26"
GREY_DIM = "#5a544d"
GREY_FAINT = "#cfc7bb"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": GREY,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "axes.edgecolor": GREY_FAINT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.fontsize": 14,
        "lines.linewidth": 2.2,
    }
)

fitter = DistributionFitter(DistributionRegistry(), seed=42)
rng = np.random.default_rng(42)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", name)


def by_name(results):
    return {f.distribution: f for f in results if isinstance(f, FitResult)}


def frozen(result):
    """The fitted scipy distribution; a failed fit stops the script."""
    if not isinstance(result, FitResult):
        raise RuntimeError(f"fit failed: {result}")
    return result.frozen_dist


def dollars(ax, axis="x"):
    fmt = FuncFormatter(lambda v, _: f"−${-v:g}" if v < 0 else f"${v:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def shape_only(ax):
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------- what makes a distribution
# a die: every outcome gets a probability, and the probabilities add up to 1
fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.0), gridspec_kw={"wspace": 0.38})
ax = axes[0]
faces = np.arange(1, 7)
ax.bar(faces, np.full(6, 1 / 6), width=0.6, color=ACCENT)
for f in faces:
    ax.text(f, 1 / 6 + 0.006, "1/6", ha="center", color=ACCENT_LIGHT, fontsize=15)
ax.set_xticks(faces)
ax.set_ylim(0, 0.25)
ax.set_yticks([0, 0.1, 0.2])
ax.set_xlabel("outcome: every face, also the ones you never rolled")
ax.set_ylabel("probability")
ax.set_title("discrete: bar heights are probabilities  ·  6 × 1/6 = 1")
# a measurement: probability is the area under the curve
adelie_flipper = (
    load_showcase("penguins")
    .query("species == 'Adelie'")["flipper_length_mm"]
    .dropna()
    .to_numpy()
    .astype(float)
)
adelie_norm = stats.norm(adelie_flipper.mean(), adelie_flipper.std(ddof=1))
ax = axes[1]
x = np.linspace(165, 215, 400)
ax.plot(x, adelie_norm.pdf(x), color=ACCENT, lw=3)
band = (x >= 190) & (x <= 200)
ax.fill_between(x[band], adelie_norm.pdf(x[band]), color=ACCENT, alpha=0.35, lw=0)
area = adelie_norm.cdf(200) - adelie_norm.cdf(190)
ax.annotate(
    f"area = P(190–200 mm) = {area:.2f}",
    (197, adelie_norm.pdf(197) * 0.5),
    xytext=(40, 40),
    textcoords="offset points",
    color=ACCENT_LIGHT,
    fontsize=15,
    arrowprops={"arrowstyle": "-", "color": GREY_DIM, "lw": 1.2},
)
ax.set_ylim(0, adelie_norm.pdf(adelie_norm.mean()) * 1.3)
ax.set_xlabel("Adelie flipper length, mm")
ax.set_ylabel("density")
ax.set_title("continuous: area is probability  ·  total area = 1")
save(fig, "rules")
print(f"Adelie flipper: P(190-200) = {area:.3f}")

# a measurement has no probability at a single value, only over a range
room = stats.norm(21.0, 0.3)  # an example: a room's temperature over a day
fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.0), gridspec_kw={"wspace": 0.3})
x = np.linspace(19.9, 22.1, 600)
peak = room.pdf(21.0)
for ax in axes:
    ax.plot(x, room.pdf(x), color=ACCENT, lw=3)
    ax.set_xlim(19.9, 22.1)
    ax.set_ylim(0, peak * 1.3)
    ax.set_xlabel("room temperature, °C")
ax = axes[0]
ax.plot([21.0, 21.0], [0, peak], color=SECOND, lw=2.4)
ax.annotate(
    f"height {peak:.2f}: a density,\nnot a probability",
    (21.0, peak),
    xytext=(30, -10),
    textcoords="offset points",
    color=TEXT,
    fontsize=15,
)
ax.annotate(
    "exactly 21.0 °C: a line,\nno area, probability 0",
    (21.0, peak * 0.35),
    xytext=(30, 0),
    textcoords="offset points",
    color=TEXT,
    fontsize=15,
    arrowprops={"arrowstyle": "-", "color": GREY_DIM, "lw": 1.2},
)
ax.set_ylabel("density")
ax.set_title("a single value")
ax = axes[1]
wide_band = (x >= 20.5) & (x <= 21.5)
ax.fill_between(x[wide_band], room.pdf(x[wide_band]), color=ACCENT, alpha=0.3, lw=0)
sliver = np.linspace(20.95, 21.05, 50)
ax.fill_between(sliver, room.pdf(sliver), color=SECOND, alpha=0.9, lw=0)
p_range = room.cdf(21.5) - room.cdf(20.5)
p_reading = room.cdf(21.05) - room.cdf(20.95)
ax.annotate(
    f"20.5–21.5 °C: {p_range:.2f}",
    (20.62, room.pdf(20.62) * 0.45),
    xytext=(-150, 60),
    textcoords="offset points",
    color=ACCENT_LIGHT,
    fontsize=15,
    arrowprops={"arrowstyle": "-", "color": GREY_DIM, "lw": 1.2},
)
ax.annotate(
    f"the thermometer’s “21.0”\n= 20.95–21.05 °C: {p_reading:.2f}",
    (21.03, peak * 0.6),
    xytext=(45, 30),
    textcoords="offset points",
    color=TEXT,
    fontsize=15,
    arrowprops={"arrowstyle": "-", "color": GREY_DIM, "lw": 1.2},
)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_title("a range")
save(fig, "range")
print(
    f"room: peak density {peak:.2f}, P(20.5-21.5) {p_range:.3f}, P(20.95-21.05) {p_reading:.3f}"
)

# the same ten rolls, drawn as a count and as a distribution
rolls = np.random.default_rng(296).integers(1, 7, 10)
counts = np.bincount(rolls, minlength=7)[1:]
seen = faces[counts > 0]
fig, axes = plt.subplots(1, 2, figsize=(16, 5.0))
ax = axes[0]
ax.bar([str(f) for f in seen], counts[counts > 0], width=0.6, color=GREY_FAINT)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("outcome: only the faces that came up")
ax.set_ylabel("count")
ax.set_title("ten rolls, as a count")
ax = axes[1]
ax.bar(
    faces,
    counts / counts.sum(),
    width=0.6,
    color=GREY_FAINT,
    label="share of your rolls",
)
ax.plot(
    [0.5, 6.5],
    [1 / 6, 1 / 6],
    color=ACCENT,
    lw=3,
    ls="--",
    label="a fair die: 1/6 each",
)
ax.set_xticks(faces)
ax.set_xlim(0.4, 6.6)
ax.set_ylim(0, 0.42)
ax.set_xlabel("outcome: every face a die can show")
ax.set_ylabel("probability")
ax.set_title("the same rolls, as a distribution")
ax.legend(loc="upper right")
save(fig, "dice")
print(f"dice rolls {rolls.tolist()} counts {counts.tolist()}")

# ---------------------------------------------------------------- six families
families = [
    (
        "normal",
        stats.norm(0, 1),
        "many small parts, added",
        "height, measurement error",
        (-4, 4),
    ),
    (
        "lognormal",
        stats.lognorm(s=0.8, scale=np.exp(1)),
        "factors that multiply",
        "how long a project takes",
        (0, 16),
    ),
    (
        "exponential",
        stats.expon(scale=1),
        "waiting, at a steady rate",
        "time to the next message",
        (0, 6),
    ),
    (
        "poisson",
        stats.poisson(mu=4),
        "counting, at a steady rate",
        "leaves falling per minute",
        (-0.5, 12.5),
    ),
    (
        "weibull",
        stats.weibull_min(c=1.5),
        "time until something stops",
        "how long a member stays",
        (0, 3.5),
    ),
    ("pareto", stats.pareto(b=2.5), "the rich get richer", "word frequency", (1, 6)),
]
fig, axes = plt.subplots(2, 3, figsize=(16, 6.8), gridspec_kw={"hspace": 0.75})
for ax, (name, dist, mechanism, example, xlim) in zip(axes.flat, families):
    sample = dist.rvs(4000, random_state=rng)
    if name == "poisson":
        values, counts = np.unique(sample, return_counts=True)
        ax.bar(values, counts / counts.sum(), width=0.8, color=GREY_FAINT)
        k = np.arange(0, 13)
        ax.vlines(k, 0, dist.pmf(k), color=ACCENT, lw=2.4)
        ax.plot(k, dist.pmf(k), "o", color=ACCENT, ms=6)
    else:
        # density relative to the whole sample, so the clipped tail does not inflate the bars
        lo, hi = xlim
        shown = sample[(sample >= lo) & (sample <= hi)]
        width = (hi - lo) / 40
        ax.hist(
            shown,
            bins=40,
            range=xlim,
            weights=np.full(len(shown), 1 / (len(sample) * width)),
            color=GREY_FAINT,
            edgecolor=BG,
            lw=0.4,
        )
        x = np.linspace(lo if lo != 0 else 1e-3, hi, 400)
        ax.plot(x, dist.pdf(x), color=ACCENT, lw=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.set_title(name, fontsize=19, pad=40)
    ax.text(0, 1.06, mechanism, transform=ax.transAxes, color=ACCENT_LIGHT, fontsize=16)
    ax.text(
        0.98,
        0.9,
        f"e.g. {example}",
        transform=ax.transAxes,
        ha="right",
        color=GREY_DIM,
        fontsize=14,
    )
    shape_only(ax)
fig.tight_layout()
# a rule between the rows, so a lower title never reads as a caption of the panel above
top_row = axes[0, 0].get_position().y0
bottom_title = axes[1, 0].get_position().y1
fig.add_artist(
    plt.Line2D(
        [0.02, 0.98],
        [(top_row + bottom_title) / 2 + 0.035] * 2,
        transform=fig.transFigure,
        color=GREY_FAINT,
        lw=1.2,
    )
)
fig.savefig(OUT / "families.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print("wrote families")

# ---------------------------------------------------------------- sums and products
ns = [1, 2, 5, 30]
fig, axes = plt.subplots(2, 4, figsize=(16, 5.6))
for j, n in enumerate(ns):
    for i, (draws, combine) in enumerate(
        [
            (rng.uniform(-1, 1, size=(20000, n)), np.sum),
            (rng.uniform(1, 2, size=(20000, n)), np.prod),
        ]
    ):
        values = combine(draws, axis=1)
        ax = axes[i, j]
        hi = np.quantile(values, 0.995)
        ax.hist(
            values[values <= hi],
            bins=50,
            density=True,
            color=GREY_FAINT,
            edgecolor=BG,
            lw=0.4,
        )
        x = np.linspace(values.min(), hi, 400)
        ax.plot(
            x,
            stats.norm(values.mean(), values.std()).pdf(x),
            color=ACCENT,
            lw=2.6,
            ls="--",
        )
        ax.set_xticks([])
        shape_only(ax)
        if i == 0:
            ax.set_title(f"{n} draw{'s' if n > 1 else ''}", fontsize=17)
axes[0, 0].set_ylabel("added", fontsize=17, color=TEXT)
axes[1, 0].set_ylabel("multiplied", fontsize=17, color=TEXT)
save(fig, "sums_products")

# ---------------------------------------------------------------- project planning: real tasks
# The SiP dataset: estimated and actual hours of ~10,000 tasks at one software company
# (Jones & Cullum 2019, github.com/Derek-Jones/SiP_dataset, CC BY-NC-ND 4.0). It is
# downloaded once to a cache and not kept in the repo.
SIP_URL = (
    "https://raw.githubusercontent.com/Derek-Jones/SiP_dataset/master/Sip-task-info.csv"
)
sip_path = Path.home() / ".cache" / "mads-dav" / "Sip-task-info.csv"
if not sip_path.exists():
    sip_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(SIP_URL, sip_path)  # noqa: S310 -- a fixed https URL
sip = pd.read_csv(sip_path, encoding_errors="replace")
# one row per developer on a task: keep each task once, and drop cancelled work
tasks = sip.drop_duplicates("TaskNumber")
tasks = tasks[
    (tasks.StatusCode != "CANCELLED")
    & (tasks.HoursEstimate > 0)
    & (tasks.HoursActual > 0)
]
blowup = (tasks.HoursActual / tasks.HoursEstimate).to_numpy()
n_tasks = len(blowup)
share = np.full(n_tasks, 1 / n_tasks)
fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.0), gridspec_kw={"wspace": 0.25})
ax = axes[0]
edges = np.arange(-0.125, 8.0, 0.25)
edges[0] = 0
ax.hist(
    blowup, bins=edges.tolist(), weights=share, color=GREY_FAINT, edgecolor=BG, lw=0.4
)
top = 0.16
exact = (blowup == 1).mean()
ax.set_ylim(0, top)
ax.annotate(
    f"↑ {exact:.0%} booked exactly the estimate",
    (1.0, top * 0.97),
    xytext=(16, -2),
    textcoords="offset points",
    color=GREY,
    fontsize=14,
    va="top",
)
ax.axvline(np.median(blowup), color=ACCENT, lw=3, label="median: 1×, on time")
ax.axvline(blowup.mean(), color=SECOND, lw=3, label=f"mean: {blowup.mean():.2f}×")
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
ax.set_xlim(0, 8)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
ax.set_xlabel("actual ÷ estimated hours")
ax.set_ylabel("share of tasks")
ax.set_title("a long tail of late tasks")
ax = axes[1]
octaves = np.arange(-6, 6.01, 1 / 3) - 1 / 6  # one bin centred on exactly 1×
ax.hist(
    blowup,
    bins=(2.0**octaves).tolist(),
    weights=share,
    color=ACCENT,
    edgecolor=BG,
    lw=0.4,
)
ax.set_xscale("log")
ticks = [1 / 16, 1 / 4, 1, 4, 16, 64]
ax.set_xticks(ticks)
ax.set_xticklabels(["1/16×", "1/4×", "1×", "4×", "16×", "64×"])
ax.minorticks_off()
ax.grid(axis="x", color=GREY_FAINT, lw=0.8, alpha=0.7)
ax.set_axisbelow(True)
ax.set_xlim(2.0**-6, 2.0**6.5)
log_top = 0.1
ax.set_ylim(0, log_top)
ax.annotate(
    f"↑ {exact:.0%} at exactly 1×",
    (1.0, log_top * 0.97),
    xytext=(16, -2),
    textcoords="offset points",
    color=GREY,
    fontsize=14,
    va="top",
)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("actual ÷ estimated, log scale: every step is ×4")
ax.set_title("the same tasks on a log axis: a bell around 1×")
save(fig, "sip_tasks")
bell = stats.norm(blowup.mean(), blowup.std())
print(
    f"\nSiP: {n_tasks} tasks; median {np.median(blowup):.2f}, mean {blowup.mean():.2f}, "
    f"exactly 1x {exact:.1%}, late {(blowup > 1).mean():.1%}, early {(blowup < 1).mean():.1%}\n"
    f"  >2x {(blowup > 2).mean():.1%}, >4x {(blowup > 4).mean():.1%}, <0.5x {(blowup < 0.5).mean():.1%}, "
    f"99th pct {np.quantile(blowup, 0.99):.1f}x, max {blowup.max():.0f}x; "
    f"log sd {np.log(blowup).std():.2f}; a bell with this mean and sd puts {bell.cdf(0):.0%} below 0"
)

# ---------------------------------------------------------------- taxi fares
taxis = load_showcase("taxis")
fare = taxis["fare"].to_numpy().astype(float)
n_rides = len(fare)
fare_fits = fitter.fit(fare, discrete=False)
fare_by = by_name(fare_fits)
norm_fit, lognorm_fit = fare_by["norm"], fare_by["lognorm"]
table = fit_table(fare_fits)
print("\ntaxi fit table")
print(
    table[
        [
            "distribution",
            "log_likelihood",
            "ks_stat",
            "ks_pvalue",
            "best_likelihood",
            "best_ks",
        ]
    ].to_string()
)

med, mean = np.median(fare), fare.mean()
xs = np.sort(fare)

# qq-plots
pp = (np.arange(1, n_rides + 1) - 0.5) / n_rides
fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), sharey=True)
for ax, fit, name, color in [
    (axes[0], norm_fit, "normal", SECOND),
    (axes[1], lognorm_fit, "lognormal", ACCENT),
]:
    theo = fit.frozen_dist.ppf(pp)
    lo, hi = min(theo.min(), 0), max(theo.max(), xs.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], color=GREY_DIM, ls="--", lw=1.6)
    ax.scatter(theo, xs, s=14, color=color, lw=0, alpha=0.8)
    ax.set_xlim(lo, hi)
    ax.set_title(f"fare against the fitted {name}")
    ax.set_xlabel(f"fare the {name} expects at that rank")
    if name == "normal":
        ax.annotate(
            "the $150 rides",
            (theo[-2], xs[-2]),
            xytext=(-18, -30),
            textcoords="offset points",
            ha="right",
            color=GREY,
            fontsize=14,
        )
    dollars(ax)
axes[0].set_ylabel("observed fare")
dollars(axes[0], "y")
save(fig, "taxi_qq")

# the whiskers on fare and on log(fare)
fig, axes = plt.subplots(2, 1, figsize=(9.4, 5.4))
flags = {}
for ax, logged in [(axes[0], False), (axes[1], True)]:
    v = np.log(fare) if logged else fare
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = (v < lo) | (v > hi)
    back = np.exp if logged else (lambda a: a)
    box = {
        "med": back(q2),
        "q1": back(q1),
        "q3": back(q3),
        "whislo": back(v[v >= lo].min()),
        "whishi": back(v[v <= hi].max()),
        "fliers": [],
    }
    ax.bxp(
        [box],
        orientation="horizontal",
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": GREY_FAINT, "edgecolor": GREY, "lw": 1.6},
        medianprops={"color": GREY, "lw": 2.6},
        whiskerprops={"color": GREY, "lw": 1.6},
        capprops={"color": GREY, "lw": 1.6},
    )
    ax.scatter(
        fare[out],
        1 + rng.uniform(-0.2, 0.2, out.sum()),
        s=12,
        color=ACCENT,
        lw=0,
        alpha=0.7,
    )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    flags[logged] = int(out.sum())
    label = "log(fare)" if logged else "fare"
    ax.set_title(
        f"the whisker rule on {label}:  {out.sum()} rides flagged  ({out.mean():.1%})"
    )
    if logged:
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100])
        ax.set_xlim(0.9, 170)
    else:
        ax.set_xlim(0, 155)
    dollars(ax)
save(fig, "taxi_whiskers")

# the most expensive ride: how often is a ride at least this expensive?
worst = xs[-1]
z = (worst - mean) / fare.std()
p_norm = norm_fit.frozen_dist.sf(worst)
p_lognorm = lognorm_fit.frozen_dist.sf(worst)
# outside goad's registry: the log-logistic, a right-skewed family with a heavier tail
loglogistic = stats.fisk(*stats.fisk.fit(fare, floc=0))
p_loglogistic = loglogistic.sf(worst)
fig, ax = plt.subplots(figsize=(9.4, 5.8))
surv = 1 - np.arange(n_rides) / n_rides
ax.step(xs, surv, where="post", color=GREY, lw=2.6, label="the data")
x = np.linspace(1, 160, 800)
ax.plot(
    x,
    loglogistic.sf(x),
    color=THIRD,
    lw=2.6,
    label=f"log-logistic: $150 is 1 in {1 / p_loglogistic:,.0f}",
)
ax.plot([worst], [p_loglogistic], "o", color=THIRD, ms=9)
ax.plot(
    x,
    lognorm_fit.frozen_dist.sf(x),
    color=ACCENT,
    lw=3,
    label=f"lognormal: 1 in {1 / p_lognorm:,.0f}",
)
ax.plot(
    x,
    norm_fit.frozen_dist.sf(x),
    color=SECOND,
    lw=2.6,
    ls="--",
    label=f"normal: 1 in $10^{{{-np.log10(p_norm):.0f}}}$",
)
ax.set_yscale("log")
ax.set_ylim(1e-12, 2)
ax.set_xlim(0, 160)
ticks = [1, 1e-3, 1e-6, 1e-9, 1e-12]
ax.set_yticks(ticks)
ax.set_yticklabels(
    ["every ride", "1 in 1,000", "1 in a million", "1 in a billion", "1 in a trillion"]
)
ax.minorticks_off()
ax.axvline(worst, color=GREY_FAINT, lw=1.4)
ax.plot([worst], [p_lognorm], "o", color=ACCENT, ms=9)
ax.set_xlabel("fare")
ax.set_ylabel("rides at least this expensive")
ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))
dollars(ax)
save(fig, "taxi_tail")

print(f"\ntaxi: n={n_rides} mean {mean:.2f} median {med:.2f} std {fare.std():.2f}")
print(
    f"normal with that mean and sd puts {stats.norm(mean, fare.std()).cdf(0):.1%} of rides below $0"
)
print(f"lognorm params {lognorm_fit.params}")
print(
    f"LL lognorm - norm = {lognorm_fit.log_likelihood - norm_fit.log_likelihood:,.0f}"
)
print(
    f"worst ${worst:.0f}: z {z:.1f}, normal tail {p_norm:.1e}, lognormal tail {p_lognorm:.1e} (1 in {1 / p_lognorm:,.0f}), "
    f"log-logistic 1 in {1 / p_loglogistic:,.0f}"
)
print(
    f"rides >= $100: data {int((fare >= 100).sum())}, lognormal expects {n_rides * lognorm_fit.frozen_dist.sf(100):.1f}, "
    f"log-logistic {n_rides * loglogistic.sf(100):.1f}; LL log-logistic {loglogistic.logpdf(fare).sum():,.0f} "
    f"vs lognormal {lognorm_fit.log_likelihood:,.0f}"
)
print(
    f"whiskers flag {flags[False]} on fare, {flags[True]} on log fare; a normal expects 0.70%"
)

# ---------------------------------------------------------------- penguins
penguins = (
    load_showcase("penguins")
    .dropna(subset=["flipper_length_mm", "sex"])
    .reset_index(drop=True)
)
flipper = penguins["flipper_length_mm"].to_numpy().astype(float)
n_birds = len(flipper)
flipper_fits = fitter.fit(flipper, discrete=False)
flipper_by = by_name(flipper_fits)
print("\npenguin pooled fit table")
print(
    fit_table(flipper_fits)[
        ["distribution", "log_likelihood", "ks_pvalue", "best_likelihood", "best_ks"]
    ].to_string()
)
pooled = next(f for f in flipper_fits if isinstance(f, FitResult) and f.best_likelihood)
species_colors = {"Adelie": SECOND, "Chinstrap": THIRD, "Gentoo": ACCENT}

bins = np.arange(
    169.5, 236, 3
)  # flippers are measured to the mm: three whole mm per bin
x = np.linspace(168, 236, 500)
fig, axes = plt.subplots(1, 2, figsize=(16, 5.0), sharex=True, sharey=True)
for ax in axes:
    ax.hist(flipper, bins=bins, density=True, color=GREY_FAINT, edgecolor=BG, lw=0.4)
ax = axes[0]
ax.plot(
    x,
    pooled.frozen_dist.pdf(x),
    color=GREY,
    lw=3,
    label=f"{pooled.distribution} · best by likelihood",
)
ax.plot(
    x,
    flipper_by["norm"].frozen_dist.pdf(x),
    color=GREY_DIM,
    lw=2.2,
    ls="--",
    label="normal",
)
ax.set_title(f"all {n_birds} birds, one family")
ax.legend(loc="upper right")
ax.set_ylabel("density")
ax = axes[1]
mixture = np.zeros_like(x)
for name, group in penguins.groupby("species"):
    values = group["flipper_length_mm"].to_numpy().astype(float)
    share = len(values) / n_birds
    curve = share * frozen(fitter.fit_distribution("norm", values)).pdf(x)
    mixture += curve
    ax.plot(x, curve, color=species_colors[str(name)], lw=3, label=f"{name} · normal")
ax.plot(x, mixture, color=GREY, lw=2, ls=":", label="the three added up")
ax.set_title("the same birds, one normal per species")
ax.legend(loc="upper right")
for ax in axes:
    ax.set_xlabel("flipper length, mm")
save(fig, "penguins_split")


def qq_points(values, dist):
    """Sorted data against the quantiles the fitted family expects at the same ranks."""
    s = np.sort(values)
    ranks = (np.arange(1, len(s) + 1) - 0.5) / len(s)
    return dist.ppf(ranks), s


fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), sharex=True, sharey=True)
line = [168, 236]
ax = axes[0]
ax.plot(line, line, color=GREY_DIM, ls="--", lw=1.6)
theo, obs = qq_points(flipper, frozen(flipper_by["norm"]))
ax.scatter(theo, obs, s=20, color=GREY, lw=0, alpha=0.85)
ax.set_title("all birds against one normal")
ax.set_ylabel("observed flipper length, mm")
ax = axes[1]
ax.plot(line, line, color=GREY_DIM, ls="--", lw=1.6)
for name, group in penguins.groupby("species"):
    values = group["flipper_length_mm"].to_numpy().astype(float)
    theo, obs = qq_points(values, frozen(fitter.fit_distribution("norm", values)))
    ax.scatter(
        theo,
        obs,
        s=20,
        color=species_colors[str(name)],
        lw=0,
        alpha=0.9,
        label=str(name),
    )
ax.set_title("each species against its own normal")
ax.legend(loc="upper left")
for ax in axes:
    ax.set_xlim(*line)
    ax.set_ylim(*line)
    ax.set_xlabel("length the normal expects at that rank, mm")
save(fig, "penguins_qq")

for name, group in penguins.groupby("species"):
    values = group["flipper_length_mm"].to_numpy().astype(float)
    t = fit_table(fitter.fit(values, discrete=False))
    normal = t.loc[t.distribution == "norm"].iloc[0]
    print(
        f"{name}: n={len(values)} mean {values.mean():.1f} sd {values.std(ddof=1):.1f}  "
        f"norm KS p {normal.ks_pvalue:.2f}, {normal.log_likelihood - t.log_likelihood.max():+.1f} from winner {t.iloc[0].distribution}"
    )

by_group = {
    key: frozen(
        fitter.fit_distribution("norm", g["flipper_length_mm"].to_numpy().astype(float))
    )
    for key, g in penguins.groupby(["species", "sex"])
}


def two_sided(dist, value):
    p = dist.cdf(value)
    return 2 * min(p, 1 - p)


adelie_f = (
    penguins.query("species == 'Adelie' and sex == 'Female'")["flipper_length_mm"]
    .to_numpy()
    .astype(float)
)
bird = 202.0
tail_pooled = two_sided(pooled.frozen_dist, bird)
tail_own = two_sided(by_group[("Adelie", "Female")], bird)
own = by_group[("Adelie", "Female")]
fig, ax = plt.subplots(figsize=(9.4, 5.4))
bins = np.arange(170.0, 234.0, 2.0).tolist()
ax.hist(
    flipper,
    bins=bins,
    color=GREY_FAINT,
    edgecolor=BG,
    lw=0.4,
    label=f"all {n_birds} birds",
)
ax.hist(
    adelie_f,
    bins=bins,
    color=ACCENT,
    edgecolor=BG,
    lw=0.4,
    label=f"the {len(adelie_f)} Adelie females",
)
ax.axvline(bird, color=SECOND, lw=3)
ax.annotate(
    f"one Adelie female, {bird:.0f} mm",
    (bird, ax.get_ylim()[1] * 0.96),
    xytext=(10, -2),
    textcoords="offset points",
    color=SECOND,
    fontsize=15,
    va="top",
)
ax.set_xlabel("flipper length, mm")
ax.set_ylabel("birds")
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
save(fig, "penguin_adelie")

longest = penguins.loc[penguins["flipper_length_mm"].idxmax()]
lv = float(longest["flipper_length_mm"])
print(
    f"\nAdelie female {bird:.0f} mm: pooled two-sided tail {tail_pooled:.3f}, own group {tail_own:.4f} "
    f"(1 in {1 / tail_own:.0f}), z within group {(bird - own.mean()) / own.std():.1f}"
)
print(
    f"longest {lv:.0f} mm {longest['sex']} {longest['species']}: pooled one-sided "
    f"{pooled.frozen_dist.sf(lv):.4f} (1 in {1 / pooled.frozen_dist.sf(lv):.0f}), own "
    f"{by_group[(longest['species'], longest['sex'])].sf(lv):.3f} (1 in {1 / by_group[(longest['species'], longest['sex'])].sf(lv):.0f}); "
    f"pooled support ends at {pooled.frozen_dist.support()[1]:.1f}"
)
for key, dist in by_group.items():
    print(f"  {key}: mean {dist.mean():.1f} sd {dist.std():.1f}")

# ---------------------------------------------------------------- irc: before and after
hourly = load_showcase("ubuntu_irc_release_hourly")


def messages_per_day(channel):
    return (
        Pipeline()
        .add(Filter, expr=f"channel == '{channel}'")
        .add(GroupAgg, by=["date", "is_release"], column="messages", agg="sum")
        .apply(hourly)
    )


days = {ch: messages_per_day(ch) for ch in ["#ubuntu-it", "#ubuntu"]}
ordinary = {
    ch: d.loc[~d.is_release, "messages"].to_numpy().astype(float)
    for ch, d in days.items()
}
release = {
    ch: d.loc[d.is_release, "messages"].to_numpy().astype(float)
    for ch, d in days.items()
}
fits = {ch: by_name(fitter.fit(ordinary[ch], discrete=True)) for ch in days}
it_ord, it_rel = ordinary["#ubuntu-it"], release["#ubuntu-it"]
it_fits = fits["#ubuntu-it"]
poisson, nbinom = it_fits["poisson"].frozen_dist, it_fits["nbinom"].frozen_dist

bw = 50.0
edges = np.arange(0.0, 1850.0 + bw, bw)
centers = (edges[:-1] + edges[1:]) / 2
n_it = len(it_ord)
fig, ax = plt.subplots(figsize=(9.4, 5.6))
ax.hist(
    it_ord,
    bins=edges.tolist(),
    color=GREY_FAINT,
    edgecolor=BG,
    lw=0.4,
    label=f"{n_it} ordinary Thursdays",
)
expected_nb = n_it * (nbinom.cdf(edges[1:] - 1) - nbinom.cdf(edges[:-1] - 1))
expected_po = n_it * (poisson.cdf(edges[1:] - 1) - poisson.cdf(edges[:-1] - 1))
visible = expected_po > 0.05
ax.plot(
    centers[visible],
    expected_po[visible],
    color=SECOND,
    lw=2.8,
    label="Poisson: one steady rate",
)
ax.plot(
    centers,
    expected_nb,
    color=ACCENT,
    lw=3.4,
    label="negative binomial: a rate that wanders",
)
top = 34
ax.set_ylim(0, top)
peak = centers[np.argmax(expected_po)]
ax.annotate(
    f"↑ Poisson: {expected_po.max():.0f} of {n_it} days in this one bar",
    (peak, top * 0.97),
    xytext=(14, -2),
    textcoords="offset points",
    color=SECOND,
    fontsize=14,
    va="top",
)
ax.set_xlabel("messages per day")
ax.set_ylabel("Thursdays")
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
save(fig, "irc_poisson")


def draw_null(ax, result, unit, title):
    """The shuffled cloud as a histogram, the real value as a line: notebook 04.4's NullPlot."""
    ax.hist(result.null, bins=40, color=GREY_FAINT, edgecolor=BG, lw=0.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.axvline(result.observed, color=SECOND, lw=3)
    p = result.p_value()
    p_text = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax.annotate(
        f"the real difference: {result.observed:+.0f} {unit}\n{p_text}",
        (result.observed, ax.get_ylim()[1] * 0.97),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        color=SECOND,
        fontsize=15,
    )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(title)


# is a release day busier than an ordinary Thursday? shuffle the label, as notebook 04.4 does
null = NullDistribution(
    lambda d: d.groupby("is_release").messages.mean().diff().iloc[-1],
    n_iter=2000,
    seed=4,
)
irc_null = {ch: null.run(d, label="is_release") for ch, d in days.items()}
fig, axes = plt.subplots(1, 2, figsize=(16, 5.0))
print()
for ax, ch in zip(axes, ["#ubuntu-it", "#ubuntu"]):
    lift = release[ch].mean() / ordinary[ch].mean()
    draw_null(ax, irc_null[ch], "msg/day", f"{ch}: release days ×{lift:.2f}")
    ax.set_xlabel("release − ordinary, messages per day")
    cv = ordinary[ch].std() / ordinary[ch].mean()
    print(
        f"{ch}: ordinary mean {ordinary[ch].mean():.0f} sd/mean {cv:.2f}; release mean {release[ch].mean():.0f} "
        f"n={len(release[ch])}; lift x{lift:.2f}; difference {irc_null[ch].observed:+.0f}; "
        f"two-sided p {irc_null[ch].p_value():.4f}, greater {irc_null[ch].p_value('greater'):.4f}; "
        f"null 2.5-97.5% {np.percentile(irc_null[ch].null, [2.5, 97.5]).round(0)}"
    )
save(fig, "irc_null")

# ---------------------------------------------------------------- texting: did something change?
# Daily text messages of one person over 74 days, from Cameron Davidson-Pilon,
# "Probabilistic Programming and Bayesian Methods for Hackers", chapter 1 (data/txtdata.csv).
texts = np.array(
    [13, 24, 8, 24, 7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11, 19, 29, 6, 19, 12, 22, 12, 18, 72,
     32, 9, 7, 13, 19, 23, 27, 20, 6, 17, 13, 10, 14, 6, 16, 15, 7, 2, 15, 15, 19, 70, 49, 7, 53,
     22, 21, 31, 19, 11, 18, 20, 12, 35, 17, 23, 17, 4, 2, 31, 30, 13, 27, 0, 39, 37, 5, 14, 13, 22]
)  # fmt: skip
SWITCH = 45  # the day the book's switchpoint model puts the change
day = np.arange(len(texts))
after = day >= SWITCH
fig, ax = plt.subplots(figsize=(16, 5.0))
ax.bar(day[~after], texts[~after], width=0.8, color=GREY_FAINT)
ax.bar(day[after], texts[after], width=0.8, color=ACCENT)
for mask, color, label in [(~after, GREY, "before"), (after, ACCENT_LIGHT, "after")]:
    m = texts[mask].mean()
    ax.plot(
        [day[mask].min() - 0.4, day[mask].max() + 0.4],
        [m, m],
        color=color,
        lw=3,
        ls="--",
        label=f"{label} day {SWITCH}: {m:.1f} a day",
    )
ax.legend(loc="upper left")
ax.set_xlim(-1, len(texts))
ax.set_xlabel("day")
ax.set_ylabel("messages")
save(fig, "texting")

texting = pd.DataFrame({"messages": texts, "after": after})
texting_null = NullDistribution(
    lambda d: d.groupby("after").messages.mean().diff().iloc[-1], n_iter=10000, seed=4
).run(texting, label="after")
fig, ax = plt.subplots(figsize=(9.4, 5.6))
draw_null(ax, texting_null, "a day", "10,000 shuffles of before/after")
ax.set_xlabel("after − before, messages per day")
save(fig, "texting_null")
print(
    f"\ntexting: before {texts[~after].mean():.1f} (var {texts[~after].var():.0f}, {(~after).sum()} days), "
    f"after {texts[after].mean():.1f} (var {texts[after].var():.0f}, {after.sum()} days); "
    f"difference {texting_null.observed:+.1f}; two-sided p {texting_null.p_value():.4f}, "
    f"greater {texting_null.p_value('greater'):.4f}; "
    f"null 2.5-97.5% {np.percentile(texting_null.null, [2.5, 97.5]).round(1)}"
)

# Davidson-Pilon's switchpoint model, and the same model with room for day-to-day variation.
# Messages per day follow one rate before day tau and another from tau on; tau is uniform over
# the days and each rate has an exponential prior with mean = the overall mean, as in the book.
# The book samples this with MCMC; with one discrete tau the posterior can be computed exactly.
n_days_txt = len(texts)
alpha = 1 / texts.mean()
taus = np.arange(1, n_days_txt)
cum = np.r_[0, np.cumsum(texts)]
cum_logfact = np.r_[0, np.cumsum(special.gammaln(texts + 1))]


def poisson_evidence(total, days, logfact):
    """log p(counts) for a Poisson rate with an Exp(alpha) prior, the rate integrated out."""
    return (
        np.log(alpha)
        + special.gammaln(total + 1)
        - (total + 1) * np.log(days + alpha)
        - logfact
    )


log_post = poisson_evidence(cum[taus], taus, cum_logfact[taus]) + poisson_evidence(
    cum[-1] - cum[taus], n_days_txt - taus, cum_logfact[-1] - cum_logfact[taus]
)
tau_poisson = np.exp(log_post - log_post.max())
tau_poisson /= tau_poisson.sum()
draw = np.random.default_rng(1)
t_draws = draw.choice(taus, 40000, p=tau_poisson)
rate_before = draw.gamma(cum[t_draws] + 1, 1 / (t_draws + alpha))
rate_after = draw.gamma(cum[-1] - cum[t_draws] + 1, 1 / (n_days_txt - t_draws + alpha))
diff_poisson = rate_after - rate_before

# negative binomial: a mean per period and one shared dispersion r, on grids
mus = np.linspace(1, 60, 400)
rs = np.geomspace(0.5, 100, 60)
log_prior_mu = np.log(alpha) - alpha * mus
parts = []
for r in rs:
    loglik = stats.nbinom.logpmf(texts[None, :], r, (r / (r + mus))[:, None])
    running = np.concatenate(
        [np.zeros((len(mus), 1)), np.cumsum(loglik, axis=1)], axis=1
    )
    before_mu = running[:, taus] + log_prior_mu[:, None]
    after_mu = running[:, [n_days_txt]] - running[:, taus] + log_prior_mu[:, None]
    parts.append((before_mu, after_mu))
log_w = np.array(
    [special.logsumexp(b, axis=0) + special.logsumexp(a, axis=0) for b, a in parts]
)  # (r, tau)
weights = np.exp(log_w - log_w.max())
weights /= weights.sum()
tau_nbinom = weights.sum(axis=0)
picks = draw.choice(weights.size, 40000, p=weights.ravel())
r_idx, t_idx = np.unravel_index(picks, weights.shape)
diff_nbinom = np.empty(len(picks))
for k, (i, j) in enumerate(zip(r_idx, t_idx)):
    b, a = parts[i]
    pb = np.exp(b[:, j] - b[:, j].max())
    pa = np.exp(a[:, j] - a[:, j].max())
    diff_nbinom[k] = draw.choice(mus, p=pa / pa.sum()) - draw.choice(
        mus, p=pb / pb.sum()
    )

fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.0), gridspec_kw={"wspace": 0.25})
ax = axes[0]
ax.bar(
    taus,
    tau_poisson,
    width=0.8,
    color=ACCENT,
    label="the book's model",
)
ax.bar(
    taus,
    tau_nbinom,
    width=0.8,
    color=GREY_DIM,
    alpha=0.9,
    label="days may vary",
)
ax.set_xlim(0, n_days_txt)
ax.set_xlabel("τ, the day the change starts")
ax.set_ylabel("posterior probability")
ax.set_title("τ, the switch day, in two models  ·  prior: uniform")
ax.legend(loc="upper left")
ax = axes[1]
bins_d = np.linspace(-15, 20, 71).tolist()
ax.hist(diff_poisson, bins=bins_d, density=True, color=ACCENT, alpha=0.85)
ax.hist(diff_nbinom, bins=bins_d, density=True, color=GREY_DIM, alpha=0.6)
ax.axvline(0, color=GREY_DIM, ls=":", lw=1.6)
ax.text(
    0.03,
    0.92,
    f"P(λ₂ > λ₁)\nthe book's model: {(diff_poisson > 0).mean():.3f}\n"
    f"days may vary: {(diff_nbinom > 0).mean():.2f}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    color=GREY,
    fontsize=14,
)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("after − before, messages per day")
ax.set_title("λ₂ − λ₁: how much more after the switch?")
save(fig, "texting_switch")

# the book's own figure: posteriors of lambda_1, lambda_2 and tau under the Poisson model
fig, axes = plt.subplots(
    1, 2, figsize=(16.5, 4.8), gridspec_kw={"wspace": 0.25, "width_ratios": [1.4, 1]}
)
ax = axes[0]
rate_bins = np.linspace(14, 28, 57).tolist()
for draws, color, name in [
    (rate_before, ACCENT_LIGHT, "λ₁, before τ"),
    (rate_after, ACCENT, "λ₂, from τ on"),
]:
    ax.hist(
        draws,
        bins=rate_bins,
        density=True,
        color=color,
        alpha=0.9,
        edgecolor=BG,
        lw=0.4,
        label=f"{name}: {draws.mean():.1f} a day",
    )
ax.set_xlim(14, 28)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("messages per day")
ax.set_title("the two rates: hardly any overlap")
ax.legend(loc="upper right")
ax = axes[1]
ax.bar(taus, tau_poisson, width=0.8, color=ACCENT)
ax.set_xlim(0, n_days_txt)
ax.set_xlabel("τ, the day the change starts")
ax.set_ylabel("posterior probability")
ax.set_title("τ: day 44 or 45  ·  prior: uniform")
save(fig, "texting_book")


# picking the switch day from the same data: every split day is another try for chance
split_days = np.arange(10, n_days_txt - 9)  # at least ten days on each side


def split_differences(values):
    """After minus before, in messages per day, for every split day in split_days."""
    running = np.cumsum(values)
    before_mean = running[split_days - 1] / split_days
    after_mean = (running[-1] - running[split_days - 1]) / (n_days_txt - split_days)
    return after_mean - before_mean


observed_curve = split_differences(texts)
observed_45 = observed_curve[split_days == SWITCH][0]
shuffled = np.array([split_differences(draw.permutation(texts)) for _ in range(10000)])
fixed_null = shuffled[:, split_days == SWITCH][:, 0]
search_null = shuffled.max(axis=1)
p_fixed = (fixed_null >= observed_45).mean()
p_search = (search_null >= observed_curve.max()).mean()
fig, axes = plt.subplots(1, 2, figsize=(16.5, 4.8), gridspec_kw={"wspace": 0.25})
ax = axes[0]
low, high = np.percentile(shuffled, [2.5, 97.5], axis=0)
ax.fill_between(
    split_days, low, high, color=GREY_FAINT, lw=0, label="95% of shuffled days"
)
ax.plot(split_days, observed_curve, color=GREY, lw=2.6, label="the real days")
ax.axhline(0, color=GREY_DIM, ls=":", lw=1.4)
ax.plot([SWITCH], [observed_45], "o", color=SECOND, ms=10)
ax.annotate(
    f"day {SWITCH}: +{observed_45:.1f}",
    (SWITCH, observed_45),
    xytext=(10, 8),
    textcoords="offset points",
    color=SECOND,
    fontsize=15,
)
ax.set_xlabel("split day")
ax.set_ylabel("after − before, messages per day")
ax.set_title("every possible split day")
ax.legend(loc="lower left")
ax = axes[1]
bins_s = np.linspace(-10, 16, 66).tolist()
ax.hist(
    search_null,
    bins=bins_s,
    density=True,
    color=GREY_FAINT,
    label="best of all split days",
)
ax.hist(
    fixed_null,
    bins=bins_s,
    density=True,
    color=ACCENT,
    alpha=0.75,
    label=f"day {SWITCH}, fixed in advance",
)
ax.axvline(observed_45, color=SECOND, lw=3)
ax.text(
    0.98,
    0.62,
    f"beat +{observed_45:.1f}:\nfixed day: {p_fixed:.0%}\nbest day: {p_search:.0%}",
    transform=ax.transAxes,
    ha="right",
    color=GREY,
    fontsize=14,
)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("difference chance produces, messages per day")
ax.set_title("10,000 shuffles")
ax.legend(loc="upper right")
save(fig, "texting_search")
print(
    f"search: observed +{observed_45:.2f} at day {SWITCH}, largest split difference "
    f"{observed_curve.max():.2f} at day {split_days[np.argmax(observed_curve)]}; "
    f"one-sided p fixed day {p_fixed:.3f}, searched {p_search:.3f}"
)

top_tau = taus[np.argsort(tau_poisson)[-2:]]
print(
    f"switchpoint Poisson: tau {sorted(top_tau.tolist())} carry {np.sort(tau_poisson)[-2:].sum():.2f}; "
    f"rates {rate_before.mean():.1f} -> {rate_after.mean():.1f}; P(after>before) {(diff_poisson > 0).mean():.4f}\n"
    f"switchpoint negative binomial: max tau probability {tau_nbinom.max():.3f}; "
    f"r {weights.sum(axis=1) @ rs:.2f}; P(after>before) {(diff_nbinom > 0).mean():.3f}; "
    f"Poisson sd at 18 = {np.sqrt(18):.1f}"
)

print(
    f"#ubuntu-it poisson {it_fits['poisson'].params} sd {poisson.std():.0f}; nbinom sd {nbinom.std():.0f}"
)
print(
    f"#ubuntu-it ordinary: mean {it_ord.mean():.0f} var {it_ord.var():.0f} min {it_ord.min():.0f} max {it_ord.max():.0f}"
)
rel_fits = by_name(fitter.fit(it_rel, discrete=True))
print(
    f"#ubuntu-it release nbinom {rel_fits['nbinom'].params} mean {rel_fits['nbinom'].frozen_dist.mean():.0f} "
    f"sd {rel_fits['nbinom'].frozen_dist.std():.0f}"
)

# ---------------------------------------------------------------- covid: fitting the residual
data = CovidDataProcessor(FileConfig(), DataConfig()).process()
VACCINATION_START = "2021-01-06"
before = data.index < VACCINATION_START
tests = data["positivetests"].to_numpy()
deaths = data["deaths_shifted"].to_numpy()
params_before = train_model(
    tests[before],
    deaths[before],
    linear_model,
    mse,
    [0.01, 1.0],
    bounds=[(0, 1.0), (0, None)],
)
params_full = train_model(
    tests, deaths, linear_model, mse, [0.01, 1.0], bounds=[(0, 1.0), (0, None)]
)
covid_fitter = DistributionFitter(seed=42)
vax = np.datetime64(VACCINATION_START)


def residual_figure(mask, params, name, fits_to_draw, title, legend_loc, label_y):
    dates = data.index[mask]
    predicted = linear_model(tests[mask], params)
    residual = deaths[mask] - predicted
    results = covid_fitter.fit(residual, discrete=False)
    fitted = by_name(results)
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.9), gridspec_kw={"width_ratios": [1.25, 1.25, 1]}
    )
    ax = axes[0]
    ax.plot(dates, deaths[mask], color=GREY, lw=2.2, label="deaths, 14 days later")
    ax.plot(dates, predicted, color=ACCENT, lw=3, label="the line")
    ax.set_ylim(0, 150)
    ax.set_title("deaths against a line in positive tests")
    ax.legend(loc=legend_loc)
    ax = axes[1]
    ax.axhline(0, color=GREY_DIM, ls=":", lw=1.5)
    ax.fill_between(
        dates, residual, 0, where=residual >= 0, color=ACCENT, alpha=0.35, lw=0
    )
    ax.fill_between(
        dates, residual, 0, where=residual < 0, color=SECOND, alpha=0.35, lw=0
    )
    ax.plot(dates, residual, color=GREY, lw=1.8)
    ax.set_ylim(-60, 45)
    ax.set_title("the residual: what the line missed")
    for a in axes[:2]:
        a.set_xlim(dates.min(), dates.max() + np.timedelta64(3, "D"))
        a.axvline(vax, color=SECOND, lw=1.6)
        a.xaxis.set_major_locator(
            mdates.MonthLocator(
                bymonth=[10, 11, 12, 1] if mask.sum() < 100 else [10, 12, 2, 4, 6]
            )
        )
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].annotate(
        "vaccination starts",
        (vax, label_y),
        xytext=(-8, 0),
        textcoords="offset points",
        color=SECOND,
        fontsize=14,
        va="center",
        ha="right",
    )
    ax = axes[2]
    ax.hist(residual, bins=16, density=True, color=GREY_FAINT, edgecolor=BG, lw=0.4)
    xr = np.linspace(residual.min() - 5, residual.max() + 5, 400)
    readable = {"norm": "normal", "skewnorm": "skew-normal"}
    for fam, color, ls in fits_to_draw:
        ax.plot(
            xr,
            fitted[fam].frozen_dist.pdf(xr),
            color=color,
            lw=2.6,
            ls=ls,
            label=readable.get(fam, fam),
        )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.set_title(title)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", fontsize=13)
    save(fig, name)
    table = fit_table(results)
    print(
        f"\n{name}: line {params[0]:.4f} * tests + {params[1]:.1f}, {mask.sum()} days"
    )
    print(
        table[
            [
                "distribution",
                "log_likelihood",
                "ks_pvalue",
                "best_likelihood",
                "best_ks",
            ]
        ].to_string()
    )
    return residual


residual_figure(
    before,
    params_before,
    "covid_before",
    [("skewnorm", ACCENT, "-"), ("weibull", GREY, "-"), ("norm", SECOND, "--")],
    "its distribution: the families tie",
    "upper left",
    36,
)
full_residual = residual_figure(
    np.ones(len(data), dtype=bool),
    params_full,
    "covid_full",
    [("uniform", ACCENT, "-"), ("norm", SECOND, "--")],
    "its distribution: every family rejected",
    "upper right",
    -50,
)
negative = full_residual < 0
run_start = data.index[np.where(~negative)[0].max() + 1]
print(f"full-window residual negative from {run_start.date()} to the end")
