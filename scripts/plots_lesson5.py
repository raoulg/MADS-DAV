"""Figures for the lesson 5 slides, in the pttrn Paper palette.

Run from the repo root:  uv run python scripts/plots_lesson5.py
Writes PNGs to presentations/img/lesson5/ and prints the numbers the slides quote.
Data: the mpg, flights, penguins_raw and ubuntu_irc showcases, data/sim/correlation.csv,
the Dutch COVID figures that goad's CovidDataProcessor downloads once to
~/.cache/goad/covid/. The duckweed pond is an illustration with made-up growth, not data.
Each computation repeats the code and seeds of its notebook, so the numbers match 05.1–05.5.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor, lasso_path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

from goad_toolkit.analytics import DistributionFitter, FitResult, fit_table
from goad_toolkit.config import DataConfig, FileConfig
from goad_toolkit.dataprocessor import CovidDataProcessor
from goad_toolkit.datatransforms import (
    CountValues,
    DiffValues,
    Filter,
    FlagDates,
    GroupAgg,
    Head,
    Pipeline,
    RegexFeature,
    RollingAvg,
    SelectDataRange,
    TransformBase,
)
from goad_toolkit.models import linear_model, logistic, mse, train_model
from scripts.pipelines import build_irc_pipeline
from wa_analyzer.data import load_showcase

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentations" / "img" / "lesson5"
OUT.mkdir(parents=True, exist_ok=True)

# pttrn tokens, Paper theme (~/code/packages/pttrn-huisstijl/tokens.json): petrol for the main
# series, vermilion only as a line or a dot, warm greys for everything else
BG = "#f4f1ec"
TEXT = "#111111"
ACCENT = "#1d5a73"
ACCENT_LIGHT = "#5e93a8"
SECOND = "#c43a14"
THIRD = "#8a6d3b"
GREY = "#2e2a26"
GREY_DIM = "#5a544d"
GREY_FAINT = "#cfc7bb"
# Ink theme, for the cover
INK_BG = "#141210"
INK_TEXT = "#f4f1ec"
INK_MUTED = "#b8b0a5"
INK_PETROL = "#86b9c9"
INK_ACCENT = "#f2643c"

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


def save(fig, name, facecolor=BG):
    fig.tight_layout()
    fig.savefig(
        OUT / f"{name}.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor=facecolor,
    )
    plt.close(fig)
    print("wrote", name)


def by_name(results):
    return {f.distribution: f for f in results if isinstance(f, FitResult)}


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GREY_FAINT)


def months(ax, which=(10, 12, 2, 4, 6)):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=list(which)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def r_squared(y, predicted):
    return 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------- the four shapes
x = np.linspace(0, 10, 300)
SHAPES = [
    ("linear", "is proportional", 0.8 * x + 1, "y = a·x + b"),
    ("sine", "repeats", np.sin(2 * np.pi * x / 4), "y = A·sin(2πx / P)"),
    ("exponential", "compounds", np.exp(0.45 * x), "y = a·eᵏˣ"),
    ("logistic", "turns", logistic(x, k=1.4, x0=5), "y = L / (1 + e⁻ᵏ⁽ˣ⁻ˣ⁰⁾)"),
]

# the cover: the four shapes as line art on Ink
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), gridspec_kw={"hspace": 0.55})
for ax, (name, _, y, _) in zip(axes.flat, SHAPES):
    ax.set_facecolor(INK_BG)
    color = INK_ACCENT if name == "logistic" else INK_PETROL
    ax.plot(x, y, color=color, lw=4, solid_capstyle="round")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#5a5148")
    ax.set_title(name, color=INK_MUTED, fontsize=17, family="monospace", loc="left")
save(fig, "cover_shapes", facecolor=INK_BG)

# the four shapes, each with the claim it makes
fig, axes = plt.subplots(1, 4, figsize=(17, 4.0), gridspec_kw={"wspace": 0.3})
for ax, (name, claim, y, formula) in zip(axes, SHAPES):
    ax.plot(x, y, color=SECOND if name == "logistic" else ACCENT, lw=3.2)
    bare(ax)
    ax.set_title(f"{name}\n", fontsize=19, loc="left")
    ax.text(0, 1.03, f"it {claim}", transform=ax.transAxes, color=GREY_DIM, fontsize=16)
    ax.set_xlabel(formula, fontsize=16, color=GREY, labelpad=10)
save(fig, "horsemen")


# ---------------------------------------------------------------- duckweed: growth that levels off
# An illustration, not measured data: duckweed that starts on 0.4% of a pond and doubles
# every two days while there is room. The exponential keeps doubling; the logistic has the
# same start and the same doubling, and stops at the edge of the pond.
START, DOUBLING = 0.4, 2.0
rate = np.log(2) / DOUBLING
days = np.linspace(0, 34, 500)


def exponential_cover(t):
    return START * np.exp(rate * t)


def logistic_cover(t):
    return 100 / (1 + (100 / START - 1) * np.exp(-rate * t))


half_day = np.log(100 / START - 1) / rate
twice_day = np.log(200 / START) / rate
fig = plt.figure(figsize=(16, 5.6))
grid = fig.add_gridspec(
    2, 4, width_ratios=[2.6, 0.08, 0.75, 0.75], wspace=0.08, hspace=0.35
)
ax = fig.add_subplot(grid[:, 0])
ax.plot(
    days,
    exponential_cover(days),
    color=SECOND,
    lw=2.6,
    ls="--",
    label="exponential: doubling forever",
)
ax.plot(
    days,
    logistic_cover(days),
    color=ACCENT,
    lw=3.2,
    label="logistic: until the pond is full",
)
ax.axhline(100, color=GREY_DIM, lw=1.2, ls=":")
ax.annotate(
    "the whole pond",
    (34, 100),
    xytext=(0, 6),
    textcoords="offset points",
    ha="right",
    color=GREY_DIM,
    fontsize=14,
)
ax.scatter([twice_day], [200], color=SECOND, s=60, zorder=3, clip_on=False)
ax.annotate(
    f"day {twice_day:.0f}: twice the pond",
    (twice_day, 200),
    xytext=(-12, -2),
    textcoords="offset points",
    color=SECOND,
    fontsize=14,
    va="top",
    ha="right",
)
ax.set_ylim(0, 205)
ax.set_xlim(0, 34)
ax.set_xlabel("days")
ax.set_ylabel("share of the pond covered (%)")
ax.text(
    14.2, 62, "exponential:\ndoubling forever", color=SECOND, fontsize=14, ha="right"
)
ax.text(
    22.5, 62, "logistic:\nuntil the pond is full", color=ACCENT, fontsize=14, ha="left"
)
# four small ponds, filled from two clumps outward, at the share the logistic gives
cells = 60
yy, xx = np.mgrid[0:cells, 0:cells] / (cells - 1)
inside = (xx - 0.5) ** 2 + (yy - 0.5) ** 2 <= 0.25
distance = np.minimum(np.hypot(xx - 0.3, yy - 0.35), np.hypot(xx - 0.72, yy - 0.68))
distance = distance + np.random.default_rng(5).normal(0, 0.04, distance.shape)
order = np.argsort(np.where(inside, distance, np.inf), axis=None)
for k, day in enumerate([8, 14, 18, 24]):
    pond_ax = fig.add_subplot(grid[k // 2, 2 + k % 2])
    covered = logistic_cover(day)
    image = np.full((cells, cells, 3), [0.957, 0.945, 0.925])
    image[inside] = [0.84, 0.88, 0.89]
    filled = np.zeros(cells * cells, dtype=bool)
    filled[order[: int(round(covered / 100 * inside.sum()))]] = True
    image[filled.reshape(cells, cells)] = [0.114, 0.353, 0.451]
    pond_ax.imshow(image, interpolation="nearest")
    pond_ax.set_xticks([])
    pond_ax.set_yticks([])
    for spine in pond_ax.spines.values():
        spine.set_visible(False)
    pond_ax.set_title(f"day {day}: {covered:.0f}%", fontsize=14, loc="center")
save(fig, "pond")
print(
    f"pond (illustration): half covered on day {half_day:.1f}; the exponential reaches 100% on day "
    f"{np.log(100 / START) / rate:.1f} and 200% on day {twice_day:.1f}; logistic at days 8/14/18/24: "
    f"{[round(float(logistic_cover(d)), 1) for d in (8, 14, 18, 24)]}"
)


# ---------------------------------------------------------------- airline: a line plus a sine
flights = load_showcase("flights")
flight_months = pd.to_datetime(
    flights["year"].astype(str) + "-" + flights["month"], format="%Y-%b"
)
tm = np.arange(len(flights)).astype(float)
passengers = flights["passengers"].to_numpy().astype(float)


def trend_plus_seasonal(t, params):
    a, b, amplitude, phase = params
    return a * t + b + amplitude * np.sin(2 * np.pi * t / 12 + phase)


trend_only = train_model(tm, passengers, linear_model, mse, [1.0, 100.0])
trend_and_cycle = train_model(
    tm, passengers, trend_plus_seasonal, mse, [1.0, 100.0, 30.0, 0.0]
)
line_fit = linear_model(tm, trend_only)
cycle_fit = trend_plus_seasonal(tm, trend_and_cycle)
explained = {
    name: 1 - np.var(passengers - fit) / np.var(passengers)
    for name, fit in [("line", line_fit), ("line + sine", cycle_fit)]
}
fig, ax = plt.subplots(figsize=(14, 5.0))
ax.scatter(flight_months, passengers, color=GREY, s=14, zorder=3, label="passengers")
ax.plot(
    flight_months,
    line_fit,
    color=GREY_DIM,
    lw=2.2,
    ls="--",
    label=f"a line: {explained['line']:.0%} explained",
)
ax.plot(
    flight_months,
    cycle_fit,
    color=ACCENT,
    lw=2.6,
    label=f"a line plus a 12-month sine: {explained['line + sine']:.0%}",
)
ax.set_ylabel("passengers per month (×1000)")
ax.legend(loc="upper left")
save(fig, "airline")
print(f"airline: {explained}")
print(f"  line + sine params {np.round(trend_and_cycle, 2)}")


# the answer to the question on the slide: let the height of the wave grow with the trend
def trend_times_seasonal(t, params):
    a, b, amplitude, phase = params
    return (a * t + b) * (1 + amplitude * np.sin(2 * np.pi * t / 12 + phase))


growing = train_model(
    tm, passengers, trend_times_seasonal, mse, [2.0, 90.0, 0.1, -1.45]
)
growing_fit = trend_times_seasonal(tm, growing)
print(
    f"  line x (1 + sine): {1 - np.var(passengers - growing_fit) / np.var(passengers):.1%} explained, "
    f"params {np.round(growing, 3)}"
)


# ---------------------------------------------------------------- a model that fits everything
mpg = load_showcase("mpg").dropna(subset=["weight", "mpg"])
# twelve cars spread evenly over the weights, so no stretch of the axis is empty
by_weight = mpg.sort_values("weight")
cars = by_weight.iloc[np.linspace(0, len(by_weight) - 1, 12).round().astype(int)]
others = mpg.drop(cars.index)
others = others[others.weight.between(cars.weight.min(), cars.weight.max())]


def poly(frame, degree):
    return np.polynomial.Polynomial.fit(frame.weight, frame.mpg, degree)


def line_r2(frame):
    model = poly(frame, 1)
    return r_squared(frame.mpg.to_numpy(), model(frame.weight.to_numpy()))


# the figure shows a typical shuffle: the one whose straight-line R² is the median of 200
shuffles = [
    cars.assign(mpg=np.random.default_rng(seed).permutation(cars.mpg.to_numpy()))
    for seed in range(200)
]
shuffle_r2 = np.array([line_r2(frame) for frame in shuffles])
shuffled = shuffles[int(np.argmin(np.abs(shuffle_r2 - np.median(shuffle_r2))))]
print(
    f"flexible: straight-line R2 over 200 shuffles: median {np.median(shuffle_r2):.2f}, "
    f"mean {shuffle_r2.mean():.2f}, 95th percentile {np.quantile(shuffle_r2, 0.95):.2f}"
)


grid = np.linspace(cars.weight.min() - 60, cars.weight.max() + 60, 800)
ylim = (mpg.mpg.min() - 4, mpg.mpg.max() + 4)
fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), sharex=True, sharey=True)
for row, (frame, label) in enumerate(
    [(cars, "12 real cars"), (shuffled, "the same 12, mpg shuffled: no relation left")]
):
    for col, (degree, name) in enumerate(
        [(1, "a straight line"), (11, "a polynomial of degree 11")]
    ):
        ax = axes[row, col]
        model = poly(frame, degree)
        fit = r_squared(frame.mpg.to_numpy(), model(frame.weight.to_numpy()))
        ax.plot(grid, model(grid), color=ACCENT, lw=2.8)
        ax.scatter(frame.weight, frame.mpg, color=GREY, s=55, zorder=3)
        ax.set_ylim(*ylim)
        ax.set_title(f"{label} · {name}", fontsize=15)
        ax.text(
            0.97,
            0.9,
            f"R² = {fit:.2f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=19,
            color=SECOND if degree == 11 else ACCENT,
            bbox={"facecolor": BG, "edgecolor": "none", "pad": 4},
            zorder=5,
        )
        print(f"flexible: {label}, degree {degree}: R2 {fit:.3f}")
for ax in axes[1]:
    ax.set_xlabel("weight (lbs)")
for ax in axes[:, 0]:
    ax.set_ylabel("miles per gallon")
save(fig, "flexible")

# too simple, about right, too flexible: the 12 cars it saw against the cars it did not see
fig, axes = plt.subplots(1, 3, figsize=(17, 5.0), sharey=True)
for ax, (degree, name) in zip(
    axes,
    [
        (0, "too simple: one number"),
        (1, "about right: a line"),
        (11, "too flexible: degree 11"),
    ],
):
    model = poly(cars, degree)
    seen = np.sqrt(np.mean((cars.mpg - model(cars.weight)) ** 2))
    unseen = np.sqrt(np.mean((others.mpg - model(others.weight)) ** 2))
    ax.scatter(
        others.weight, others.mpg, color=GREY_FAINT, s=12, label="cars it never saw"
    )
    ax.plot(grid, model(grid), color=ACCENT, lw=2.8)
    ax.scatter(
        cars.weight,
        cars.mpg,
        color=GREY,
        s=55,
        zorder=3,
        label="the 12 it was fitted on",
    )
    ax.set_ylim(*ylim)
    ax.set_title(name, fontsize=16)
    ax.text(
        0.03,
        0.05,
        f"off by {seen:.1f} mpg on its 12 cars\noff by {unseen:,.1f} mpg on the others",
        transform=ax.transAxes,
        fontsize=14.5,
        color=SECOND if unseen > 2 * max(seen, 1) else GREY,
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 4},
        zorder=5,
    )
    ax.set_xlabel("weight (lbs)")
    print(
        f"new data: degree {degree}: rmse seen {seen:.2f}, unseen {unseen:.2f} (n={len(others)})"
    )
axes[0].set_ylabel("miles per gallon")
axes[0].legend(loc="upper right", fontsize=13)
save(fig, "new_data")


# ---------------------------------------------------------------- the line you draw is a claim
fig, axes = plt.subplots(1, 3, figsize=(17, 5.0), sharey=True)
w = np.linspace(mpg.weight.min(), mpg.weight.max(), 300)
smooth = lowess(mpg.mpg, mpg.weight, frac=2 / 3)
curves = [
    ("a straight line", np.polyval(np.polyfit(mpg.weight, mpg.mpg, 1), w), w),
    ("a curve (order 2)", np.polyval(np.polyfit(mpg.weight, mpg.mpg, 2), w), w),
    ("lowess: no shape assumed", smooth[:, 1], smooth[:, 0]),
]
for ax, (name, y, xs) in zip(axes, curves):
    ax.scatter(mpg.weight, mpg.mpg, color=GREY_FAINT, s=12)
    ax.plot(xs, y, color=ACCENT, lw=3)
    ax.set_title(name, fontsize=16)
    ax.set_xlabel("weight (lbs)")
axes[0].set_ylabel("miles per gallon")
save(fig, "mpg_claims")
fit = stats.linregress(mpg.weight, mpg.mpg)
log_fit = stats.linregress(np.log(mpg.weight), np.log(mpg.mpg))
print(
    f"mpg: line slope {fit.slope:.5f} r2 {fit.rvalue**2:.3f} p {fit.pvalue:.1e}; "
    f"log-log slope {log_fit.slope:.3f} r2 {log_fit.rvalue**2:.3f}; "
    f"doubling weight multiplies mpg by {2**log_fit.slope:.2f}"
)


# ---------------------------------------------------------------- covid: the reasoning steps
VACCINATION_START = "2021-01-06"
vax = float(mdates.date2num(np.datetime64(VACCINATION_START)))
config = DataConfig()
processor = CovidDataProcessor(FileConfig(), config)
raw = processor.filehandler.load(None, True)
data = processor.process()

# the same steps without the shift: daily deaths and tests, raw and as a 7-day average
daily = (
    Pipeline()
    .add(DiffValues, column="deaths")
    .add(SelectDataRange, start_date=config.start_date, end_date=config.end_date)
    .apply(raw.copy())
)
smoothed = (
    Pipeline()
    .add(RollingAvg, column="deaths", window=config.window, rename=True)
    .add(RollingAvg, column="positivetests", window=config.window, rename=True)
    .apply(daily.copy())
)

# step 0: the deaths as delivered, a running total since the start of the epidemic
total = raw.loc[config.start_date : config.end_date, "deaths"]
fig, ax = plt.subplots(figsize=(16, 6.2))
ax.plot(total.index, total, color=ACCENT, lw=3)
for when, ha, offset in [
    (total.index[0], "left", (6, -30)),
    (total.index[-1], "right", (0, -28)),
]:
    ax.annotate(
        f"{total[when]:,.0f} on {when.day} {when:%B %Y}",
        (when, total[when]),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        color=GREY,
        fontsize=15,
    )
ax.set_ylim(0, total.max() * 1.1)
ax.set_ylabel("deaths so far, since February 2020")
months(ax)
save(fig, "covid_total")
print(
    f"running total: {total.iloc[0]:,.0f} on {total.index[0].date()} -> {total.iloc[-1]:,.0f} on {total.index[-1].date()}"
)

# step 1, before any answer: only the numbers as reported, for "what do you see?"
fig, axes = plt.subplots(2, 1, figsize=(16, 6.2), sharex=True)
for ax, column, label in [
    (axes[0], "positivetests", "positive tests per day, as reported"),
    (axes[1], "deaths", "deaths per day, as reported"),
]:
    ax.plot(daily.index, daily[column], color=GREY, lw=1.6)
    ax.set_title(label, fontsize=16)
    ax.set_ylim(0, None)
months(axes[1])
save(fig, "covid_raw")

# step 1: too much noise
fig, axes = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True)
for ax, column, label in [
    (axes[0], "positivetests", "positive tests per day"),
    (axes[1], "deaths", "deaths per day"),
]:
    ax.plot(daily.index, daily[column], color=GREY_FAINT, lw=1.4, label="as reported")
    ax.plot(
        smoothed.index,
        smoothed[f"{column}_rolling_avg"],
        color=ACCENT,
        lw=2.8,
        label="7-day average",
    )
    ax.set_title(label, fontsize=16)
    ax.set_ylim(0, None)
axes[0].legend(loc="upper right", ncols=2)
months(axes[1])
save(fig, "covid_smooth")
by_weekday = daily.groupby(daily.index.to_series().dt.dayofweek)["deaths"].mean()
print(
    "deaths reported per weekday (Mon..Sun):", by_weekday.round(0).astype(int).tolist()
)

# step 2: two weeks from a positive test to a death
tests7 = smoothed["positivetests_rolling_avg"]
deaths7 = smoothed["deaths_rolling_avg"]
peaks = []
for a, b in [("2020-10-01", "2020-12-01"), ("2020-12-01", "2021-02-15")]:
    peaks.append((tests7.loc[a:b].idxmax(), deaths7.loc[a:b].idxmax()))
pre = smoothed.loc[:"2021-01-05"]
lags = np.arange(0, 29)
lag_corr = [
    pre["positivetests_rolling_avg"].corr(
        smoothed["deaths_rolling_avg"].shift(-lag).loc[:"2021-01-05"]
    )
    for lag in lags
]
fig = plt.figure(figsize=(17, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], hspace=0.12, wspace=0.18)
top = fig.add_subplot(gs[0, 0])
bottom = fig.add_subplot(gs[1, 0], sharex=top)
window = slice("2020-10-07", "2021-02-28")
top.plot(tests7.loc[window].index, tests7.loc[window], color=ACCENT, lw=2.8)
bottom.plot(deaths7.loc[window].index, deaths7.loc[window], color=GREY, lw=2.8)
top.set_title("positive tests per day (7-day average)", fontsize=15)
bottom.set_title("deaths per day (7-day average)", fontsize=15)
for test_peak, death_peak in peaks:
    top.axvline(test_peak, color=GREY_FAINT, lw=1.2)
    bottom.axvline(test_peak, color=GREY_FAINT, lw=1.2)
    bottom.axvline(death_peak, color=SECOND, lw=1.6)
    bottom.annotate(
        f"+{(death_peak - test_peak).days} days",
        (death_peak, deaths7.loc[death_peak] * 0.35),
        xytext=(6, 0),
        textcoords="offset points",
        color=SECOND,
        fontsize=14,
    )
plt.setp(top.get_xticklabels(), visible=False)
bottom.set_ylim(0, 125)
top.set_ylim(0, 13000)
months(bottom, (10, 11, 12, 1, 2))
ax = fig.add_subplot(gs[:, 1])
# one correlation per whole-day shift: discrete measurements, joined by a thin line for the eye
ax.plot(lags, lag_corr, color=ACCENT_LIGHT, lw=1.2, zorder=1)
ax.scatter(lags, lag_corr, color=ACCENT, s=46, zorder=2)
best = lags[int(np.argmax(lag_corr))]
ax.axvline(14, color=SECOND, lw=1.6)
ax.annotate(
    "14 days",
    (14, 0.45),
    xytext=(6, 0),
    textcoords="offset points",
    color=SECOND,
    fontsize=14,
)
ax.set_xlabel("deaths shifted back by … days")
ax.set_ylabel("correlation with tests")
ax.set_title("before vaccination: which shift lines them up?", fontsize=15)
ax.set_ylim(min(lag_corr) - 0.05, 0.92)
ax.set_xticks([0, 7, 14, 21, 28])
save(fig, "covid_lag")
print(
    f"lag: peaks {[(a.date(), b.date(), (b - a).days) for a, b in peaks]}; best corr lag {best} ({max(lag_corr):.3f}), at 14: {lag_corr[14]:.3f}"
)

# step 3: tens against thousands
pre_vax = data.loc[:"2021-01-05"]
fig, axes = plt.subplots(1, 2, figsize=(17, 5.0), gridspec_kw={"wspace": 0.22})
ax = axes[0]
ax.plot(
    pre_vax.index, pre_vax.positivetests, color=ACCENT, lw=2.8, label="positive tests"
)
ax.plot(
    pre_vax.index,
    pre_vax.deaths_shifted,
    color=GREY,
    lw=2.8,
    label="deaths, 14 days later",
)
ax.set_title("on their own scale: the deaths are a flat line", fontsize=15)
ax.legend(loc="upper center")
ax = axes[1]
ax.axhline(0, color=GREY_FAINT, lw=1)
ax.plot(
    pre_vax.index,
    pre_vax.positivetests_zscore,
    color=ACCENT,
    lw=2.8,
    label="positive tests",
)
ax.plot(
    pre_vax.index,
    pre_vax.deaths_shifted_zscore,
    color=GREY,
    lw=2.8,
    label="deaths, 14 days later",
)
ax.set_title("as z-scores: the same shape", fontsize=15)
ax.set_ylabel("z-score")
for ax in axes:
    months(ax, (10, 11, 12, 1))
save(fig, "covid_scale")

# step 4: vaccination started on 6 January 2021
fig, ax = plt.subplots(figsize=(14, 5.0))
ax.axhline(0, color=GREY_FAINT, lw=1)
ax.plot(
    data.index, data.positivetests_zscore, color=ACCENT, lw=2.8, label="positive tests"
)
ax.plot(
    data.index,
    data.deaths_shifted_zscore,
    color=GREY,
    lw=2.8,
    label="deaths, 14 days later",
)
ax.axvline(vax, color=SECOND, lw=1.8)
ax.annotate(
    "6 January 2021: first vaccination",
    (vax, 2.4),
    xytext=(8, 0),
    textcoords="offset points",
    color=SECOND,
    fontsize=15,
)
ax.set_ylabel("z-score")
ax.legend(loc="lower left")
months(ax)
save(fig, "covid_event")

# step 5: the line, and how far off it is allowed to be
tests = data["positivetests"].to_numpy()
deaths = data["deaths_shifted"].to_numpy()
before = data.index < VACCINATION_START
params_before = train_model(
    tests[before],
    deaths[before],
    linear_model,
    mse,
    [0.01, 1.0],
    bounds=[(0, 1.0), (0, None)],
)
line = linear_model(tests, params_before)
sigma = np.std(deaths[before] - line[before], ddof=2)
inside = np.mean(np.abs(deaths[before] - line[before]) <= 2 * sigma)
fig, ax = plt.subplots(figsize=(11, 5.2))
dates_before = data.index[before]
ax.fill_between(
    dates_before,
    line[before] - 2 * sigma,
    line[before] + 2 * sigma,
    color=ACCENT,
    alpha=0.16,
    lw=0,
    label=f"the line ± 2σ   (σ = {sigma:.0f} deaths a day)",
)
ax.plot(dates_before, line[before], color=ACCENT, lw=3, label="a·tests + b")
ax.plot(dates_before, deaths[before], color=GREY, lw=2.4, label="deaths, 14 days later")
ax.set_ylim(0, 150)
ax.set_ylabel("deaths per day")
ax.legend(loc="upper left")
months(ax, (10, 11, 12, 1))
save(fig, "covid_band")
print(
    f"line before vaccination: deaths = {params_before[0]:.4f} * tests + {params_before[1]:.1f}; "
    f"sigma {sigma:.1f}; {inside:.0%} of {before.sum()} days within 2 sigma"
)

# what b adds: the line through zero against the line with an intercept, on the same days
tests_before, deaths_before = tests[before], deaths[before]
a_only = np.sum(tests_before * deaths_before) / np.sum(tests_before**2)
a_fit, b_fit = params_before
reach = np.linspace(0, tests_before.max() * 1.05, 100)
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.scatter(
    tests_before,
    deaths_before,
    color=GREY,
    s=30,
    zorder=2,
    label="one day before vaccination",
)
ax.plot(
    reach,
    a_only * reach,
    color=GREY_DIM,
    lw=2.2,
    ls="--",
    label=f"a · tests  ({a_only:.4f} · tests)",
)
ax.plot(
    reach,
    a_fit * reach + b_fit,
    color=ACCENT,
    lw=3,
    label=f"a · tests + b  ({a_fit:.4f} · tests + {b_fit:.0f})",
)
ax.scatter([0], [b_fit], color=SECOND, s=90, zorder=3, clip_on=False)
ax.annotate(
    f"b = {b_fit:.0f} deaths a day,\neven at zero positive tests",
    (0, b_fit),
    xytext=(16, -6),
    textcoords="offset points",
    color=SECOND,
    fontsize=15,
    va="top",
)
ax.set_xlim(0, reach.max())
ax.set_ylim(0, 125)
ax.set_xlabel("positive tests on a day (7-day average)")
ax.set_ylabel("deaths 14 days later")
ax.legend(loc="upper left")
save(fig, "covid_fit")
print(
    f"fit: through zero a={a_only:.4f} (mse {mse(deaths_before, a_only * tests_before):.0f}); "
    f"with b a={a_fit:.4f} b={b_fit:.1f} (mse {mse(deaths_before, line[before]):.0f}); "
    f"tests before vaccination range {tests_before.min():.0f}-{tests_before.max():.0f}"
)

# the pre-vaccination line over the whole window, and its residual
residual = deaths - line
outside_after = np.mean(deaths[~before] < line[~before] - 2 * sigma)
negative = residual < 0
run_start = data.index[np.where(~negative)[0].max() + 1]
fig, axes = plt.subplots(1, 2, figsize=(17, 5.0), gridspec_kw={"wspace": 0.2})
ax = axes[0]
ax.fill_between(
    data.index, line - 2 * sigma, line + 2 * sigma, color=ACCENT, alpha=0.16, lw=0
)
ax.plot(data.index, line, color=ACCENT, lw=3, label="the line from before vaccination")
ax.plot(data.index, deaths, color=GREY, lw=2.4, label="deaths, 14 days later")
ax.set_ylim(0, 150)
ax.set_title("deaths against the line, over all 232 days", fontsize=15)
ax.legend(loc="upper right")
ax = axes[1]
ax.axhline(0, color=GREY_DIM, ls=":", lw=1.5)
ax.fill_between(
    data.index, residual, 0, where=residual >= 0, color=ACCENT, alpha=0.35, lw=0
)
ax.fill_between(
    data.index, residual, 0, where=residual < 0, color=SECOND, alpha=0.35, lw=0
)
ax.plot(data.index, residual, color=GREY, lw=1.8)
ax.set_title("the residual: deaths minus the line", fontsize=15)
for ax in axes:
    ax.axvline(vax, color=SECOND, lw=1.6)
    months(ax)
save(fig, "covid_residual")
print(
    f"full window: mse {mse(deaths, line):.0f}; residual negative from {run_start.date()} to the end; "
    f"{outside_after:.0%} of days after 6 Jan below the band"
)

# what does the ratio of deaths to tests do over time?
ratio = data["deaths_shifted"] / data["positivetests"] * 100
fig, ax = plt.subplots(figsize=(14, 5.0))
ax.plot(data.index, ratio, color=GREY, lw=2.6)
ax.axvline(vax, color=SECOND, lw=1.8)
ax.annotate(
    "first vaccination",
    (vax, ratio.max() * 0.95),
    xytext=(-8, 0),
    textcoords="offset points",
    color=SECOND,
    fontsize=15,
    ha="right",
)
ax.set_ylim(0, ratio.max() * 1.05)
ax.set_ylabel("deaths per 100 positive tests")
months(ax)
save(fig, "covid_ratio")
print(
    "ratio (deaths per 100 tests), monthly means:",
    ratio.resample("MS").mean().round(2).to_dict(),
)


# the share of positive tests that ends in death switches off; b and the noise stay
def covid_model(X, params):
    a, b, k, x0 = params
    return a * X[:, 0] * logistic(X[:, 1], k=k, x0=x0) + b


day = np.arange(len(data)).astype(float)
X = np.stack([tests, day], axis=1)
vaccination_day = float(np.argmax(data.index >= VACCINATION_START))
initial = [params_before[0], params_before[1], -0.1, vaccination_day + 30]
params = train_model(
    X,
    deaths,
    covid_model,
    mse,
    initial,
    bounds=[(0, 1.0), (0, None), (-1.0, 0), (0, len(data))],
)
a, b, k, x0 = params
halfway = data.index[int(round(x0))]
turning = covid_model(X, params)
switch = logistic(day, k=k, x0=x0)
fig, axes = plt.subplots(
    1, 2, figsize=(17, 5.0), gridspec_kw={"wspace": 0.2, "width_ratios": [1.3, 1]}
)
ax = axes[0]
ax.plot(data.index, deaths, color=GREY, lw=2.4, label="deaths, 14 days later")
ax.plot(data.index, turning, color=ACCENT, lw=3, label="a·tests × switch + b")
ax.set_ylim(0, 150)
ax.set_title("a switch on the share of tests that ends in death", fontsize=15)
ax.legend(loc="upper right")
ax = axes[1]
ax.plot(data.index, switch, color=ACCENT, lw=3)
ax.axvline(halfway, color=GREY_DIM, lw=1.4, ls="--")
ax.scatter([halfway], [0.5], color=SECOND, s=80, zorder=3)
ax.annotate(
    f"halfway: {halfway.day} {halfway:%B}",
    (halfway, 0.5),
    xytext=(12, 4),
    textcoords="offset points",
    color=SECOND,
    fontsize=15,
)
ax.set_ylim(-0.02, 1.08)
ax.set_title("the switch the fit found", fontsize=15)
ax.set_ylabel("share of the old a")
for ax in axes:
    ax.axvline(vax, color=SECOND, lw=1.6)
    months(ax)
axes[1].annotate(
    "first vaccination",
    (vax, 0.12),
    xytext=(-8, 0),
    textcoords="offset points",
    color=SECOND,
    fontsize=14,
    ha="right",
)
save(fig, "covid_switch")
print(
    f"switch: a={a:.4f} b={b:.1f} k={k:.3f} x0=day {x0:.0f} = {halfway.date()} "
    f"({(halfway - pd.Timestamp(VACCINATION_START)).days} days after the first vaccination); "
    f"mse line {mse(deaths, line):.0f} -> switch {mse(deaths, turning):.0f}"
)
for d in [
    "2021-01-06",
    "2021-02-01",
    "2021-03-01",
    "2021-04-01",
    "2021-05-01",
    "2021-06-01",
]:
    i = int(np.argmax(data.index >= d))
    print(f"  switch on {d}: {switch[i]:.2f}")

# the two knobs of a logistic switch: x0 says when, k says how fast
fig, axes = plt.subplots(
    1, 2, figsize=(17, 5.0), sharey=True, gridspec_kw={"wspace": 0.12}
)
for ax, variants, title in [
    (
        axes[0],
        [(k * 3, x0, "steeper k"), (k, x0, "the fitted k"), (k / 3, x0, "flatter k")],
        "k: how fast it turns",
    ),
    (
        axes[1],
        [
            (k, x0 - 40, "x0 earlier"),
            (k, x0, "the fitted x0"),
            (k, x0 + 40, "x0 later"),
        ],
        "x0: when it is halfway",
    ),
]:
    for (kk, xx, label), color, width in zip(
        variants, [GREY_FAINT, ACCENT, GREY_DIM], [2.2, 3.4, 2.2]
    ):
        ax.plot(
            data.index, logistic(day, k=kk, x0=xx), color=color, lw=width, label=label
        )
    ax.axhline(0.5, color=GREY_FAINT, lw=1, ls=":")
    ax.scatter([halfway], [0.5], color=SECOND, s=70, zorder=3)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="lower left")
    months(ax)
axes[0].set_yticks([0, 0.5, 1])
axes[0].set_yticklabels(
    [
        "0: no deaths\nfrom positive tests",
        "0.5",
        "1: the old share\nof tests ends in death",
    ]
)
save(fig, "switch_knobs")

# is it done? the residual once the ratio may turn
after_residual = deaths - turning
fits = DistributionFitter(seed=42).fit(after_residual, discrete=False)
fitted = by_name(fits)
fig, axes = plt.subplots(
    1, 2, figsize=(17, 5.0), gridspec_kw={"wspace": 0.2, "width_ratios": [1.6, 1]}
)
ax = axes[0]
ax.axhline(0, color=GREY_DIM, ls=":", lw=1.5)
ax.fill_between(
    data.index,
    after_residual,
    0,
    where=after_residual >= 0,
    color=ACCENT,
    alpha=0.35,
    lw=0,
)
ax.fill_between(
    data.index,
    after_residual,
    0,
    where=after_residual < 0,
    color=SECOND,
    alpha=0.35,
    lw=0,
)
ax.plot(data.index, after_residual, color=GREY, lw=1.8)
ax.axvline(vax, color=SECOND, lw=1.6)
ax.set_ylim(-60, 45)
ax.set_title("the residual, once the ratio may turn", fontsize=15)
months(ax)
ax = axes[1]
ax.hist(after_residual, bins=24, density=True, color=GREY_FAINT, edgecolor=BG, lw=0.4)
xr = np.linspace(after_residual.min() - 5, after_residual.max() + 5, 400)
table = fit_table(fits)
top3 = table.distribution.head(3).tolist()
for fam, color, ls in zip(top3, [ACCENT, GREY, SECOND], ["-", "-", "--"]):
    ax.plot(xr, fitted[fam].frozen_dist.pdf(xr), color=color, lw=2.6, ls=ls, label=fam)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_title("its distribution: the top three tie", fontsize=15)
ax.legend(loc="upper left", fontsize=13)
save(fig, "covid_after")
print(
    table[["distribution", "log_likelihood", "ks_pvalue", "best_likelihood", "best_ks"]]
    .head(6)
    .to_string()
)


# ---------------------------------------------------------------- 05.1: a finding about nothing
msgs = build_irc_pipeline().apply(load_showcase("ubuntu_irc"))
uk = (
    Pipeline()
    .add(Filter, expr="channel == '#ubuntu-uk'")
    .add(
        RegexFeature,
        name="words",
        column="message",
        pattern=r"\S+",
        feature="words",
        mode="count",
    )
    .add(
        RegexFeature,
        name="caps",
        column="message",
        pattern=r"^[A-Z]",
        feature="starts_upper",
        mode="has",
    )
    .add(
        RegexFeature,
        name="exclaims",
        column="message",
        pattern="!",
        feature="exclaims",
        mode="count",
    )
    .apply(msgs)
)
uk["length"] = uk.message.str.len()
uk["night"] = uk.hh.between(0, 5)
uk["is_weekend"] = uk.date.dt.dayofweek >= 5
authors = uk.groupby("author").agg(
    n=("message", "size"),
    mean_length=("length", "mean"),
    median_length=("length", "median"),
    mean_words=("words", "mean"),
    night_share=("night", "mean"),
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
authors["unique_share"] = (
    uk.groupby("author").message.nunique().reindex(authors.index) / authors.n
)
authors["hour_spread"] = uk.groupby("author").hh.std().reindex(authors.index)
METRICS = [
    "mean_length", "median_length", "mean_words", "night_share", "upper_share",
    "question_rate", "exclaim_rate", "url_share", "address_share", "active_days",
    "msgs_per_day", "unique_share", "hour_spread", "weekend_share", "n",
]  # fmt: skip
rng = np.random.default_rng(60)
authors["cohort"] = rng.permutation(["A", "B"] * (len(authors) // 2 + 1))[
    : len(authors)
]

comparison = []
for metric in METRICS:
    ca = authors.loc[authors.cohort == "A", metric].dropna()
    cb = authors.loc[authors.cohort == "B", metric].dropna()
    comparison.append(
        {
            "metric": metric,
            "A": ca.mean(),
            "B": cb.mean(),
            "p": stats.ttest_ind(ca, cb, equal_var=False).pvalue,
        }
    )
comparison = pd.DataFrame(comparison).sort_values("p").reset_index(drop=True)
winner = comparison.metric[0]
fig, ax = plt.subplots(figsize=(15, 5.4))
colors = [ACCENT if m == winner else GREY_FAINT for m in comparison.metric]
ax.bar(comparison.metric.str.replace("_", " "), comparison.p, color=colors, width=0.7)
ax.axhline(0.05, color=SECOND, lw=1.8)
ax.annotate(
    "p = 0.05",
    (14.4, 0.05),
    xytext=(0, 6),
    textcoords="offset points",
    color=SECOND,
    fontsize=14,
    ha="right",
)
ax.set_yscale("log")
ax.set_ylabel("p-value, cohort A against B (log scale)")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=14)
save(fig, "hunt")
print(
    f"hunt: {len(authors)} authors; {comparison.head(3).round(4).to_string(index=False)}"
)


def hunt(seed):
    labels = pd.Series(
        np.random.default_rng(seed).permutation(authors.cohort.to_numpy()),
        index=authors.index,
    )
    found = 0
    for metric in METRICS:
        ca = authors.loc[labels == "A", metric].dropna()
        cb = authors.loc[labels == "B", metric].dropna()
        found += stats.ttest_ind(ca, cb, equal_var=False).pvalue < 0.05
    return found


findings = pd.Series([hunt(seed) for seed in range(300)])
share = findings.value_counts(normalize=True).sort_index()
eigenvalues = np.linalg.eigvalsh(authors[METRICS].corr().fillna(0))
effective = (eigenvalues.sum() ** 2) / (eigenvalues**2).sum()
nothing = (findings == 0).mean()
bars = [
    ("measured:\n300 hunts over random labels", 1 - nothing, ACCENT),
    ("formula:\n15 independent tests", 1 - 0.95**15, GREY_DIM),
    (
        f"formula:\nabout {effective:.0f} independent tests",
        1 - 0.95**effective,
        GREY_FAINT,
    ),
]
fig, ax = plt.subplots(figsize=(13, 5.2))
labels, values, colors = zip(*bars[::-1])
ax.barh(labels, values, height=0.6, color=colors)
for y, value in enumerate(values):
    ax.text(value + 0.01, y, f"{value:.0%}", va="center", fontsize=18)
ax.set_xlim(0, 0.7)
ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
ax.set_xlabel("share of hunts with at least one “finding” (p < 0.05)")
plt.setp(ax.get_yticklabels(), fontsize=16)
save(fig, "null_hunts")
print(
    f"null hunts: something in {(findings > 0).mean():.0%}; formula {1 - 0.95**15:.0%}; "
    f"effective metrics {effective:.1f} -> {1 - 0.95**effective:.0%}; distribution {share.round(3).to_dict()}"
)


def p_of(group, metric):
    return stats.ttest_ind(
        group.loc[group.cohort == "A", metric].dropna(),
        group.loc[group.cohort == "B", metric].dropna(),
        equal_var=False,
    ).pvalue


def hunt_in(group):
    """The metric with the smallest p-value between cohort A and B in this group of people."""
    pvalues = {metric: p_of(group, metric) for metric in METRICS}
    return min(pvalues.items(), key=lambda item: item[1])


def split_people(seed):
    """Fresh random cohorts, then half the people to hunt in and half to hold out."""
    rng = np.random.default_rng(seed)
    labelled = authors.assign(cohort=rng.permutation(authors.cohort.to_numpy()))
    order = rng.permutation(labelled.index.to_numpy())
    return labelled.loc[order[: len(order) // 2]], labelled.loc[
        order[len(order) // 2 :]
    ]


# how often does a hunt in one half of the people carry over to the other half?
carried = []
for seed in range(200):
    search, holdout = split_people(seed)
    metric, p_search = hunt_in(search)
    carried.append((seed, metric, p_search, p_of(holdout, metric)))
carried = pd.DataFrame(carried, columns=["seed", "metric", "p_search", "p_holdout"])
found = carried[carried.p_search < 0.05]
print(
    f"holdout by people: the hunt finds p < 0.05 in {len(found) / len(carried):.0%} of 200 splits; "
    f"of those, the same metric has p < 0.05 in the held-out half "
    f"{(found.p_holdout < 0.05).mean():.0%} of the time"
)
# the figure shows the first split whose hunt found something, whatever the holdout said
shown = carried[carried.p_search < 0.01].iloc[0]
search, holdout = split_people(int(shown.seed))
metric = shown.metric
fig, axes = plt.subplots(
    1, 2, figsize=(14, 5.2), sharey=True, gridspec_kw={"wspace": 0.1}
)
for ax, (title, group, pvalue) in zip(
    axes,
    [
        ("where the hunt searched", search, shown.p_search),
        ("new people, held out", holdout, shown.p_holdout),
    ],
):
    for j, (cohort, color) in enumerate([("A", SECOND), ("B", ACCENT)]):
        values = group.loc[group.cohort == cohort, metric].dropna()
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        ax.bar(j, values.mean(), width=0.6, color=color, alpha=0.85)
        ax.errorbar(j, values.mean(), yerr=half, color=GREY, capsize=8, lw=1.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cohort A", "cohort B"], fontsize=16)
    ax.set_title(f"{title}: {len(group)} people, p = {pvalue:.3f}", fontsize=16)
axes[0].set_ylabel(metric.replace("_", " "))
save(fig, "held_out")
print(
    f"held out (people): split seed {int(shown.seed)}, metric {metric}, "
    f"search p {shown.p_search:.4f}, holdout p {shown.p_holdout:.3f}"
)

# claim 1: release days, against two different baselines
RELEASES = pd.to_datetime([
    "2013-04-25", "2013-10-17", "2014-04-17", "2014-10-23", "2015-04-23",
    "2015-10-22", "2016-04-21", "2016-10-13", "2017-04-13", "2017-10-19",
])  # fmt: skip
thursdays = (
    Pipeline()
    .add(Filter, expr="day_name == 'Thursday'")
    .add(FlagDates, column="date", dates=RELEASES, feature="is_release")
    .apply(uk)
)
thursday_counts = Pipeline().add(GroupAgg, by=["date", "is_release"]).apply(thursdays)
global_median = thursday_counts.loc[~thursday_counts.is_release, "n"].median()
release_counts = thursday_counts.loc[thursday_counts.is_release, "n"]


def local_baseline(frame, weeks=4):
    counts = frame[frame.date.dt.dayofweek == 3].groupby("date").size().sort_index()
    rows = []
    for release_date in RELEASES:
        if release_date not in counts.index:
            continue
        window = counts[
            (counts.index >= release_date - pd.Timedelta(weeks=weeks))
            & (counts.index <= release_date + pd.Timedelta(weeks=weeks))
        ]
        rows.append(
            {
                "release": release_date,
                "nearby Thursday": window.drop(release_date).median(),
                "on the day": counts[release_date],
            }
        )
    out = pd.DataFrame(rows)
    out["lift"] = out["on the day"] / out["nearby Thursday"]
    return out


local = local_baseline(uk)
above_local = int((local.lift > 1).sum())
sign_p = stats.binomtest(above_local, len(local), 0.5).pvalue
fig, ax = plt.subplots(figsize=(13, 5.6))
ypos = np.arange(len(local))[::-1]
ax.axvline(global_median, color=GREY_DIM, lw=1.8, ls="--")
ax.annotate(
    f"a typical Thursday, all five years: {global_median:.0f}",
    (global_median, len(local) - 0.2),
    xytext=(8, 0),
    textcoords="offset points",
    color=GREY_DIM,
    fontsize=14,
)
for yy, (_, row) in zip(ypos, local.iterrows()):
    ax.plot(
        [row["nearby Thursday"], row["on the day"]],
        [yy, yy],
        color=GREY_FAINT,
        lw=3,
        zorder=1,
    )
ax.scatter(
    local["nearby Thursday"],
    ypos,
    color=ACCENT,
    s=70,
    zorder=2,
    label="the Thursdays around it (median)",
)
ax.scatter(local["on the day"], ypos, color=SECOND, s=90, zorder=3, label="release day")
ax.set_yticks(ypos)
ax.set_yticklabels([d.strftime("%b %Y") for d in local.release])
ax.set_xlabel("messages in #ubuntu-uk on that Thursday")
ax.set_ylim(-0.7, len(local) + 0.3)
ax.legend(loc="lower right")
save(fig, "release")
print(
    f"release: above global median {(release_counts > global_median).sum()}/{len(release_counts)}; "
    f"above local {above_local}/{len(local)}, median lift {local.lift.median():.2f}, sign p {sign_p:.3f}"
)

# claim 3: weekday messages, per message and per author
weekday_len = uk.loc[~uk.is_weekend, "length"]
weekend_len = uk.loc[uk.is_weekend, "length"]
p_message = stats.ttest_ind(weekday_len, weekend_len, equal_var=False).pvalue
by_author = uk.groupby(["author", "is_weekend"]).length.mean().unstack().dropna()
by_author = by_author[uk.groupby("author").size().reindex(by_author.index) >= 30]
by_author.columns = pd.Index(["weekday", "weekend"])
by_author["difference"] = by_author.weekday - by_author.weekend
paired = stats.ttest_rel(by_author.weekday, by_author.weekend)
longer = (by_author.difference > 0).mean()
weekend_posts = uk.loc[uk.is_weekend, "author"].value_counts()
top_weekend = weekend_posts.head(10) / weekend_posts.sum()
fig, axes = plt.subplots(
    1, 2, figsize=(17, 5.2), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.3]}
)
ax = axes[0]
ax.barh(top_weekend.index[::-1], top_weekend.values[::-1] * 100, color=ACCENT)
ax.set_xlabel("share of all weekend messages (%)")
ax.set_title(
    f"10 people write {top_weekend.sum():.0%} of the weekend messages", fontsize=16
)
ax = axes[1]
bins = np.linspace(by_author.difference.min(), by_author.difference.max(), 41)
ax.hist(
    by_author.difference[by_author.difference < 0], bins=bins, color=SECOND, alpha=0.8
)
ax.hist(
    by_author.difference[by_author.difference >= 0], bins=bins, color=ACCENT, alpha=0.8
)
ax.axvline(0, color=GREY_DIM, lw=1.4)
ax.set_title(
    f"one number per person: {1 - longer:.0%} longer in the weekend, {longer:.0%} on weekdays",
    fontsize=16,
)
ax.set_xlabel("mean message length, weekday − weekend (characters)")
ax.set_ylabel("people")
save(fig, "weekday")
weekday_mean = uk.loc[~uk.is_weekend].groupby("author").length.mean()
posters = weekend_posts.index.intersection(weekday_mean.index)
as_on_weekdays = (weekday_mean[posters] * weekend_posts[posters]).sum() / weekend_posts[
    posters
].sum()
print(
    f"weekday: top 10 weekend posters write {top_weekend.sum():.0%} of weekend messages; "
    f"weekend mean if weekend posters wrote as on weekdays: {as_on_weekdays:.2f}"
)
print(
    f"weekday: per message {weekday_len.mean():.2f} vs {weekend_len.mean():.2f}, p {p_message:.0e}, n {len(uk):,}; "
    f"per author diff {by_author.difference.mean():+.2f}, paired p {paired.pvalue:.3f}, longer {longer:.0%}, n {len(by_author)}"
)


# ---------------------------------------------------------------- 05.2: what a correlation hides
penguins = load_showcase("penguins_raw")
select = [
    "Species", "Island", "Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)",
    "Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Sex", "Body Mass (g)",
]  # fmt: skip
subset = penguins[select].dropna()
subset["species"] = subset["Species"].str.split(" ").str[0]
floats = subset.select_dtypes(include="float64")


def within_species(column):
    return subset[column] - subset.groupby("species")[column].transform("mean")


SHORT = {
    "Culmen Length (mm)": "bill length",
    "Culmen Depth (mm)": "bill depth",
    "Flipper Length (mm)": "flipper",
    "Delta 15 N (o/oo)": "δ15N",
    "Delta 13 C (o/oo)": "δ13C",
    "Body Mass (g)": "body mass",
}
pairs = []
columns = list(floats.columns)
for i, ca in enumerate(columns):
    for cb in columns[i + 1 :]:
        pairs.append(
            {
                "pair": f"{SHORT[ca]} / {SHORT[cb]}",
                "pooled": subset[ca].corr(subset[cb]),
                "within": within_species(ca).corr(within_species(cb)),
            }
        )
pairs = pd.DataFrame(pairs).sort_values("pooled").reset_index(drop=True)
flips = int((np.sign(pairs.pooled) != np.sign(pairs.within)).sum())

bx, by = "Culmen Length (mm)", "Culmen Depth (mm)"
species_colors = dict(zip(sorted(subset.species.unique()), [ACCENT, THIRD, GREY_DIM]))
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), sharex=True, sharey=True)
ax = axes[0]
ax.scatter(subset[bx], subset[by], color=GREY_FAINT, s=22)
slope, intercept = np.polyfit(subset[bx], subset[by], 1)
xs = np.linspace(subset[bx].min(), subset[bx].max(), 50)
ax.plot(xs, slope * xs + intercept, color=SECOND, lw=3)
ax.set_title(
    f"all penguins at once: r = {subset[bx].corr(subset[by]):+.2f}", fontsize=16
)
ax = axes[1]
for species, group in subset.groupby("species"):
    color = species_colors[species]
    ax.scatter(group[bx], group[by], color=color, s=22, alpha=0.55)
    slope, intercept = np.polyfit(group[bx], group[by], 1)
    xs = np.linspace(group[bx].min(), group[bx].max(), 50)
    ax.plot(
        xs,
        slope * xs + intercept,
        color=color,
        lw=3,
        label=f"{species}: r = {group[bx].corr(group[by]):+.2f}",
    )
ax.set_title("one species at a time", fontsize=16)
ax.legend(loc="lower right")
for ax in axes:
    ax.set_xlabel("bill length (mm)")
axes[0].set_ylabel("bill depth (mm)")
save(fig, "simpson")

fig, ax = plt.subplots(figsize=(12, 6.4))
ypos = np.arange(len(pairs))
for yy, (_, row) in zip(ypos, pairs.iterrows()):
    ax.plot([row.pooled, row.within], [yy, yy], color=GREY_FAINT, lw=3, zorder=1)
ax.scatter(
    pairs.pooled, ypos, color=SECOND, s=70, zorder=2, label="all penguins pooled"
)
ax.scatter(pairs.within, ypos, color=ACCENT, s=90, zorder=3, label="within species")
ax.axvline(0, color=GREY_DIM, lw=1.6)
ax.set_yticks(ypos)
ax.set_yticklabels(pairs.pair, fontsize=14)
ax.set_xlabel("correlation r")
ax.set_xlim(-1, 1)
ax.legend(loc="upper left")
save(fig, "pairs")
print(f"penguins: {flips} of {len(pairs)} change sign")
print(pairs.round(2).to_string(index=False))

# regularisation: 100 features, a few of them useful
sim = pd.read_csv(ROOT / "data" / "sim" / "correlation.csv")
features = sim.drop(columns=["target"])
correlations = features.corrwith(sim.target)
grid_search = GridSearchCV(
    SGDRegressor(penalty="elasticnet", random_state=42),
    param_grid={
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        "l1_ratio": [0.1, 0.15, 0.3, 0.6, 0.75, 0.9, 0.95, 0.99, 1],
    },
    cv=5,
)
grid_search.fit(features.values, sim.target.values)
weights = pd.Series(grid_search.best_estimator_.coef_, index=features.columns)
chosen = weights.abs().sort_values(ascending=False).index[:10]
fig, axes = plt.subplots(1, 2, figsize=(17, 5.0), gridspec_kw={"wspace": 0.18})
idx = np.arange(len(features.columns))
mark = features.columns.isin(chosen)
for ax, values, title in [
    (axes[0], correlations, "correlation of each feature with the target"),
    (
        axes[1],
        weights,
        f"weights of a penalised regression (l1_ratio {grid_search.best_params_['l1_ratio']})",
    ),
]:
    ax.axhline(0, color=GREY_FAINT, lw=1)
    ax.scatter(idx[~mark], values.values[~mark], color=GREY_FAINT, s=30)
    ax.scatter(idx[mark], values.values[mark], color=ACCENT, s=60, zorder=3)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("feature 1 … 100")
    ax.set_xticks([])
save(fig, "regularisation")
print(
    f"regularisation: best {grid_search.best_params_}; nonzero weights {(weights.abs() > 1e-6).sum()}; "
    f"|r| > 0.2: {(correlations.abs() > 0.2).sum()}; rank of the 10 chosen by |r|: "
    f"{sorted(correlations.abs().rank(ascending=False)[chosen].astype(int).tolist())}"
)

# the same weights as the penalty grows: L1 sets them to exactly 0, L2 only shrinks them
l1_alphas, l1_path, _ = lasso_path(features.values, sim.target.values, n_alphas=80)
penalties = l1_alphas
l2_path = np.array(
    [
        Ridge(alpha=alpha * len(sim)).fit(features.values, sim.target.values).coef_
        for alpha in penalties
    ]
).T
mark = features.columns.isin(chosen)
fig, axes = plt.subplots(
    1, 2, figsize=(17, 5.2), sharey=True, gridspec_kw={"wspace": 0.1}
)
for ax, (path, alphas, title) in zip(
    axes,
    [
        (l1_path, l1_alphas, "L1: weights that don't help become exactly 0"),
        (l2_path, penalties, "L2: every weight shrinks, none becomes 0"),
    ],
):
    for weights_of, useful in zip(path, mark):
        ax.plot(
            alphas,
            weights_of,
            color=ACCENT if useful else GREY_FAINT,
            lw=2.2 if useful else 1.0,
            zorder=2 if useful else 1,
        )
    ax.axhline(0, color=GREY_DIM, lw=1)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("penalty α (strict on the left, mild on the right)")
    ax.set_title(title, fontsize=16)
axes[0].set_ylabel("weight of each feature")
save(fig, "reg_paths")
print(
    f"paths: L1 nonzero weights at the strictest / mildest penalty: "
    f"{(np.abs(l1_path[:, 0]) > 1e-6).sum()} / {(np.abs(l1_path[:, -1]) > 1e-6).sum()}"
)


# ---------------------------------------------------------------- 05.5: how people type
typing = (
    Pipeline()
    .add(Filter, expr="channel == '#ubuntu-uk' and not is_action")
    .add(
        RegexFeature,
        name="caps",
        column="message",
        pattern=r"^[A-Z]",
        feature="starts_upper",
        mode="has",
    )
    .apply(msgs)
)
typing["length"] = typing.message.str.len()
typing["night"] = typing.hh.between(0, 5)
top8 = Pipeline().add(CountValues, column="author").add(Head, n=8).apply(typing)
people = typing[typing.author.isin(top8.author)].copy()
baseline = top8.n.iloc[0] / top8.n.sum()

train_msgs, test_msgs = train_test_split(
    people, test_size=0.25, random_state=42, stratify=people.author
)
words = TfidfVectorizer(min_df=5, sublinear_tf=True)
one_liner = LogisticRegression(max_iter=1000, C=5)
one_liner.fit(words.fit_transform(train_msgs.message), train_msgs.author)
one_message_accuracy = one_liner.score(
    words.transform(test_msgs.message), test_msgs.author
)
vocabulary = np.array(words.get_feature_names_out())
print(
    f"typing: {len(people):,} messages; baseline {baseline:.1%}; one message {one_message_accuracy:.1%}"
)
for i, author in enumerate(one_liner.classes_):
    print(
        f"  {author:12s} {', '.join(vocabulary[np.argsort(one_liner.coef_[i])[::-1][:8]])}"
    )

habits = (
    Pipeline()
    .add(RegexFeature, name="nosed", column="message", pattern=r"[:;=]-[)DPp(\]]", feature="nosed", mode="count")
    .add(RegexFeature, name="noseless", column="message", pattern=r"[:;=][)DPp(\]]|\bXD\b", feature="noseless", mode="count")
    .add(RegexFeature, name="smiley", column="message", pattern=r"[☺☻☹㋛]", feature="unicode_smiley", mode="count")
    .add(
        RegexFeature, name="apos_drop", column="message",
        pattern=r"(?i)\b(?:dont|doesnt|didnt|cant|wont|isnt|im|thats|its|ive|youre)\b",
        feature="apos_drop", mode="count",
    )
    .add(
        RegexFeature, name="apos_keep", column="message",
        pattern=r"(?i)\b(?:don't|doesn't|didn't|can't|won't|isn't|i'm|that's|it's|i've|you're)\b",
        feature="apos_keep", mode="count",
    )
    .apply(people)
)  # fmt: skip
habits["addresses"] = habits.addressed_to.notna()
per_author = habits.groupby("author")
smileys = per_author[["nosed", "noseless", "unicode_smiley"]].sum()
dialect = smileys.div(smileys.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(15, 5.0))
order = dialect.sort_values("nosed", ascending=False).index
xs = np.arange(len(order))
for j, (column, label, color) in enumerate(
    [
        ("nosed", "with a nose  :-)", ACCENT),
        ("noseless", "without  :)", GREY_FAINT),
        ("unicode_smiley", "unicode  ☺", THIRD),
    ]
):
    ax.bar(
        xs + (j - 1) * 0.27,
        dialect.loc[order, column],
        width=0.26,
        color=color,
        label=label,
    )
ax.set_xticks(xs)
ax.set_xticklabels(order, rotation=20, ha="right")
ax.set_ylabel("share of that author's smileys (%)")
ax.legend(loc="upper right")
save(fig, "smileys")

has_smiley = habits[["nosed", "noseless", "unicode_smiley"]].sum(axis=1) > 0
has_contraction = habits[["apos_drop", "apos_keep"]].sum(axis=1) > 0
print(f"  no fingerprint in {(~has_smiley & ~has_contraction).mean():.1%} of messages")


class BlockMessages(TransformBase):
    """Pool each author's shuffled messages into fixed-size blocks: habit means, text joined."""

    def transform(self, data, features, size=50, seed=0):
        shuffled = data.sample(frac=1, random_state=seed)
        shuffled["block"] = shuffled.groupby("author").cumcount() // size
        grouped = shuffled.groupby(["author", "block"])
        blocks = grouped[features].mean()
        blocks["text"] = grouped.message.apply(" ".join)
        return blocks[grouped.size() == size].reset_index()


FEATURES = ["nosed", "noseless", "unicode_smiley", "apos_drop", "apos_keep",
            "starts_upper", "addresses", "night", "length", "n_question"]  # fmt: skip
blocks = Pipeline().add(BlockMessages, features=FEATURES, size=50).apply(habits)
train, test = train_test_split(
    blocks, test_size=0.25, random_state=42, stratify=blocks.author
)
vec = TfidfVectorizer(min_df=3, sublinear_tf=True)
vocabulary_model = LogisticRegression(max_iter=1000, C=5)
vocabulary_model.fit(vec.fit_transform(train.text), train.author)
scaler = StandardScaler().fit(train[FEATURES])
habit_model = LogisticRegression(max_iter=2000)
habit_model.fit(scaler.transform(train[FEATURES]), train.author)
scores = pd.Series(
    {
        "guess the busiest\nauthor every time": baseline,
        "one message,\nall words": one_message_accuracy,
        "50 messages,\nall words": vocabulary_model.score(
            vec.transform(test.text), test.author
        ),
        "50 messages,\nten habits": habit_model.score(
            scaler.transform(test[FEATURES]), test.author
        ),
    }
)
fig, ax = plt.subplots(figsize=(13, 5.0))
bars = ax.bar(
    scores.index,
    scores.values,
    color=[GREY_FAINT, GREY_FAINT, GREY_DIM, ACCENT],
    width=0.6,
)
for bar, value in zip(bars, scores.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.0%}",
        ha="center",
        fontsize=16,
    )
ax.set_ylim(0, 1.12)
ax.set_ylabel("accuracy on held-out messages")
save(fig, "blocks")
print(
    f"  blocks: {len(blocks):,} blocks; vocabulary {len(vec.vocabulary_):,} words; scores {scores.round(3).to_dict()}"
)

early_msgs = habits[habits.date.dt.year <= 2014]
late_msgs = habits[habits.date.dt.year >= 2016]
checks = [
    ("addresses someone by nick (%)", "addresses", "mean", 100),
    ("starts with a capital (%)", "starts_upper", "mean", 100),
    ("median message length", "length", "median", 1),
    ("asks a question (%)", "n_question", "mean", 100),
]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.9), gridspec_kw={"wspace": 0.32})
for ax, (label, column, how, scale) in zip(axes, checks):
    pair = pd.concat(
        [
            early_msgs.groupby("author")[column].agg(how) * scale,
            late_msgs.groupby("author")[column].agg(how) * scale,
        ],
        axis=1,
    ).dropna()
    pair.columns = pd.Index(["early", "late"])
    rho = pair.early.corr(pair.late, method="spearman")
    lo, hi = pair.min().min(), pair.max().max()
    pad = (hi - lo) * 0.08
    ax.plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY_FAINT, ls="--", lw=1.2
    )
    ax.scatter(
        pair.early, pair.late, color=SECOND if rho < 0.8 else ACCENT, s=70, zorder=3
    )
    ax.set_title(f"{label}\nspearman {rho:.2f}", fontsize=14.5)
    ax.set_xlabel("2013–2014")
    print(f"  stability {label}: spearman {rho:.2f}")
axes[0].set_ylabel("2016–2017")
save(fig, "stability")


# ---------- formulas: typeset with mathtext, one line each, at 2x for the slides ----------

# each formula is a row of (tex, colour) pieces set on one baseline, so a symbol can carry
# the colour it has elsewhere on the slide
FORMULAS = {
    "formula_linear": [
        (r"$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_{100}\, x_{100}$", TEXT)
    ],
    "formula_l1": [(r"$\mathrm{L1}\!\!:\quad \mathrm{MSE} + \alpha \sum\, |W|$", TEXT)],
    "formula_l2": [(r"$\mathrm{L2}\!\!:\quad \mathrm{MSE} + \alpha \sum\, W^2$", TEXT)],
    "formula_epsilon": [
        (r"$\varepsilon$", SECOND),
        (r"$\sim\, \mathcal{N}(0,\, \sigma)$", TEXT),
    ],
}
FORMULA_SIZE = 30  # points
with plt.rc_context({"mathtext.fontset": "cm"}):
    for name, pieces in FORMULAS.items():
        # one inch wide, so a figure fraction is a length in inches
        fig = plt.figure(figsize=(1, 1))
        x = 0.0
        for tex, color in pieces:
            piece = fig.text(x, 0, tex, fontsize=FORMULA_SIZE, color=color)
            x += piece.get_window_extent().width / fig.dpi
        fig.savefig(
            OUT / f"{name}.png",
            dpi=144,
            bbox_inches="tight",
            pad_inches=0.04,
            transparent=True,
        )
        plt.close(fig)
        print("wrote", name)
