"""Figures for the lesson 3 slides, on the Nocturne palette.

Run from the repo root:  uv run python presentations/plots_lesson3.py
Writes PNGs to presentations/img/lesson3/.
Data: data/showcase/flights.csv, ubuntu_irc_release_hourly.csv, and the SILSO sunspot file
(downloaded to ~/.cache/mads-dav/raw/sunspots.txt if missing).
"""

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "showcase"
OUT = ROOT / "presentations" / "img" / "lesson3"
OUT.mkdir(parents=True, exist_ok=True)

# Nocturne tokens
BG = "#161826"
TEXT = "#e9e9ed"
ACCENT = "#9184d9"
ACCENT_LIGHT = "#c3bbea"
SECOND = "#e8b04b"  # warm contrast hue for two-series plots (purple vs amber)
GREY = "#dcdde3"  # main neutral line: near text colour
GREY_DIM = "#9c9eac"  # secondary neutral line
GREY_FAINT = "#5b5e72"  # fills (histogram bars, bands): visibly above the ground

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


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", name)


def clock(ax):
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"])
    ax.set_xlabel("hour (UTC)")


# ---------------------------------------------------------------- flights
flights = pd.read_csv(DATA / "flights.csv")
flights["date"] = pd.to_datetime(
    flights["year"].astype(str) + "-" + flights["month"], format="%Y-%b"
)
flights = flights.set_index("date").sort_index()[["passengers"]]
p = flights["passengers"]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(p.index, p, color=GREY_DIM, lw=1.6, label="raw, monthly")
ax.plot(
    p.index, p.rolling(3).mean(), color=SECOND, lw=2.2, ls="--", label="3-month window"
)
ax.plot(p.index, p.rolling(12).mean(), color=ACCENT, lw=3, label="12-month window")
ax.set_ylabel("passengers")
ax.legend(loc="upper left")
save(fig, "flights_smoothing")

holey = flights.drop(flights.index[60:64])
full_index = pd.date_range(flights.index.min(), flights.index.max(), freq="MS")
reindexed = holey.reindex(full_index)
gap = (flights.index[60], flights.index[63])
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for ax, s, title in [
    (axes[0], holey["passengers"], "df.plot()  — rows dropped, the line continues"),
    (axes[1], reindexed["passengers"], "df.reindex(full_range)  — the gap is a gap"),
]:
    ax.axvspan(gap[0], gap[1], color=ACCENT, alpha=0.18, lw=0)
    ax.plot(s.index, s, color=ACCENT if ax is axes[1] else GREY, marker="o", ms=3)
    ax.set_title(title, family="monospace", fontsize=14)
    ax.set_xlim(pd.Timestamp("1952-06-01"), pd.Timestamp("1956-06-01"))
axes[0].set_ylabel("passengers")
save(fig, "flights_gap")

# ---------------------------------------------------------------- ubuntu irc
hourly = pd.read_csv(DATA / "ubuntu_irc_release_hourly.csv", parse_dates=["date"])
hourly["is_release"] = hourly["is_release"].astype(str).str.lower().eq("true")
other = [c for c in hourly.channel.unique() if c != "#ubuntu-it"][0]
CHANNELS = {other: SECOND, "#ubuntu-it": ACCENT}


def day_shares(df):
    """Each day normalised to its own total: share of the day's messages per hour."""
    tot = df.groupby(["channel", "date"]).messages.transform("sum")
    return df.assign(share=df.messages / tot)


shares = day_shares(hourly)
ordinary = (
    shares[~shares.is_release].groupby(["channel", "hour"]).share.mean().unstack(0)
)
release = shares[shares.is_release].groupby(["channel", "hour"]).share.mean().unstack(0)

fig, ax = plt.subplots(figsize=(12, 5))
for ch, col in CHANNELS.items():
    ax.plot(ordinary.index, ordinary[ch] * 100, color=col, lw=3, label=ch)
clock(ax)
ax.set_ylabel("share of the day's messages, %")
ax.set_title("Ordinary Thursdays — the model")
ax.legend()
save(fig, "irc_ordinary_day")

# pooling: raw counts of each release day vs each day's own share
rel = hourly[hourly.is_release & (hourly.channel == "#ubuntu-it")]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
loud = rel.groupby("date").messages.sum().idxmax()
for d, g in rel.groupby("date"):
    is_loud = d == loud
    axes[0].plot(
        g.hour,
        g.messages,
        color=ACCENT if is_loud else GREY_DIM,
        lw=3 if is_loud else 1.4,
    )
    axes[1].plot(
        g.hour,
        g.messages / g.messages.sum() * 100,
        color=ACCENT if is_loud else GREY_DIM,
        lw=3 if is_loud else 1.4,
    )
axes[0].set_title(f"raw messages per hour · {loud:%Y-%m-%d} dominates")
axes[1].set_title("share of each day's own total, %")
for ax in axes:
    clock(ax)
save(fig, "irc_pooling")

# residual
resid = (release - ordinary) * 100
fig, ax = plt.subplots(figsize=(12, 5))
ax.axhline(0, color=GREY_DIM, ls=":", lw=1.5)
ax.axvline(14, color=ACCENT, lw=1, alpha=0.6)
for ch, col in CHANNELS.items():
    ax.plot(resid.index, resid[ch], color=col, lw=3, label=ch)
    peak = resid[ch].idxmax()
    ax.annotate(
        f"{int(peak)}:00  +{resid[ch].max():.1f} pts",
        (peak, resid[ch].max()),
        xytext=(10, 6),
        textcoords="offset points",
        color=col,
        fontsize=14,
    )
clock(ax)
ax.set_ylabel("release day − ordinary Thursday, pct points")
ax.set_title("Residual: what the release adds, hour by hour")
ax.legend(loc="upper left")
save(fig, "irc_residual")

# three-panel: release − model = residual (for #ubuntu-it)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
axes[0].plot(release.index, release["#ubuntu-it"] * 100, color=ACCENT, lw=3)
axes[0].set_title("release day  (10 days)")
axes[1].plot(ordinary.index, ordinary["#ubuntu-it"] * 100, color=GREY_DIM, lw=3)
axes[1].set_title("−  ordinary Thursday  (the model)")
axes[2].axhline(0, color=GREY_DIM, ls=":", lw=1.5)
axes[2].plot(resid.index, resid["#ubuntu-it"], color=ACCENT, lw=3)
axes[2].set_title("=  residual")
for ax in axes:
    clock(ax)
axes[0].set_ylabel("share, %")
save(fig, "irc_three_panel")

# night share with intervals
night = (
    shares[shares.hour.between(1, 6)]
    .groupby(["channel", "date", "is_release"])
    .share.sum()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(10, 5))
x = 0
ticks, labels = [], []
for ch in CHANNELS:
    for is_rel, col, lab in [
        (False, GREY_FAINT, "ordinary Thursday"),
        (True, ACCENT, "release day"),
    ]:
        s = night[(night.channel == ch) & (night.is_release == is_rel)].share * 100
        m, se = s.mean(), s.std(ddof=1) / np.sqrt(len(s))
        ax.bar(
            x,
            m,
            width=0.8,
            color=ACCENT if is_rel else GREY_FAINT,
            label=f"{lab} (n≈{len(s)})" if ch == list(CHANNELS)[0] else None,
        )
        ax.errorbar(x, m, yerr=1.96 * se, color=TEXT, capsize=6, lw=2)
        x += 1
    ticks.append(x - 1.5)
    labels.append(ch)
    x += 0.8
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_ylabel("share of messages 01:00–06:00, %")
ax.set_title("Do people stay up on release night?  Bars with 95% intervals")
ax.legend()
save(fig, "irc_night_share")

# ---------------------------------------------------------------- permutation test
days = hourly.groupby(["channel", "date", "is_release"]).messages.sum().reset_index()


def lift(d):
    return (
        d.loc[d.is_release, "messages"].mean() / d.loc[~d.is_release, "messages"].mean()
    )


def null_lifts(d, n_iter=2000, seed=4):
    rng = np.random.default_rng(seed)
    labels = d.is_release.to_numpy()
    out = np.empty(n_iter)
    for i in range(n_iter):
        out[i] = lift(d.assign(is_release=rng.permutation(labels)))
    return out


test_channels = [c for c in ["#ubuntu", "#ubuntu-it"] if c in set(days.channel)]
results = {}
for ch in test_channels:
    d = days[days.channel == ch]
    results[ch] = (lift(d), null_lifts(d), d.loc[~d.is_release, "messages"])

# single-channel picture, for the "shuffle" slide
ch = test_channels[-1]
obs, null, ordn = results[ch]
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(null, bins=40, color=GREY_FAINT, edgecolor=BG)
ax.axvline(obs, color=ACCENT, lw=4)
p = (null >= obs).mean()
ax.text(
    obs,
    ax.get_ylim()[1] * 0.95,
    f"  observed {obs:.2f}×\n  p = {p:.3f}  ({int((null >= obs).sum())} of {len(null)} shuffles)",
    color=ACCENT_LIGHT,
    va="top",
    fontsize=15,
)
ax.set_xlabel("release ÷ ordinary, messages per day  —  2,000 shuffled labels")
ax.set_ylabel("shuffles")
ax.set_title(f"{ch}: what the lift looks like when the label means nothing")
save(fig, "perm_shuffle")

# two channels side by side, shared x
fig, axes = plt.subplots(1, len(test_channels), figsize=(14, 4.8), sharex=True)
axes = np.atleast_1d(axes)
lo = min(r[1].min() for r in results.values())
hi = max(max(r[1].max(), r[0]) for r in results.values()) * 1.05
for ax, ch in zip(axes, test_channels):
    obs, null, ordn = results[ch]
    ax.hist(null, bins=40, range=(lo, hi), color=GREY_FAINT, edgecolor=BG)
    ax.axvline(obs, color=ACCENT, lw=4)
    p = (null >= obs).mean()
    cv = ordn.std() / ordn.mean()
    ax.set_title(f"{ch}   lift {obs:.2f}×   p = {p:.3f}")
    ax.text(
        0.02,
        0.92,
        f"ordinary days wander ±{cv:.0%} of their mean",
        transform=ax.transAxes,
        color=GREY,
        fontsize=14,
    )
    ax.set_xlabel("release ÷ ordinary")
save(fig, "perm_two_channels")

# ---------------------------------------------------------------- sunspots
cache = Path.home() / ".cache" / "mads-dav" / "raw" / "sunspots.txt"
if not cache.exists():
    cache.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve("https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt", cache)
sun = pd.read_csv(
    cache,
    sep=r"\s+",
    header=None,
    names=[
        "year",
        "month",
        "decimal_date",
        "sunspots",
        "std_dev",
        "observations",
        "definitive",
    ],
)
sun["date"] = pd.to_datetime(
    pd.DataFrame({"year": sun.year, "month": sun.month, "day": 1})
)
sun = sun.set_index("date")["sunspots"]

# autocorrelation: shifted-copy sketch + acf with confidence band
acf_vals, conf = acf(sun, nlags=30 * 12, fft=True, alpha=0.05)
skip = 12 * 3
lag = int(np.argmax(acf_vals[skip:])) + skip
fig, axes = plt.subplots(
    1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.5]}
)
seg = sun["1900":"1960"]
axes[0].plot(seg.index, seg, color=SECOND, lw=1.8, label="the series")
axes[0].plot(
    seg.index + pd.DateOffset(months=lag),
    seg,
    color=ACCENT,
    lw=1.8,
    alpha=0.9,
    label=f"shifted by lag {lag} months",
)
axes[0].set_xlim(seg.index.min(), seg.index.max())
axes[0].legend(loc="upper left")
axes[0].set_title("Correlate the series with a shifted copy of itself")
lags = np.arange(len(acf_vals))
axes[1].fill_between(
    lags,
    conf[:, 0] - acf_vals,
    conf[:, 1] - acf_vals,
    color=GREY_FAINT,
    alpha=0.6,
    lw=0,
)
axes[1].vlines(lags, 0, acf_vals, color=ACCENT, lw=1.2)
axes[1].axhline(0, color=GREY_DIM, lw=1)
axes[1].annotate(
    f"first peak after the dip: lag {lag} months ≈ {lag / 12:.1f} years",
    (lag, acf_vals[lag]),
    xytext=(12, 8),
    textcoords="offset points",
    color=ACCENT_LIGHT,
    fontsize=14,
)
axes[1].set_xticks(np.arange(0, 361, 60))
axes[1].set_xticklabels([f"{m // 12} y" for m in np.arange(0, 361, 60)])
axes[1].set_xlabel("lag")
axes[1].set_title("Autocorrelation against lag")
save(fig, "sun_acf")

# decomposition with the 11-year period: observed vs seasonal, and the residual
res = seasonal_decompose(sun, model="additive", period=lag)
resid_var = res.resid.var() / sun.var()
resid_acf = acf(res.resid.dropna(), nlags=30 * 12, fft=True)
resid_lag = int(np.argmax(resid_acf[skip:])) + skip
win = slice("1900", "2000")
fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)
axes[0].plot(
    sun[win].index,
    sun[win],
    color=SECOND,
    lw=1.6,
    label="observed: 9–14 years, fast rise, slow fall",
)
axes[0].plot(
    res.seasonal[win].index,
    res.seasonal[win] + sun.mean(),
    color=ACCENT,
    lw=2.4,
    label=f"seasonal_decompose: one fixed {lag}-month shape, repeated",
)
axes[0].legend(loc="upper left")
axes[1].axhline(0, color=GREY_DIM, ls=":", lw=1.5)
axes[1].plot(res.resid[win].index, res.resid[win], color=ACCENT_LIGHT, lw=1.4)
axes[1].set_title(
    f"residual keeps {resid_var:.0%} of the variance and still peaks at lag {resid_lag / 12:.1f} years"
)
save(fig, "sun_decompose")

# fourier: spectrum + reconstruction with k components
y = sun["1900":].to_numpy(dtype=float)
n = len(y)
yf = fft(y - y.mean())
freqs = fftfreq(n, d=1 / 12)  # cycles per year
amp = np.abs(yf[: n // 2])
pos = freqs[: n // 2]


def reconstruct(k):
    keep = np.argsort(np.abs(yf[: n // 2]))[::-1][:k]
    mask = np.zeros(n, dtype=bool)
    mask[keep] = True
    mask[-keep] = True
    return np.real(ifft(np.where(mask, yf, 0))) + y.mean()


fig, axes = plt.subplots(
    1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.6]}
)
top = np.argsort(amp)[::-1][:9]
axes[0].vlines(pos[1:], 0, amp[1:], color=GREY_FAINT, lw=1.5)
axes[0].vlines(pos[top], 0, amp[top], color=ACCENT, lw=2.5)
axes[0].set_xlim(0, 0.6)
axes[0].set_xlabel("cycles per year")
axes[0].set_ylabel("amplitude")
axes[0].set_title("Spectrum — the k = 9 loudest frequencies marked")
peak_f = pos[top[0]]
axes[0].annotate(
    f"1 / {1 / peak_f:.1f} years",
    (peak_f, amp[top[0]]),
    xytext=(10, -4),
    textcoords="offset points",
    color=ACCENT_LIGHT,
    fontsize=14,
)
t = sun["1900":].index
axes[1].plot(t, y, color=GREY_DIM, lw=1.2, label="observed")
axes[1].plot(t, reconstruct(9), color=ACCENT, lw=2.6, label="k = 9 · the pattern")
axes[1].plot(
    t,
    reconstruct(200),
    color=SECOND,
    lw=1,
    ls="--",
    label="k = 200 · the noise, memorised",
)
axes[1].legend(loc="upper left")
axes[1].set_title("Rebuilt from k sines")
save(fig, "sun_fourier")

# a schematic: three sines and their sum
tt = np.linspace(0, 4, 800)
parts = [
    np.sin(2 * np.pi * tt),
    0.5 * np.sin(2 * np.pi * 3 * tt + 1),
    0.25 * np.sin(2 * np.pi * 7 * tt),
]
fig, axes = plt.subplots(4, 1, figsize=(8, 5.6), sharex=True)
for ax, part in zip(axes[:3], parts):
    ax.plot(tt, part, color=GREY_DIM, lw=2)
    ax.set_yticks([])
axes[3].plot(tt, sum(parts), color=ACCENT, lw=2.6)
axes[3].set_yticks([])
axes[3].set_title("their sum", fontsize=14)
for ax in axes:
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
save(fig, "fourier_sum")

print(
    f"\nsunspot period {lag} months, residual variance {resid_var:.0%}, residual peak {resid_lag} months"
)
for ch, (obs, null, ordn) in results.items():
    print(
        f"{ch}: lift {obs:.2f}x, p={(null >= obs).mean():.3f}, ordinary CV {ordn.std() / ordn.mean():.2f}"
    )
