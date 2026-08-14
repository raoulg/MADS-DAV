"""The lesson-5 story, as a picture: how people type is more identifying than what they say.

Evidence for the lesson-5 design (Linear PTT-45). The notebook rebuilds these four panels
step by step; this script exists so the finding is reproducible rather than a screenshot.

    uv run scripts/fingerprint_story.py
"""

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT / "data/showcase/ubuntu_irc_days.parquet")
PAT = re.compile(r"^\[(\d{2}):(\d{2})\]\s+<(\S+)>\s+(.*)$")
rows = []
for r in df.itertuples():
    for ln in r.text.split("\n"):  # ty: ignore[unresolved-attribute]
        m = PAT.match(ln)
        if m:
            hh, mm, a, msg = m.groups()
            rows.append(
                (
                    r.created,  # ty: ignore[unresolved-attribute]
                    r.channel,  # ty: ignore[unresolved-attribute]
                    int(hh),
                    a,
                    msg,
                )
            )
columns = pd.Index(["date", "channel", "hh", "author", "message"])
msgs = pd.DataFrame(rows, columns=columns)
uk = msgs[msgs.channel == "#ubuntu-uk"].copy()
top = uk.author.value_counts().head(8).index.tolist()
d = uk[uk.author.isin(top)].copy()
d["year"] = d.date.dt.year
w = d.message

# emoticon dialects: nosed :-)  vs noseless :)  vs unicode ☺
d["nosed"] = w.str.count(r"[:;=]-[)DPp(\]]")
d["noseless"] = w.str.count(r"[:;=][)DPp(\]]|\bXD\b|\bO_O\b|\bo0\b|\b0o\b")
d["unicode_smiley"] = w.str.count(r"[☺☻☹㋛]")
d["apos_drop"] = w.str.count(
    r"\b(?:dont|doesnt|didnt|cant|wont|isnt|im|thats|its|ive|youre)\b", re.I
)
d["apos_keep"] = w.str.count(
    r"\b(?:don't|doesn't|didn't|can't|won't|isn't|i'm|that's|it's|i've|you're)\b"
)
d["starts_upper"] = w.str.match(r"^[A-Z]").astype(float)
d["addresses"] = w.str.match(r"^\S+[:,]\s").astype(float)

g = d.groupby("author")
emo = g[["nosed", "noseless", "unicode_smiley"]].sum()
emo_share = emo.div(emo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
order = emo_share.sort_values("nosed").index

fig, axes = plt.subplots(2, 2, figsize=(17, 10))

# --- 1. emoticon dialect
ax = axes[0][0]
bottom = np.zeros(len(order))
for col, colour, lab in [
    ("nosed", "#4c72b0", "nosed   :-)"),
    ("noseless", "#dd8452", "noseless  :)"),
    ("unicode_smiley", "#55a868", "unicode  ☺"),
]:
    vals = emo_share.loc[order, col].values * 100
    ax.barh(order, vals, left=bottom, color=colour, label=lab)
    bottom += vals
ax.set_xlabel("% of that author's smileys")
ax.set_title("Three dialects for the same thing: a smile", fontsize=13, weight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 100)

# --- 2. apostrophes
ax = axes[0][1]
rate = (g.apos_drop.sum() / (g.apos_drop.sum() + g.apos_keep.sum())).sort_values()
colours = ["#c44e52" if v > 0.3 else "#bbbbbb" for v in rate.values]
ax.barh(rate.index, rate.values * 100, color=colours)
ax.set_xlabel("% of contractions written without the apostrophe  (dont / im / thats)")
ax.set_title("Who bothers with apostrophes", fontsize=13, weight="bold")
for i, v in enumerate(rate.values):
    ax.text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=9)
ax.set_xlim(0, 55)

# --- 3. hour profile: night owls vs never-at-night
ax = axes[1][0]
hp = d.groupby(["author", "hh"]).size().unstack(fill_value=0)
hp = hp.div(hp.sum(axis=1), axis=0) * 100
for a in top:
    night = hp.loc[a, 0:5].sum()
    if a == "daftykins":
        ax.plot(
            hp.columns,
            hp.loc[a],
            lw=2.5,
            color="#c44e52",
            label=f"{a} ({night:.0f}% at 00-05h)",
            zorder=3,
        )
    elif a == "foobarry":
        ax.plot(
            hp.columns,
            hp.loc[a],
            lw=2.5,
            color="#4c72b0",
            label=f"{a} ({night:.0f}% at 00-05h)",
            zorder=3,
        )
    else:
        ax.plot(hp.columns, hp.loc[a], lw=1, color="#cccccc", zorder=1)
ax.axvspan(0, 5, color="#f0f0f0", zorder=0)
ax.set_xlabel("hour (UTC)")
ax.set_ylabel("% of that author's messages")
ax.set_title("Same channel, opposite clocks", fontsize=13, weight="bold")
ax.legend(fontsize=9)
ax.set_xticks(range(0, 24, 3))

# --- 4. stability early vs late
ax = axes[1][1]
early, late = d[d.year <= 2014], d[d.year >= 2016]
metrics = {
    "starts with a capital": "starts_upper",
    "addresses by nick": "addresses",
}
markers = {"starts with a capital": "o", "addresses by nick": "s"}
for lab, col in metrics.items():
    a = early.groupby("author")[col].mean()
    b = late.groupby("author")[col].mean()
    both = pd.concat([a.rename("e"), b.rename("l")], axis=1).dropna()
    r = both.e.corr(both.l, method="spearman")
    ax.scatter(
        both.e * 100,
        both.l * 100,
        s=90,
        marker=markers[lab],
        label=f"{lab}  (spearman {r:.2f})",
    )
lim = [0, 80]
ax.plot(lim, lim, ls="--", color="grey", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("2013-2014  (% of messages)")
ax.set_ylabel("2016-2017  (% of messages)")
ax.set_title("The habits do not drift over four years", fontsize=13, weight="bold")
ax.legend(fontsize=9, loc="upper left")

for ax in axes.flat:
    ax.grid(alpha=0.25)
plt.suptitle(
    "What separates people is not what they talk about — it is how they type",
    fontsize=15,
    weight="bold",
)
plt.tight_layout(rect=(0, 0, 1, 0.96))
out = ROOT / "img/fingerprint_story.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=130)
print(f"wrote {out}")

print("\nemoticon dialect shares (%):")
print((emo_share * 100).round(1).to_string())
print("\nquestion rate stability (the one that does NOT hold):")
qa = early.assign(q=early.message.str.contains(r"\?")).groupby("author").q.mean()
qb = late.assign(q=late.message.str.contains(r"\?")).groupby("author").q.mean()
print(f"  spearman = {qa.corr(qb, method='spearman'):.2f}")
