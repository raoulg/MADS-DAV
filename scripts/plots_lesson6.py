"""Figures for the lesson 6 slides, in the pttrn Paper palette.

Run from the repo root:  uv run --extra huggingface python scripts/plots_lesson6.py
Writes PNGs to presentations/img/lesson6/ and prints the numbers the slides quote.
Data: the penguins and ubuntu_irc showcases, MNIST (fetch_openml, cached in ~/.cache/mads-dav),
the Q posts that 06.2 downloads to ~/.cache/mads-dav/raw/qanon-posts.json, the flower vector
cache on the hub, and all-MiniLM-L6-v2 for the chat messages. Each computation repeats the
code and seeds of its notebook, so the numbers match 06.1–06.4.
"""

import json
import time
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import pipeline as hf_pipeline
from vectormesh import VectorCache

from scripts.pipelines import build_irc_pipeline
from scripts.sessionize import fit_session_threshold, merge_messages, sessionize
from wa_analyzer.data import load_showcase

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentations" / "img" / "lesson6"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".cache/mads-dav"

# pttrn tokens, Paper theme (~/code/packages/pttrn-huisstijl/tokens.json): petrol for the main
# series, vermilion for the one thing a figure is about, warm greys for everything else
BG = "#f4f1ec"
TEXT = "#111111"
ACCENT = "#1d5a73"
ACCENT_LIGHT = "#5e93a8"
SECOND = "#c43a14"
THIRD = "#8a6d3b"
GREY = "#2e2a26"
GREY_DIM = "#5a544d"
GREY_FAINT = "#cfc7bb"
INK_BG = "#141210"
INK_TEXT = "#f4f1ec"
INK_MUTED = "#b8b0a5"
INK_PETROL = "#86b9c9"
INK_ACCENT = "#f2643c"
# categories beyond three: the brand pair first, then muted earth tones that stay apart in lightness
CATEGORIES = [
    ACCENT,
    SECOND,
    THIRD,
    ACCENT_LIGHT,
    "#7a5a8c",
    "#6b7a3a",
    GREY,
    "#c9a66b",
    "#b77a6a",
    "#9c8f7d",
]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "paper_petrol", [BG, ACCENT_LIGHT, ACCENT, "#0d2c38"]
)
ALONG = LinearSegmentedColormap.from_list("along", [ACCENT, "#c9a66b", SECOND])
# cold to hot: petrol for low, near-white in the middle, vermilion for high
DIVERGING = LinearSegmentedColormap.from_list(
    "petrol_vermilion", [ACCENT, ACCENT_LIGHT, "#fdfcfa", "#e2481f", SECOND]
)

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


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GREY_FAINT)


def clean_3d(ax3: Axes3D) -> None:
    """No ticks, and the three panes of a 3D plot in the page colour with a faint edge."""
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks(
        []
    )  # ty: ignore[call-non-callable] -- the stubs see a wrapped axis method
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        # axis3d axes, which the stubs type as the 2D XAxis/YAxis without a pane
        axis.pane.set_facecolor(BG)  # ty: ignore[unresolved-attribute]
        axis.pane.set_edgecolor(GREY_FAINT)  # ty: ignore[unresolved-attribute]


def scatter_by(
    ax, coords, labels, order=None, colours=None, s=14, legend=True, alpha=0.75
):
    order = order if order is not None else list(pd.unique(labels))
    colours = colours or dict(zip(order, CATEGORIES))
    for label in order:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=s,
            color=colours[label],
            alpha=alpha,
            linewidths=0,
            label=str(label),
        )
    bare(ax)
    if legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            markerscale=2,
            handletextpad=0.3,
        )


# ================================================================ 6.1 PCA
rng = np.random.default_rng(42)
cloud = rng.multivariate_normal([0, 0], [[3.0, 2.2], [2.2, 2.0]], size=40)
cloud -= cloud.mean(axis=0)
direction = np.linalg.svd(cloud)[2][0]
along = cloud @ direction
feet = np.outer(along, direction)
reach = np.abs(along).max() * 1.1


def sketch(ax, dots, line, lost, kept):
    ax.plot(
        [-reach * direction[0], reach * direction[0]],
        [-reach * direction[1], reach * direction[1]],
        color=line,
        lw=2,
    )
    for point, foot in zip(cloud, feet):
        ax.plot([point[0], foot[0]], [point[1], foot[1]], color=lost, lw=1.3, alpha=0.9)
    ax.scatter(cloud[:, 0], cloud[:, 1], color=dots, s=46, zorder=3)
    ax.scatter(feet[:, 0], feet[:, 1], color=kept, s=18, zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


fig, ax = plt.subplots(figsize=(6, 5.4))
ax.set_facecolor(INK_BG)
sketch(ax, INK_PETROL, INK_TEXT, INK_ACCENT, INK_TEXT)
save(fig, "cover_sketch", facecolor=INK_BG)

fig, ax = plt.subplots(figsize=(7, 6.2))
sketch(ax, ACCENT_LIGHT, TEXT, SECOND, TEXT)
save(fig, "pca_line")

# the tilted sheet: three columns that are really two
sheet_rng = np.random.default_rng(4)
angles = sheet_rng.random(100) * 3 * np.pi / 2 - 0.5
sheet = np.empty((100, 3))
sheet[:, 0] = (
    np.cos(angles) + np.sin(angles) / 2 + 0.1 * sheet_rng.standard_normal(100) / 2
)
sheet[:, 1] = np.sin(angles) * 0.7 + 0.1 * sheet_rng.standard_normal(100) / 2
sheet[:, 2] = (
    sheet[:, 0] * 0.1 + sheet[:, 1] * 0.3 + 0.1 * sheet_rng.standard_normal(100)
)
centred = sheet - sheet.mean(axis=0)
_, s, Vt = np.linalg.svd(centred)
share = s**2 / np.sum(s**2)
print(
    f"sheet: variance per direction {np.round(share, 4)}, two carry {share[:2].sum():.1%}"
)

fig = plt.figure(figsize=(17, 5.2))
ax3 = cast(Axes3D, fig.add_subplot(1, 3, 1, projection="3d"))
ax3.scatter(centred[:, 0], centred[:, 1], centred[:, 2], color=ACCENT_LIGHT, s=16)
for i, colour in enumerate([SECOND, THIRD, GREY]):
    ax3.quiver(
        0,
        0,
        0,
        *(Vt[i] * s[i] / s[0]),
        color=colour,
        arrow_length_ratio=0.12,
        linewidths=3,
    )
clean_3d(ax3)
ax3.set_xlabel("column 1", labelpad=-8)
ax3.set_ylabel("column 2", labelpad=-8)
ax3.set_zlabel("column 3", labelpad=-8)
ax3.set_title("three columns, three new directions", pad=0)
ax = fig.add_subplot(1, 3, 2)
flat = centred @ Vt.T[:, :2]
ax.scatter(flat[:, 0], flat[:, 1], color=ACCENT_LIGHT, s=22)
ax.set_aspect("equal")
bare(ax)
ax.set_xlabel("first direction", color=SECOND)
ax.set_ylabel("second direction", color=THIRD)
ax.set_title("kept: the first two, seen flat on")
ax = fig.add_subplot(1, 3, 3)
ax.bar([1, 2, 3], share, color=[SECOND, THIRD, GREY], width=0.6)
ax.set_xticks([1, 2, 3], ["1", "2", "3"])
ax.set_ylim(0, 1)
ax.set_xlabel("direction")
ax.set_ylabel("share of the variance")
ax.set_title(f"scree plot: two carry {share[:2].sum():.0%}")
save(fig, "pca_sheet")

# penguins: the unit decides the first component
penguins = load_showcase("penguins").dropna()
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SHORT = ["bill length", "bill depth", "flipper", "body mass"]
measurements = penguins[FEATURES].to_numpy()
unscaled = PCA().fit(measurements)
kilos = measurements.copy()
kilos[:, 3] /= 1000
in_kilos = PCA().fit(kilos)
standardised = StandardScaler().fit_transform(measurements)
scaled = PCA().fit(standardised)
print("penguin sd:", penguins[FEATURES].std().round(1).to_dict())

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
for ax, model, unit in [
    (axes[0], unscaled, "body mass in grams"),
    (axes[1], in_kilos, "body mass in kilograms"),
    (axes[2], scaled, "every column standardised"),
]:
    loading = model.components_[0] * np.sign(
        model.components_[0][np.argmax(np.abs(model.components_[0]))]
    )
    colours = [SECOND if abs(v) == np.abs(loading).max() else ACCENT for v in loading]
    ax.barh(SHORT[::-1], loading[::-1], color=colours[::-1], height=0.6)
    ax.axvline(0, color=GREY_FAINT, lw=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(
        f"{unit}\nPC1 = {model.explained_variance_ratio_[0]:.2%} of the variance",
        fontsize=15,
    )
    ax.set_xlabel("weight in PC1")
    print(
        f"penguins {unit}: PC1 {model.explained_variance_ratio_[0]:.4f}, loadings {np.round(loading, 2)}"
    )
save(fig, "penguin_loadings")

SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
SPECIES_COLOURS = dict(zip(SPECIES, [ACCENT, SECOND, THIRD]))
species = penguins.species.to_numpy()


def separation(scores):
    frame = pd.DataFrame({"score": scores, "species": species})
    return (
        frame.groupby("species").score.mean().var()
        / frame.groupby("species").score.var().mean()
    )


fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
for ax, model, values, title in [
    (
        axes[0],
        unscaled,
        measurements,
        f"raw units: PC1 {unscaled.explained_variance_ratio_[0]:.2%}, "
        f"PC2 {unscaled.explained_variance_ratio_[1]:.2%}",
    ),
    (
        axes[1],
        scaled,
        standardised,
        f"standardised: PC1 + PC2 = {scaled.explained_variance_ratio_[:2].sum():.0%}",
    ),
]:
    scatter_by(
        ax,
        model.transform(values)[:, :2],
        species,
        SPECIES,
        SPECIES_COLOURS,
        s=26,
        legend=ax is axes[1],
    )
    ax.set_title(title)
    print(
        f"penguins {title}: separation along PC1 {separation(model.transform(values)[:, 0]):.2f}"
    )
save(fig, "penguins_scaled")

# ================================================================ 6.1 swiss roll and MNIST
roll, position = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
roll_pca = PCA(n_components=2).fit(roll)
roll_tsne = TSNE(n_components=2, random_state=42).fit_transform(roll)
print(f"swiss roll: PCA keeps {roll_pca.explained_variance_ratio_.sum():.0%}")

fig = plt.figure(figsize=(17, 5.4))
ax3 = cast(Axes3D, fig.add_subplot(1, 3, 1, projection="3d"))
ax3.scatter(roll[:, 0], roll[:, 1], roll[:, 2], c=position, cmap=ALONG, s=8)
ax3.view_init(10, -70)
clean_3d(ax3)
ax3.set_title("a sheet, rolled up in 3D", pad=0)
for k, (coords, title) in enumerate(
    [
        (
            roll_pca.transform(roll),
            f"PCA: a shadow, {roll_pca.explained_variance_ratio_.sum():.0%} of the variance",
        ),
        (roll_tsne, "t-SNE: neighbours kept, sheet unrolled"),
    ]
):
    ax = fig.add_subplot(1, 3, k + 2)
    ax.scatter(coords[:, 0], coords[:, 1], c=position, cmap=ALONG, s=8)
    bare(ax)
    ax.set_title(title)
save(fig, "swissroll")

mnist = fetch_openml("mnist_784", version=1, as_frame=False, data_home=str(CACHE))
digits, digit_labels = mnist["data"], mnist["target"].astype(np.uint8)
sample = np.random.default_rng(42).choice(len(digits), size=5000, replace=False)
subset, subset_labels = digits[sample], digit_labels[sample]

sevens = subset[subset_labels == 7][:12]
fig, axes = plt.subplots(1, 12, figsize=(14, 1.5))
for ax, image in zip(axes, sevens):
    ax.imshow(
        image.reshape(28, 28), cmap=LinearSegmentedColormap.from_list("ink", [BG, TEXT])
    )
    ax.set_axis_off()
save(fig, "sevens")

pca = PCA().fit(subset)
cumulative = np.cumsum(pca.explained_variance_ratio_)
print(
    f"mnist: two components {cumulative[1]:.1%}, 95% needs {int(np.argmax(cumulative >= 0.95)) + 1}"
)
mnist_tsne = TSNE(n_components=2, random_state=42, init="pca").fit_transform(subset)
DIGITS = list(range(10))
DIGIT_COLOURS = dict(zip(DIGITS, CATEGORIES))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
scatter_by(
    axes[0],
    pca.transform(subset)[:, :2],
    subset_labels,
    DIGITS,
    DIGIT_COLOURS,
    s=5,
    legend=False,
)
axes[0].set_title(f"PCA: two components, {cumulative[1]:.0%} of the variance")
scatter_by(axes[1], mnist_tsne, subset_labels, DIGITS, DIGIT_COLOURS, s=5)
axes[1].set_title("t-SNE, perplexity 30")
save(fig, "mnist")

PERPLEXITIES = (5, 30, 100, 200)
embeddings = {
    p: TSNE(n_components=2, perplexity=p, random_state=42, init="pca").fit_transform(
        subset
    )
    for p in PERPLEXITIES
}
pairs = np.triu_indices(10, k=1)
gaps = {}
fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
for ax, p in zip(axes, PERPLEXITIES):
    scatter_by(
        ax, embeddings[p], subset_labels, DIGITS, DIGIT_COLOURS, s=3, legend=False
    )
    sil = silhouette_score(embeddings[p], subset_labels)
    ax.set_title(f"perplexity {p}")
    frame = pd.DataFrame(embeddings[p], columns=pd.Index(["x", "y"])).assign(
        digit=subset_labels
    )
    centroids = frame.groupby("digit")[["x", "y"]].mean().to_numpy()
    distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    gaps[p] = distances[pairs] / distances[pairs].max()
    print(f"perplexity {p}: silhouette by digit {sil:.2f}")
save(fig, "perplexity")
names = [f"{a}-{b}" for a in range(10) for b in range(a + 1, 10)]
print(f"spearman gaps 5 vs 100: {spearmanr(gaps[5], gaps[100]).statistic:.2f}")
for p in (5, 200):
    order = np.argsort(gaps[p])
    print(
        f"perplexity {p}: closest {[names[i] for i in order[:3]]}, furthest {[names[i] for i in order[-3:]]}"
    )

noise = np.random.default_rng(0).normal(size=(1000, 50))
fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
for ax, p in zip(axes, (2, 5, 30, 100)):
    embedding = TSNE(
        n_components=2, perplexity=p, random_state=0, init="pca"
    ).fit_transform(noise)
    ax.scatter(
        embedding[:, 0], embedding[:, 1], color=GREY_DIM, s=5, alpha=0.7, linewidths=0
    )
    bare(ax)
    ax.set_title(f"perplexity {p}")
    score = silhouette_score(
        embedding, KMeans(5, n_init=10, random_state=0).fit_predict(embedding)
    )
    print(f"noise perplexity {p}: k-means silhouette on the picture {score:.3f}")
save(fig, "noise")
print(
    f"noise: k-means silhouette on the raw data "
    f"{silhouette_score(noise, KMeans(5, n_init=10, random_state=0).fit_predict(noise)):.3f}"
)

# ================================================================ 6.2 the Q posts
with (CACHE / "raw" / "qanon-posts.json").open(encoding="utf-8") as f:
    posts = pd.json_normalize(json.load(f)["posts"], sep="_")
posts["time"] = pd.to_datetime(posts["post_metadata_time"], unit="s")
posts["site"] = posts["post_metadata_source_site"]
posts = posts.sort_values("time").reset_index(drop=True)


def normalise(data, strip_urls=False, letters_only=False):
    text = data["text"].fillna("").astype(str).str.lower()
    if strip_urls:
        text = text.str.replace(r"https?://\S+", " ", regex=True)
    if letters_only:
        text = text.str.replace(r"[^a-z ]+", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    data = data.assign(text=text, n_chars=text.str.len())
    return data[data.n_chars > 50]


def chunk(frame):
    size = frame.n_chars.sum() // 100
    frame = frame.assign(chunk=frame.text.str.len().cumsum() // size)
    chunks = frame.groupby("chunk").agg(
        text=("text", " ".join), label=("site", lambda s: s.mode().iloc[0])
    )
    return chunks[chunks.text.str.len() > size / 2].reset_index(drop=True)


def own_silhouette(distance, inside):
    return float(
        silhouette_samples(distance, inside, metric="precomputed")[inside].mean()
    )


SITES = ["4ch", "8ch", "8kun"]
SITE_COLOURS = {"4ch": SECOND, "8ch": ACCENT, "8kun": THIRD}
passes = {}
for name, kwargs in [
    ("everything", {}),
    ("letters", {"strip_urls": True, "letters_only": True}),
]:
    chunks = chunk(normalise(posts, **kwargs))
    counts = (
        CountVectorizer(analyzer="char", ngram_range=(3, 3))
        .fit_transform(chunks.text)
        .toarray()
    )
    distance = manhattan_distances(counts, counts)
    labels = chunks.label.to_numpy()
    early = labels == "4ch"
    position = np.arange(len(distance))
    n = int(early.sum())
    others = np.array(
        [
            own_silhouette(distance, (position >= start) & (position < start + n))
            for start in range(n, len(distance) - n + 1)
        ]
    )
    passes[name] = {
        "coords": PCA(n_components=2).fit_transform(distance),
        "labels": labels,
        "mean": silhouette_score(
            distance, np.where(early, "4chan", "after"), metric="precomputed"
        ),
        "own": own_silhouette(distance, early),
        "others": others,
        "shape": counts.shape,
    }
    print(
        f"Q {name}: {counts.shape}, mean silhouette {passes[name]['mean']:.3f}, 4chan own "
        f"{passes[name]['own']:.3f}, later runs more distinct {(others > passes[name]['own']).sum()} of {len(others)}"
    )

fig, ax = plt.subplots(figsize=(8, 6.6))
scatter_by(
    ax,
    passes["everything"]["coords"],
    passes["everything"]["labels"],
    SITES,
    SITE_COLOURS,
    s=46,
    alpha=0.85,
)
ax.set_title("100 chunks of Q, by trigram counts")
save(fig, "qanon_sites")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
for ax, (name, title) in zip(
    axes,
    [
        ("everything", "everything, links and markup included"),
        ("letters", "letters and spaces only"),
    ],
):
    scatter_by(
        ax,
        passes[name]["coords"],
        passes[name]["labels"],
        SITES,
        SITE_COLOURS,
        s=46,
        alpha=0.85,
        legend=ax is axes[1],
    )
    ax.set_title(title)
save(fig, "qanon_twice")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
for ax, (name, title) in zip(
    axes, [("everything", "everything counted"), ("letters", "letters and spaces only")]
):
    result = passes[name]
    bins = np.linspace(
        min(result["others"].min(), result["own"]) - 0.01,
        max(result["others"].max(), result["own"]) + 0.01,
        24,
    )
    ax.hist(result["others"], bins=bins, color=GREY_FAINT, edgecolor=BG)
    ax.axvline(result["own"], color=SECOND, lw=3)
    ax.text(
        result["own"],
        ax.get_ylim()[1] * 0.92,
        "  the 4chan months",
        color=SECOND,
        va="top",
        ha="left" if name == "letters" else "right",
    )
    ax.set_title(title)
    ax.set_xlabel("own silhouette of a run of consecutive chunks")
axes[0].set_ylabel("number of later runs")
save(fig, "qanon_stretches")

# ================================================================ 6.3 the flower cache
tr = VectorCache.from_hub("pttrn-io/oxford-flowers-dinov2-small", split="train")
te = VectorCache.from_hub("pttrn-io/oxford-flowers-dinov2-small", split="test")
Xtr, ytr = tr["embed"][:], tr["label"][:]
Xte, yte = te["embed"][:], te["label"][:]

ENCODER_WEIGHTS = 22_000_000
encoder_cost = ENCODER_WEIGHTS * ((224 // 14) ** 2 + 1)
head_cost = Xtr.shape[1] * 102
EPOCHS = 100
in_loop = EPOCHS * len(tr) * encoder_cost
on_cache = len(tr) * encoder_cost + EPOCHS * len(tr) * head_cost
print(
    f"cost: encoder {encoder_cost:,} per photo, head {head_cost:,}, ratio {encoder_cost / head_cost:,.0f}; "
    f"100 epochs {in_loop:.1e} vs {on_cache:.1e}"
)

fig, ax = plt.subplots(figsize=(10.5, 3.6))
bars = [
    ("encoder in the loop,\n100 epochs", in_loop, GREY_DIM),
    ("encoder once, then\n100 epochs on the cache", on_cache, ACCENT),
]
ax.barh(
    [b[0] for b in bars][::-1],
    [b[1] for b in bars][::-1],
    color=[b[2] for b in bars][::-1],
    height=0.55,
)
ax.set_xscale("log")
ax.set_xlim(1e12, 1e16)
ax.set_xlabel("multiplications (log scale: each tick is ten times more)")
for y, (_, value, _) in enumerate(bars[::-1]):
    ax.text(
        value * 1.3, y, f"{value:.1e}".replace("e+", " × 10^"), va="center", color=GREY
    )
save(fig, "cost")

sample_classes = torch.randperm(102, generator=torch.Generator().manual_seed(0))[:8]
mask_te = torch.isin(yte, sample_classes)
flower_coords = PCA(n_components=2).fit_transform(Xte[mask_te].numpy())
flower_labels = np.array([f"class {i}" for i in yte[mask_te].numpy()])
fig, ax = plt.subplots(figsize=(8, 6.4))
scatter_by(
    ax,
    flower_coords,
    flower_labels,
    [f"class {i}" for i in sample_classes.tolist()],
    s=40,
    alpha=0.8,
)
ax.set_title("eight species, PCA of the encoder's vectors")
save(fig, "flowers_pca")

a, b, c = np.array([1.0, 2.0]), np.array([3.0, 6.0]), np.array([2.0, 1.0])
arrows = {"a": a, "b": b, "c": c}
tone = {"a": ACCENT, "b": ACCENT, "c": SECOND}


def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def arrow(ax, tip, name):
    ax.annotate(
        "",
        xy=tip,
        xytext=(0, 0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": tone[name],
            "lw": 2.6,
            "mutation_scale": 22,
        },
    )


fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6.2))
for name, v in arrows.items():
    arrow(left, v, name)
    left.text(v[0] + 0.15, v[1] + 0.1, name, fontsize=20, color=tone[name])
for u, v in [(a, b), (a, c)]:
    left.plot([u[0], v[0]], [u[1], v[1]], linestyle="--", color=GREY_DIM, lw=1.6)
    left.text(
        (u[0] + v[0]) / 2 + 0.18,
        (u[1] + v[1]) / 2,
        f"{np.linalg.norm(u - v):.1f}",
        color=GREY_DIM,
        fontsize=16,
    )
left.set(
    xlim=(0, 6.5),
    ylim=(0, 6.5),
    aspect="equal",
    xlabel='count of "install"',
    ylabel='count of "haha"',
)
left.set_title("Euclidean: the gap between the tips")
quarter = np.linspace(0, np.pi / 2, 100)
right.plot(np.cos(quarter), np.sin(quarter), color=GREY_FAINT)
for name, v in arrows.items():
    arrow(right, v / np.linalg.norm(v), name)
right.text(*(a / np.linalg.norm(a) * 1.07), "a, b", fontsize=20, color=ACCENT)
right.text(*(c / np.linalg.norm(c) * 1.07), "c", fontsize=20, color=SECOND)
angle_a, angle_c = np.arctan2(a[1], a[0]), np.arctan2(c[1], c[0])
between = np.linspace(angle_c, angle_a, 50)
right.plot(0.3 * np.cos(between), 0.3 * np.sin(between), color=TEXT, lw=1.6)
middle = (angle_a + angle_c) / 2
right.text(
    0.36 * np.cos(middle),
    0.36 * np.sin(middle),
    f"{np.degrees(angle_a - angle_c):.0f}°",
    fontsize=16,
)
right.text(
    0.5,
    0.1,
    f"cosine(a, b) = {cosine(a, b):.2f}\ncosine(a, c) = {cosine(a, c):.2f}",
    fontsize=16,
)
right.set(
    xlim=(0, 1.2),
    ylim=(0, 1.2),
    aspect="equal",
    xlabel='"install", scaled',
    ylabel='"haha", scaled',
)
right.set_title("cosine: only the angle, at length one")
save(fig, "cosine")

train_unit = torch.nn.functional.normalize(Xtr, dim=1)
test_unit = torch.nn.functional.normalize(Xte, dim=1)
similarity = test_unit @ train_unit.T
same_species = yte[:, None] == ytr[None, :]
print(
    f"cosine same species {similarity[same_species].mean():.2f}, other {similarity[~same_species].mean():.2f}; "
    f"1-NN {(ytr[similarity.argmax(dim=1)] == yte).float().mean():.1%}"
)
shown = sample_classes[:6]
rows = torch.cat([torch.where(yte == k)[0][:5] for k in shown])
cols = torch.cat([torch.where(ytr == k)[0][:10] for k in shown])
fig, ax = plt.subplots(figsize=(11, 5.6))
# split high from low: full petrol at 0.4 and below, white at 0.6, full vermilion at 0.8 and up
image = ax.imshow(
    similarity[rows][:, cols].numpy(),
    cmap=DIVERGING,
    norm=TwoSlopeNorm(vmin=0.4, vcenter=0.6, vmax=0.8),
    aspect="auto",
)
names6 = [f"class {k}" for k in shown.tolist()]
ax.set_xticks(np.arange(4.5, 60, 10), names6)
ax.set_yticks(np.arange(2, 30, 5), names6)
ax.tick_params(length=0)
for edge in np.arange(-0.5, 60, 10):
    ax.axvline(edge, color=BG, lw=1.5)
for edge in np.arange(-0.5, 30, 5):
    ax.axhline(edge, color=BG, lw=1.5)
ax.set_xlabel("training photos, ten per species")
ax.set_ylabel("test photos, five per species")
fig.colorbar(
    image,
    ax=ax,
    fraction=0.03,
    pad=0.02,
    label="cosine",
    extend="both",
    ticks=[0.4, 0.6, 0.8],
)
save(fig, "nn_matrix")

torch.manual_seed(0)
head = torch.nn.Linear(Xtr.shape[1], 102)
optimiser = torch.optim.Adam(head.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
start = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    for batch in torch.randperm(len(Xtr)).split(32):
        optimiser.zero_grad()
        loss_fn(head(Xtr[batch]), ytr[batch]).backward()
        optimiser.step()
with torch.no_grad():
    print(
        f"head: {EPOCHS} epochs in {time.perf_counter() - start:.1f}s, "
        f"test accuracy {(head(Xte).argmax(dim=1) == yte).float().mean():.1%}"
    )

# ================================================================ 6.4 chat messages
EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(texts):
    return EMBEDDER.encode(list(texts), batch_size=64, show_progress_bar=False)


irc = build_irc_pipeline().apply(load_showcase("ubuntu_irc"))
uk = irc[(irc.channel == "#ubuntu-uk") & (~irc.is_action)].copy()
uk["timestamp"] = (
    pd.to_datetime(uk["date"])
    + pd.to_timedelta(uk["hh"], unit="h")
    + pd.to_timedelta(uk["mm"], unit="m")
)
uk = uk.sort_values(["author", "timestamp"]).reset_index(drop=True)
TOP_AUTHORS = uk.author.value_counts().head(6).index.tolist()
AUTHOR_COLOURS = dict(zip(TOP_AUTHORS, CATEGORIES))
messages = uk[uk.author.isin(TOP_AUTHORS)].sample(n=1500, random_state=42)
n_words = messages.message.str.count(r"\S+")
print(
    f"messages: median {n_words.median():.0f} words, {(n_words <= 5).mean():.0%} five or fewer"
)
print("first ten:", messages.message.head(10).tolist())

X = embed(messages.message)
pca = PCA().fit(X)
coords = pca.transform(X)[:, :2]
cumulative = np.cumsum(pca.explained_variance_ratio_)
print(
    f"blob: silhouette by author {silhouette_score(X, messages.author):.3f}; two components {cumulative[1]:.1%}, "
    f"half needs {int(np.argmax(cumulative >= 0.5)) + 1}"
)
fig, ax = plt.subplots(figsize=(8, 6.4))
scatter_by(
    ax, coords, messages.author.to_numpy(), TOP_AUTHORS, AUTHOR_COLOURS, s=12, alpha=0.7
)
ax.set_title("1,500 messages, six authors, one row per message")
save(fig, "blob")

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color=ACCENT)
half = int(np.argmax(cumulative >= 0.5)) + 1
ax.scatter([2], [cumulative[1]], color=SECOND, s=60, zorder=3)
ax.annotate(
    f"2 components: {cumulative[1]:.1%}",
    (2, cumulative[1]),
    xytext=(30, 0.12),
    color=SECOND,
    arrowprops={"arrowstyle": "-", "color": SECOND},
)
ax.scatter([half], [cumulative[half - 1]], color=ACCENT, s=60, zorder=3)
ax.annotate(
    f"{half} components: half",
    (half, cumulative[half - 1]),
    xytext=(90, 0.42),
    color=ACCENT,
    arrowprops={"arrowstyle": "-", "color": ACCENT},
)
ax.set_xlim(0, 384)
ax.set_ylim(0, 1.02)
ax.set_xlabel("number of components kept, of 384")
ax.set_ylabel("share of the variance")
save(fig, "seven_percent")

shuffle_rng = np.random.default_rng(0)
shuffled = np.column_stack([shuffle_rng.permutation(column) for column in X.T])
shuffled_coords = PCA(n_components=2).fit_transform(shuffled)
groups = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(coords)
shuffled_groups = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(
    shuffled_coords
)
full_groups = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
print(
    f"k-means: picture {silhouette_score(coords, groups):.2f}, shuffled {silhouette_score(shuffled_coords, shuffled_groups):.2f}, "
    f"384 dims {silhouette_score(X, full_groups):.2f}"
)
contents = pd.DataFrame(
    {
        "group": groups,
        "author": messages.author.to_numpy(),
        "words": n_words.to_numpy(),
        "message": messages.message.to_numpy(),
    }
)
by_length = contents.groupby("group").words.median().sort_values()
print("k-means groups, median words:", by_length.to_dict())
print(pd.crosstab(contents.group, contents.author, normalize="index").round(2))
for g in by_length.index:
    print(
        f"group {g} examples:", contents[contents.group == g].message.head(6).tolist()
    )
GROUP_NAMES = dict(zip(by_length.index, ["short", "medium", "long"]))
GROUP_COLOURS = {"short": SECOND, "medium": THIRD, "long": ACCENT}
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
scatter_by(
    axes[0],
    coords,
    np.array([GROUP_NAMES[g] for g in groups]),
    ["short", "medium", "long"],
    GROUP_COLOURS,
    s=10,
    legend=False,
)
axes[0].set_title("the messages, three k-means groups")
neutral = dict(zip(range(3), [SECOND, THIRD, ACCENT]))
scatter_by(
    axes[1], shuffled_coords, shuffled_groups, [0, 1, 2], neutral, s=10, legend=False
)
axes[1].set_title("the same numbers, every column shuffled")
save(fig, "kmeans")

threshold = fit_session_threshold(uk["timestamp"], uk["author"])
uk["session_id"] = sessionize(uk, "timestamp", "author", threshold)
subset = uk[uk.author.isin(TOP_AUTHORS)].copy()
subset["window_id"] = subset.groupby("author").cumcount() // 15
subset["day"] = subset["timestamp"].dt.date
print(f"session threshold {threshold:.0f}s = {threshold / 60:.1f} min")

# one person's busy day, cut three ways
per_day = subset.groupby(["author", "day"]).agg(
    n=("message", "size"), sessions=("session_id", "nunique")
)
pick = per_day[(per_day.n.between(40, 70)) & (per_day.sessions.between(4, 7))].index[0]
day = subset[(subset.author == pick[0]) & (subset.day == pick[1])].sort_values(
    "timestamp"
)
hours = day.timestamp.dt.hour + day.timestamp.dt.minute / 60
fig, ax = plt.subplots(figsize=(17, 3.6))
rows = [
    ("the day", np.zeros(len(day), int)),
    ("fixed windows of 15", day.window_id.to_numpy()),
    ("sessions, gap over 15 min", day.session_id.to_numpy()),
]
for y, (label, ids) in enumerate(rows[::-1]):
    swap = pd.Series(ids).ne(pd.Series(ids).shift()).cumsum().to_numpy() % 2
    ax.scatter(
        hours,
        np.full(len(day), y),
        c=[ACCENT if k else SECOND for k in swap],
        s=500,
        marker="|",
        linewidths=3,
    )
ax.set_yticks(range(3), [r[0] for r in rows[::-1]])
ax.set_xlabel("time of day (one colour per unit; a colour change starts a new unit)")
ax.set_xlim(hours.min() - 0.3, hours.max() + 0.3)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, 2.6)
save(fig, "units")
print(f"units figure: {pick[0]} on {pick[1]}, {len(day)} messages")


def self_retrieval(
    frame, group_cols, min_messages=6, n_eval=400, seed=42, single=False
):
    sizes = frame.groupby(group_cols).size()
    keys = sizes[sizes >= min_messages].index.tolist()
    pick_rng = np.random.default_rng(seed)
    if len(keys) > n_eval:
        keys = [keys[i] for i in pick_rng.choice(len(keys), size=n_eval, replace=False)]
    first, second = [], []
    grouped = frame.sort_values("timestamp").groupby(group_cols)
    for key in keys:
        grp = grouped.get_group(key)
        half = len(grp) // 2
        if single:
            first.append(grp.message.iloc[0])
            second.append(grp.message.iloc[half])
        else:
            first.append(" ".join(grp.message.iloc[:half]))
            second.append(" ".join(grp.message.iloc[half:]))
    A, B = embed(first), embed(second)  # noqa: N806
    A = A / np.linalg.norm(A, axis=1, keepdims=True)  # noqa: N806
    B = B / np.linalg.norm(B, axis=1, keepdims=True)  # noqa: N806
    return float(((A @ B.T).argmax(axis=1) == np.arange(len(keys))).mean())


halves = {
    "single message": self_retrieval(subset, ["author", "session_id"], single=True),
    "per day": self_retrieval(subset, ["author", "day"]),
    "fixed window": self_retrieval(subset, ["author", "window_id"]),
    "session": self_retrieval(subset, ["author", "session_id"]),
}
print("halves:", {k: round(v, 3) for k, v in halves.items()})
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.barh(
    list(halves)[::-1],
    list(halves.values())[::-1],
    height=0.6,
    color=[ACCENT if k == "session" else GREY_FAINT for k in list(halves)[::-1]],
)
ax.axvline(1 / 400, color=SECOND, lw=2)
ax.text(1 / 400 + 0.004, 3.35, "chance: 1 in 400", color=SECOND)
for y, v in enumerate(list(halves.values())[::-1]):
    ax.text(v + 0.005, y, f"{v:.0%}", va="center", color=GREY)
ax.set_xlim(0, 0.36)
ax.set_xlabel("share of first halves that find their own second half")
save(fig, "halves")

session_units = merge_messages(subset, ["author", "session_id"])
units = session_units[session_units.n_messages >= 5].reset_index(drop=True)
vectors = embed(units.message)
normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
print(
    f"session units {len(units):,}, silhouette by author {silhouette_score(normed, units.author, metric='cosine'):.3f}"
)
centroids = pd.DataFrame(
    {
        author: normed[units.author.to_numpy() == author].mean(axis=0)
        for author in TOP_AUTHORS
    }
).T
centroids = centroids.div(np.linalg.norm(centroids, axis=1), axis=0)
author_similarity = centroids.to_numpy() @ centroids.to_numpy().T
fig, ax = plt.subplots(figsize=(7.6, 6.2))
image = ax.imshow(author_similarity, cmap=SEQUENTIAL, vmin=0.5, vmax=1)
ax.set_xticks(range(6), TOP_AUTHORS, rotation=40, ha="right")
ax.set_yticks(range(6), TOP_AUTHORS)
ax.tick_params(length=0)
for i in range(6):
    for j in range(6):
        ax.text(
            j,
            i,
            f"{author_similarity[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=13,
            color=BG if author_similarity[i, j] > 0.85 else TEXT,
        )
for side in ax.spines.values():
    side.set_visible(False)
save(fig, "centroids")
off = author_similarity[np.triu_indices(6, 1)]
print(f"centroid cosines between authors {off.min():.2f}-{off.max():.2f}")

shuffled_units = units.sample(frac=1, random_state=42)
balanced = shuffled_units.groupby("author").head(600)
train, test = train_test_split(
    balanced, test_size=0.3, stratify=balanced.author, random_state=42
)
trigrams = CountVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2)
by_style = LogisticRegression(max_iter=1000).fit(
    trigrams.fit_transform(train.message), train.author
)
by_meaning = LogisticRegression(max_iter=1000).fit(embed(train.message), train.author)
scores = {
    "guess the most\ncommon author": float(
        test.author.value_counts(normalize=True).max()
    ),
    "sentence\nembedding": float(by_meaning.score(embed(test.message), test.author)),
    "trigram\ncounts": float(
        by_style.score(trigrams.transform(test.message), test.author)
    ),
}
print("race:", {k.replace(chr(10), " "): round(v, 3) for k, v in scores.items()})
vocabulary = trigrams.get_feature_names_out()
for i, author in enumerate(by_style.classes_):
    print(
        f"  {author:12s}",
        [vocabulary[j] for j in np.argsort(by_style.coef_[i])[::-1][:8]],
    )
fig, ax = plt.subplots(figsize=(10, 4.0))
ax.barh(
    list(scores)[::-1],
    list(scores.values())[::-1],
    height=0.6,
    color=[ACCENT, ACCENT_LIGHT, GREY_FAINT],
)
for y, v in enumerate(list(scores.values())[::-1]):
    ax.text(v + 0.01, y, f"{v:.0%}", va="center", color=GREY)
ax.set_xlim(0, 1)
ax.set_xlabel(
    "sessions whose author is named correctly, on sessions the model never saw"
)
save(fig, "race")

sentiment_model = hf_pipeline(  # ty: ignore[no-matching-overload]
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=None,
    truncation=True,
    max_length=512,
)
SMILE = r"(?:[:;=]-?[)\]D]|☺|☻)"
FROWN = r"(?:[:;=]-?[(\[]|☹)"
has_smile, has_frown = uk.message.str.contains(SMILE), uk.message.str.contains(FROWN)
words = (
    uk.message.str.replace(f"{SMILE}|{FROWN}", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
typed = pd.Series(
    np.select(
        [has_smile & ~has_frown, has_frown & ~has_smile, ~has_smile & ~has_frown],
        ["typed :)", "typed :(", "no smiley"],
        "both",
    ),
    index=uk.index,
)
print("smiley messages:", typed.value_counts().to_dict())
usable = (typed != "both") & (words.str.count(r"\S+") >= 4)
picked = (
    pd.DataFrame({"typed": typed[usable], "words": words[usable]})
    .sample(frac=1, random_state=0)
    .groupby("typed")
    .head(600)
)
results = sentiment_model(picked.words.tolist(), batch_size=32)
model_scores = pd.DataFrame(
    [{item["label"]: item["score"] for item in result} for result in results],
    index=picked.index,
)
picked["model says"] = model_scores[["positive", "neutral", "negative"]].idxmax(axis=1)
agreement = pd.crosstab(picked.typed, picked["model says"], normalize="index")
print(agreement.round(2))
ORDER = ["typed :(", "no smiley", "typed :)"]
fig, ax = plt.subplots(figsize=(10, 4.6))
width = 0.26
for k, (label, colour) in enumerate(
    [("negative", SECOND), ("neutral", GREY_FAINT), ("positive", ACCENT)]
):
    values = agreement.loc[ORDER, label].to_numpy()
    ax.bar(
        np.arange(3) + (k - 1) * width,
        values,
        width=width * 0.92,
        color=colour,
        label=f"model says {label}",
    )
ax.set_xticks(range(3), ["writer typed :(", "no smiley", "writer typed :)"])
ax.set_ylim(0, 0.7)
ax.set_ylabel("share of messages")
ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.12))
save(fig, "sentiment")

# ---------- formulas: typeset with mathtext, one line each, at 2x for the slides ----------
FORMULAS = {
    "formula_cosine": r"$\cos(u, v) \;=\; \dfrac{u \cdot v}{|u|\;|v|}$",
    "formula_dot": r"$u \cdot v \;=\; u_1 v_1 + u_2 v_2 + \cdots + u_{384}\, v_{384}$",
}
with plt.rc_context({"mathtext.fontset": "cm"}):
    for name, tex in FORMULAS.items():
        fig = plt.figure(figsize=(1, 1))
        fig.text(0, 0, tex, fontsize=30, color=TEXT)
        fig.savefig(
            OUT / f"{name}.png",
            dpi=144,
            bbox_inches="tight",
            pad_inches=0.04,
            transparent=True,
        )
        plt.close(fig)
        print("wrote", name)
