"""A whole analysis loop as a script: config, process, model, and test the residual.

This is the shape leerdoelen 1.8/1.10 ask for and no notebook demonstrates on its own —
a notebook cell has no natural boundary, so nothing forces the analysis into functions
small enough to test, import, or run twice with different data. A script does.

The pipeline runs on public Dutch COVID figures rather than a showcase dataset on
purpose: the loop — config -> process -> compare -> model -> residual -> distribution
fit — is the transferable part, not the topic. The model is the one lesson 04.2 arrives
at: a straight line in positive tests whose ratio turns down along a logistic curve.

    uv run python scripts/covid_pipeline.py

Every step is also importable, which is what 05.3 does to walk through it inline:

    from scripts.covid_pipeline import preprocess, fit_model

    data = preprocess()
    data = fit_model(data)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from goad_toolkit.analytics import DistributionFitter, fit_table
from goad_toolkit.config import DataConfig, FileConfig
from goad_toolkit.dataprocessor import CovidDataProcessor
from goad_toolkit.models import linear_model, logistic, mse, train_model
from goad_toolkit.visualizer import (
    ComparePlot,
    ComparePlotDate,
    FitPlotSettings,
    PlotFits,
    PlotSettings,
    ResidualPlot,
)
from loguru import logger

RESULT_DIR = Path.home() / ".cache/goad/covid/result"
VACCINATION_START = "2021-01-06"


def preprocess() -> pd.DataFrame:
    """Download (if needed), clean, lag and z-score the Dutch COVID series."""
    logger.info("Preprocessing COVID data...")
    processed = CovidDataProcessor(FileConfig(), DataConfig()).process()
    logger.success(f"Processed {len(processed):,} days.")
    return processed


def save_fig(fig, name: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / name
    fig.savefig(path)
    logger.success(f"Saved {path}")


def plot_zscores(data: pd.DataFrame):
    """Deaths and positive tests on one scale: the shape a model has to match."""
    settings = PlotSettings(
        xlabel="date",
        ylabel="normalised values",
        title="Z-scores of deaths and positive tests",
    )
    fig, ax = ComparePlot(settings).plot(
        data=data, x="date", y1="deaths_shifted_zscore", y2="positivetests_zscore"
    )
    save_fig(fig, "zscores.png")
    return fig, ax


def covid_model(X: np.ndarray, params: list[float]) -> np.ndarray:  # noqa: N803
    """Deaths as a straight line in positive tests, times a logistic switch on the day.

    `X` has two columns: positive tests, and the day number. `params` is
    `[a, b, k, x0]`: the line's slope and intercept, and the switch's steepness
    and halfway day. With `k < 0` the ratio of deaths to tests runs at its old
    value, then decays as the switch turns.
    """
    a, b, k, x0 = params
    return linear_model(X[:, 0], [a, b]) * logistic(X[:, 1], k=k, x0=x0)


def fit_model(data: pd.DataFrame) -> pd.DataFrame:
    """Fit the turning-ratio model and keep its prediction and residual."""
    tests = data["positivetests"].to_numpy()
    day = np.arange(len(data)).astype(float)
    X = np.stack([tests, day], axis=1)  # noqa: N806
    y = data["deaths_shifted"].to_numpy()

    line = train_model(tests, y, linear_model, mse, [0.01, 1.0], bounds=[(0, 1.0), (0, None)])
    vaccination_day = float(np.argmax(data.index >= VACCINATION_START))
    initial = [line[0], line[1], -0.1, vaccination_day + 30]
    params = train_model(
        X, y, covid_model, mse, initial,
        bounds=[(0, 1.0), (0, None), (-1.0, 0), (0, len(data))],
    )
    halfway = data.index[int(round(params[3]))].date()
    logger.success(f"Fitted model: a={params[0]:.4f} b={params[1]:.1f} k={params[2]:.3f}, "
                   f"switch halfway on {halfway}")

    data = data.copy()
    data["predicted deaths"] = covid_model(X, params)
    data["residual"] = y - data["predicted deaths"]
    return data


def plot_model(data: pd.DataFrame):
    """Actual deaths against the model, with the vaccination start marked."""
    settings = PlotSettings(
        xlabel="date", ylabel="deaths", title="Deaths vs. the fitted model"
    )
    fig, ax = ComparePlotDate(settings).plot(
        data=data, x="date", y1="deaths_shifted", y2="predicted deaths",
        date=VACCINATION_START, datelabel="vaccination started",
    )
    save_fig(fig, "model.png")
    return fig, ax


def plot_residual(data: pd.DataFrame, title: str = "Residual"):
    """What the model did not explain, day by day."""
    settings = PlotSettings(figsize=(12, 6), title=title, xlabel="date", ylabel="error")
    fig, ax = ResidualPlot(settings).plot(
        data=data,
        x="date",
        y="residual",
        date=VACCINATION_START,
        datelabel="vaccination started",
        interval=1,
    )
    save_fig(fig, f"{title.lower().replace(' ', '_')}.png")
    return fig, ax


def plot_residual_distribution(data: pd.DataFrame):
    """Fit every continuous family to the residual: noise, or a shape the model lacks."""
    fitter = DistributionFitter(seed=42)
    fits = fitter.fit(data["residual"].to_numpy(), discrete=False)
    logger.success(f"Best fit: {fitter.best(fits)}")
    settings = PlotSettings(
        figsize=(12, 6),
        title="Residual distribution",
        xlabel="error",
        ylabel="probability",
    )
    fig = PlotFits(settings).plot(
        data=data["residual"].to_numpy(),
        fit_results=fits,
        fitplotsettings=FitPlotSettings(bins=30, max_fits=3),
    )
    save_fig(fig, "residual_distribution.png")
    return fig, fit_table(fits)


def main() -> None:
    data = preprocess()
    plot_zscores(data)
    data = fit_model(data)
    plot_model(data)
    plot_residual(data)
    plot_residual_distribution(data)
    logger.success("All done.")


if __name__ == "__main__":
    main()
