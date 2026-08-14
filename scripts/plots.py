"""The plot classes lesson 2 derives, in a form later lessons and dashboards can import.

Lesson 2 writes both of these from scratch — that derivation is the exercise and it stays
in the notebook. This module is where the same code lives afterwards, so a dashboard can
render the chart without a copy of it drifting out of sync with the notebook's version.

    from scripts.plots import BarPlot

    fig, ax = BarPlot(settings).plot(data=frame, x="author", y="degree")

That import is the whole point of subclassing `BasePlot` rather than writing six lines of
`plt.subplots` in a cell: there is no way to import a chart that only exists as a cell.
`fig` is an ordinary matplotlib figure, so `st.pyplot(fig)` takes it as-is and the class
never learns that streamlit exists.
"""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from goad_toolkit.visualizer import BasePlot


class BarPlot(BasePlot):
    """A bar chart. All the styling lives in PlotSettings."""

    def build(self, data: pd.DataFrame, x: str, y: str, **kwargs):
        sns.barplot(data=data, x=x, y=y, ax=self.ax, **kwargs)
        return self.fig, self.ax


class BarPlotWithError(BasePlot):
    """A grouped bar chart carrying intervals that were computed elsewhere.

    `GroupedBarPlot` cannot do this: seaborn derives its intervals from raw
    observations, and these are already-summarised means with intervals of their own.

    Each bar is looked up by (category, hue level) rather than by position. Seaborn
    draws one container per hue level in `hue_order`, and the categories along the
    x-axis in tick order — so those two are what the lookup keys on, and the check
    turns a layout change into a failure rather than a silently misplaced error bar.
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: str,
        error: str,
        hue_order: list[str],
        **kwargs,
    ):
        sns.barplot(
            data=data, x=x, y=y, hue=hue, hue_order=hue_order, ax=self.ax, **kwargs
        )
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")

        intervals = data.set_index([x, hue])[error]
        categories = [label.get_text() for label in self.ax.get_xticklabels()]
        if len(self.ax.containers) != len(hue_order):
            raise ValueError(
                f"seaborn drew {len(self.ax.containers)} containers for "
                f"{len(hue_order)} hue levels; the error bars would be misplaced"
            )

        for container, level in zip(self.ax.containers, hue_order):
            for patch, category in zip(container, categories):
                self.ax.errorbar(
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_height(),
                    yerr=intervals[(category, level)],
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                )

        # Left to itself matplotlib puts this over the bars, which is the exact
        # thing 2.3 is about.
        self.ax.legend(loc="lower right", framealpha=1)
        return self.fig, self.ax
