from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

# Sample data
data = pd.DataFrame({"x": range(1, 11), "y": [2, 4, 3, 7, 5, 9, 8, 11, 10, 12]})


class BasePlot:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def plot(self, title: str):
        sns.scatterplot(data=self.data, x="x", y="y", ax=self.ax)
        self.ax.set_title(title)

    def save(self, filename: Path):
        logger.info(f"Saving file to {filename}")
        self.fig.savefig(filename)
        plt.close(self.fig)  # Close the figure to free up memory


class Annotation(ABC):
    @abstractmethod
    def annotate(self, ax, data):
        pass


class TrendlineAnnotation(Annotation):
    def annotate(self, ax, data):
        sns.regplot(data=data, x="x", y="y", ax=ax, scatter=False, color="red")


class MaxPointAnnotation(Annotation):
    def annotate(self, ax, data):
        max_point = data.loc[data["y"].idxmax()]
        ax.annotate(
            f"Max: ({max_point.x}, {max_point.y})",
            xy=(max_point.x, max_point.y),
            xytext=(5, 5),
            textcoords="offset points",
        )


class MeanLineAnnotation(Annotation):
    def annotate(self, ax, data):
        mean_y: float = data["y"].mean()
        ax.axhline(mean_y, color="green", linestyle="--")
        ax.annotate(
            f"Mean: {mean_y:.2f}",
            xy=(0, mean_y),
            xytext=(5, 5),
            textcoords="offset points",
        )


class AnnotatedPlot(BasePlot):
    def __init__(self, data):
        super().__init__(data)
        self.annotations: list[Annotation] = []

    def add_annotation(self, annotation: Annotation):
        self.annotations.append(annotation)

    def plot(self, title: str):
        super().plot(title)
        for annotation in self.annotations:
            annotation.annotate(self.ax, self.data)


if __name__ == "__main__":
    img_folder = Path("img/example")
    if not img_folder.exists():
        img_folder.mkdir(parents=True)
    plot = AnnotatedPlot(data)
    plot.plot(title="Base Plot with Annotations")
    plot.save(img_folder / "base_plot.png")
    logger.info("Base plot saved")

    plot.add_annotation(TrendlineAnnotation())
    plot.add_annotation(MaxPointAnnotation())
    plot.plot(title="Plot with Trendline and Max Point")
    plot.save(img_folder / "plot_with_trendline_and_max.png")
    logger.info("Plot with trendline and max point saved")

    # Extend with a new annotation without modifying existing code
    plot.add_annotation(MeanLineAnnotation())
    logger.info("Added annotations")

    plot.plot(title="Plot with All Annotations")
    plot.save(img_folder / "plot_with_all_annotations.png")

    logger.success("Plots have been saved")
