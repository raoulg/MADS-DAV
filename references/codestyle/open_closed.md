# Code Walkthrough: Demonstrating the Open-Closed Principle (OCP)

To understand how open-closed works, I have created a file in the src folder. Please have a look at `src/wa_analyzer/open_closed.py`. This markdown file describes and explains all of the code in that file. You can test the file for yourself by activating the `.venv` and running `python src/wa_analyzer/open_closed.py`. Have a look at the different images that are created in the `img` folder.

In the file, I demonstrate how the **Open-Closed Principle** (OCP) is applied. This principle states that software entities (classes, functions, etc.) should be open for extension but closed for modification. We'll walk through the code, explaining how it adheres to this principle.

---

### Imports

I start by importing essential libraries:

```python
from abc import ABC, abstractmethod
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
```

- `ABC` and `abstractmethod` allow us to define abstract base classes. These are used when we want to define a blueprint for other classes to follow but leave the implementation details to those classes.
- `Path` from `pathlib` helps manage file paths in a more robust way.
- `matplotlib`, `seaborn`, and `pandas` are used for plotting and handling data.
- `loguru` is a logging tool for reporting the progress of the script.

---

### Base Plotting Class

The `BasePlot` class serves as the foundation for all plot-related functionality. It takes in data and initializes the `fig` and `ax` objects.

```python
class BasePlot:
    def __init__(self, data):
        self.data = data
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
```

- **fig and ax**: `fig` is the figure object (the entire plot), and `ax` is the axes object (the part of the figure where data is plotted). These are crucial to understanding how `matplotlib` operates, and in this case, we will explicitly pass the `ax` to Seaborn functions. This reduces confusion since Seaborn automagically uses the active `ax` object if it's not explicitly passed. We do this to ensure we have control over where the plot is drawn.
  
```python
    def plot(self, title: str):
        sns.scatterplot(data=self.data, x="x", y="y", ax=self.ax)
        self.ax.set_title(title)
```
- **Plot method**: Here, we create a scatter plot using Seaborn. As noted before, passing the ax object is not required by Seaborn but is a good practice as it ensures that the plot is drawn on the right axes (which we defined in the constructor). If `ax` is omitted, Seaborn would plot on the default axes, which might lead to confusion.

```python
    def save(self, filename):
        self.fig.savefig(filename)
        plt.close(self.fig)  # Close the figure to free up memory
```

- **Save method**: This method saves the figure to a file. We close the figure after saving it to free up memory, especially useful when dealing with many plots.

---

### Abstract Base Class for Annotations

Now, we introduce the concept of **annotations**. Annotations allow us to extend the functionality of the plot without changing the base class. This is where we see the **Open-Closed Principle** in action.

```python
class Annotation(ABC):
    @abstractmethod
    def annotate(self, ax, data):
        pass
```

- The `Annotation` class is an abstract base class (ABC). This means it provides a common interface for all annotation types but doesn't implement any functionality itself. The `annotate` method is defined as an abstract method, which means any subclass must provide its own implementation.
- **Why use ABC?**: ABCs are useful when you want to define common behavior for subclasses but leave the actual implementation up to each subclass. If you know that multiple subclasses will share a method (like `annotate`), but the details will vary, this is an ideal solution. If your system requires more flexibility, an ABC might not be necessary.

---

### Concrete Annotations

Now we create several concrete implementations of the `Annotation` class, each with a different way of annotating the plot.

```python
class TrendlineAnnotation(Annotation):
    def annotate(self, ax, data):
        sns.regplot(data=data, x="x", y="y", ax=ax, scatter=False, color="red")
```

- **TrendlineAnnotation**: This class adds a trendline to the plot. It uses `regplot` from Seaborn to draw a regression line without scatter points (`scatter=False`).
- **NOTE** you might also decide to abstract away the hardcoded variable names and the color. See more details about this in [never hardcode](never_hardcode.md), [encapsulation](encapsulation.md) and [pydantic](pydantic.md). In the rest of this file I will ignore the hardcoding, the avoid explaining multiple principles at the same time, but keep in mind that every time you hardcode something, a kitten dies.


```python
class MaxPointAnnotation(Annotation):
    def annotate(self, ax, data):
        max_point = data.loc[data["y"].idxmax()]
        ax.annotate(
            f"Max: ({max_point.x}, {max_point.y})",
            xy=(max_point.x, max_point.y),
            xytext=(5, 5),
            textcoords="offset points",
        )
```

- **MaxPointAnnotation**: This class finds the maximum `y` value in the dataset and adds an annotation at that point. It uses `ax.annotate` to add the label to the plot.

```python
class MeanLineAnnotation(Annotation):
    def annotate(self, ax, data):
        mean_y = data["y"].mean()
        ax.axhline(mean_y, color="green", linestyle="--")
        ax.annotate(
            f"Mean: {mean_y:.2f}",
            xy=(0, mean_y),
            xytext=(5, 5),
            textcoords="offset points",
        )
```

- **MeanLineAnnotation**: This class adds a horizontal line representing the mean `y` value. The line is labeled with the value of the mean.

---

### Extending the Plot Class for Annotations

The `AnnotatedPlot` class extends the `BasePlot` class. It allows us to add multiple annotations to a single plot.

```python
class AnnotatedPlot(BasePlot):
    def __init__(self, data):
        super().__init__(data)
        self.annotations: list[Annotation] = []
```

- **Constructor**: We call `super().__init__(data)` to ensure that `AnnotatedPlot` inherits all the setup from `BasePlot`. This will also store the data parameter in the `BasePlot` and setup the fig and ax (see the first two lines of `BasePlot`). Then, we initialize an empty list to store the annotations.
  
```python
def add_annotation(self, annotation: Annotation):
        self.annotations.append(annotation)
```

- **Add Annotation**: This method allows new annotations to be added to the plot. We store each annotation in the `annotations` list. This method is actually where the magic happens: every time we add an annotation, it is stored in the list of annotations. This ensures our class is indeed open for extension (quite literal by adding an annotation to the list) while we dont need to bother about the code that is already there (which is in this way indeed closed for modification). 

```python
    def plot(self, title: str):
        super().plot(title)
        for annotation in self.annotations:
            annotation.annotate(self.ax, self.data)
```

- **Plot method**: We first call the `plot` method of the `BasePlot` class using `super()`. This makes sure the title is added, and the base scatterplot is created. After that, we can start extending by looping through all the annotations and apply each one to the plot. This shows how we can extend the functionality of the plot without modifying the `BasePlot` class, adhering to the **Open-Closed Principle**.
  
    - **Why use `super().plot()`?**: By calling `super()`, we ensure that the plotting logic from the base class is executed before we apply any annotations. This promotes reuse and avoids code duplication.

---

### Main Execution

Finally, the main block of the script demonstrates how to use the `AnnotatedPlot` class with different annotations.

- **img_folder**: We create a folder to store the plots if it doesn't already exist.
- **AnnotatedPlot**: We instantiate `AnnotatedPlot` and add different annotations in steps, showing how we can easily extend the functionality without modifying the existing code. This is a nice illustration of the **Open-Closed Principle** in action.

