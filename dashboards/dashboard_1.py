import warnings

import matplotlib.pyplot as plt  # noqa: INP001
import pandas as pd
import seaborn as sns
import streamlit as st
from mads_datasets import DatasetFactoryProvider, DatasetType

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_penguins_dataset() -> pd.DataFrame:
    """_summary_
    allow_output_mutation=True argument is used because the
    penguinsdataset object is mutable, and we want to allow modifications to it.

    In addition to that, we want to cache the object, so that it is not reloaded
    every time the user interacts with the dashboard.
    """
    penguinsdataset = DatasetFactoryProvider.create_factory(DatasetType.PENGUINS)
    penguinsdataset.download_data()
    df = pd.read_parquet(penguinsdataset.filepath)  # noqa: PD901
    select = [
        "Species",
        "Island",
        "Culmen Length (mm)",
        "Culmen Depth (mm)",
        "Flipper Length (mm)",
        "Delta 15 N (o/oo)",
        "Delta 13 C (o/oo)",
        "Sex",
        "Body Mass (g)",
    ]

    return df[select].dropna()


def main() -> None:
    if "penguins" not in st.session_state:
        st.session_state.penguins = load_penguins_dataset()

    option1 = st.selectbox(
        "Select the x-axis",
        st.session_state.penguins.columns,
        index=2,
    )
    option2 = st.selectbox(
        "Select the y-axis",
        st.session_state.penguins.columns,
        index=3,
    )
    color = st.selectbox("Select the color", st.session_state.penguins.columns, index=0)

    fig, ax = plt.subplots()

    sns.scatterplot(data=st.session_state.penguins, x=option1, y=option2, hue=color)

    st.pyplot(fig)


if __name__ == "__main__":
    main()
