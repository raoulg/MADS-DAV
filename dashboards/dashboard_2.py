import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from mads_datasets import DatasetFactoryProvider, DatasetType

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_penguins_dataset() -> pd.DataFrame:
    penguinsdataset = DatasetFactoryProvider.create_factory(DatasetType.PENGUINS)
    penguinsdataset.download_data()
    df = pd.read_parquet(penguinsdataset.filepath)
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
    return df[select].dropna()  # type: ignore


def main() -> None:
    if "penguins" not in st.session_state:
        st.session_state.penguins = load_penguins_dataset()

    st.title("Penguins Dashboard")

    species_filter: list = st.multiselect(
        "Select Species", options=st.session_state.penguins["Species"].unique()
    )
    if species_filter:
        st.session_state.penguins = st.session_state.penguins[
            st.session_state.penguins["Species"].isin(species_filter)
        ]

    plot_type = st.radio("Choose a Plot Type", ["Scatterplot", "Histogram", "Boxplot"])

    if plot_type == "Scatterplot":
        option1: str = st.selectbox(
            "Select the x-axis",
            st.session_state.penguins.columns,
            index=2,  # this will pick a default item
        )
        option2: str = st.selectbox(
            "Select the y-axis",
            st.session_state.penguins.columns,
            index=3,
        )
        color: str = st.selectbox(
            "Select the color", st.session_state.penguins.columns, index=0
        )

        fig, ax = plt.subplots()
        sns.scatterplot(data=st.session_state.penguins, x=option1, y=option2, hue=color)  # type: ignore
        st.pyplot(fig)

    elif plot_type == "Histogram":
        option: str = st.selectbox(
            "Select variable for histogram",
            st.session_state.penguins.columns,
            index=4,
        )
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.penguins[option], kde=True)  # type: ignore
        st.pyplot(fig)

    elif plot_type == "Boxplot":
        option = st.selectbox(
            "Select variable for boxplot",
            st.session_state.penguins.columns,
            index=4,
        )
        fig, ax = plt.subplots()
        sns.boxplot(x="Species", y=option, data=st.session_state.penguins)  # type: ignore
        st.pyplot(fig)


if __name__ == "__main__":
    main()
