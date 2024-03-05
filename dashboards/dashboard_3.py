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
    return df[select].dropna()


def main():
    st.title("Penguins Dashboard")

    if "penguins" not in st.session_state:
        st.session_state.penguins = load_penguins_dataset()

    species = st.selectbox(
        "Select a penguin species",
        options=st.session_state.penguins["Species"].unique(),
    )
    filtered_df = st.session_state.penguins[
        st.session_state.penguins["Species"] == species
    ]

    # Scatter plot
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Scatterplot for {species}")
        x_axis = st.selectbox(
            "Select the x-axis",
            filtered_df.columns,
            index=2,
        )
        y_axis = st.selectbox(
            "Select the y-axis",
            filtered_df.columns,
            index=3,
        )
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue="Sex", ax=ax1)
        st.pyplot(fig1)

    # Histogram
    with col2:
        st.subheader(f"Histogram of Flipper Lengths for {species}")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.histplot(filtered_df["Flipper Length (mm)"], ax=ax2, kde=True)
        st.pyplot(fig2)

    # Box plot
    st.subheader(f"Boxplot of Culmen Depth for {species}")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Sex", y="Culmen Depth (mm)", data=filtered_df, ax=ax3)
    st.pyplot(fig3)


if __name__ == "__main__":
    main()
