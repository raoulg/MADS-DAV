import pandas as pd
import streamlit as st
from mads_datasets import DatasetFactoryProvider, DatasetType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from loguru import logger


def load_penguins_dataset() -> pd.DataFrame:
    """_summary_
    allow_output_mutation=True argument is used because the
    penguinsdataset object is mutable, and we want to allow modifications to it.

    In addition to that, we want to cache the object, so that it is not reloaded
    every time the user interacts with the dashboard.
    """
    logger.info("Loading dataset")
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


def train_model(data):
    logger.info("Training model")
    X = data[
        [
            "Culmen Length (mm)",
            "Culmen Depth (mm)",
            "Flipper Length (mm)",
            "Body Mass (g)",
        ]
    ]
    y = data["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"columns: {X_train.columns}")

    model = RandomForestClassifier()
    model.fit(X_train.values, y_train)
    y_pred = model.predict(X_test.values)
    accuracy = accuracy_score(y_test, y_pred)
    logger.success(f"Trained model with accuracy {accuracy}")

    return {"model" : model , "accuracy" : accuracy}


def main():
    st.title("Penguins Species Prediction with Machine Learning")
    if "penguins" not in st.session_state:
        st.session_state.penguins = load_penguins_dataset()
    if "model" not in st.session_state:
        st.session_state.model = train_model(st.session_state.penguins)

    accuracy = st.session_state.model["accuracy"]

    st.write(f"Model Accuracy: {accuracy:.2f}")

    st.subheader("Predict Penguin Species")
    culmen_length = st.number_input(
        "Culmen Length (mm)", min_value=30.0, max_value=60.0, value=40.0, step=0.1
    )
    culmen_depth = st.number_input(
        "Culmen Depth (mm)", min_value=10.0, max_value=25.0, value=15.0, step=0.1
    )
    flipper_length = st.number_input(
        "Flipper Length (mm)", min_value=170.0, max_value=240.0, value=200.0, step=1.0
    )
    body_mass = st.number_input(
        "Body Mass (g)", min_value=2500, max_value=6500, value=4000, step=100
    )

    if st.button("Predict"):
        model = st.session_state.model["model"]
        prediction = model.predict(
            [[culmen_length, culmen_depth, flipper_length, body_mass]]
        )
        st.write(f"The predicted species is: {prediction[0]}")


if __name__ == "__main__":
    main()
