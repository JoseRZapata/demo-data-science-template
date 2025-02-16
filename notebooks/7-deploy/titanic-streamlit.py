import os

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

# https://docs.streamlit.io/library/api-reference

# HOW TO RUN THE APP:
# streamlit run notebooks/7-deploy/titanic-streamlit.py


def get_user_data() -> pd.DataFrame:
    """
    Get the data provided by the user. Preprocess the data and create a
    DataFrame to feed the model and make the prediction.

    :return: preprocessed user information from the app
    """
    user_data = {}

    col_a, col_b = st.columns(2)
    with col_a:
        user_data["age"] = st.number_input(
            label="Age:", min_value=0, max_value=100, value=20, step=1
        )
        user_data["sibsp"] = st.slider(
            label="Number of siblings and spouses aboard:",
            min_value=0,
            max_value=15,
            value=3,
            step=1,
        )
    with col_b:
        user_data["fare"] = st.number_input(
            label="How much did your ticket cost you?:",
            min_value=0,
            max_value=300,
            value=80,
            step=1,
        )
        user_data["parch"] = st.slider(
            label="Number of parents and children aboard:",
            min_value=0,
            max_value=15,
            value=3,
            step=1,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        user_data["pclass"] = st.radio(
            label="Ticket class:", options=["1st", "2nd", "3rd"], horizontal=False
        )
    with col2:
        user_data["sex"] = st.radio(label="Sex:", options=["Woman", "Man"], horizontal=False)
    with col3:
        user_data["embarked"] = st.radio(
            label="Port of Embarkation:",  # hidden
            options=["Cherbourg", "Queenstown", "Southampton"],
            index=1,
        )

    df = pd.DataFrame.from_dict(user_data, orient="index").T

    # some preprocessing of the raw data from the user.
    # Follow the same data structure than in the Kaggle competition
    df["sex"] = df["sex"].map({"Man": "male", "Woman": "female"})
    df["pclass"] = df["pclass"].map({"1st": 1, "2nd": 2, "3rd": 3})
    df["embarked"] = df["embarked"].map(
        {
            "Cherbourg": "C",
            "Queenstown": "Q",
            "Southampton": "S",
        }
    )

    return df


@st.cache_resource
def load_model(model_file_path: str) -> Pipeline:
    """
    Loads a model in joblib format (.joblib extension) from the '/models' directory.

    Args:
        model_file_path (str): The path where the trained model is stored in pickle format.

    Returns:
        Pipeline: The trained model, a scikit-learn Pipeline object.
    """

    with st.spinner("Loading model..."):
        model = load(model_file_path)

    return model


def main() -> None:
    # choose the trained model you want to use to make predictions
    model_name = "titanic_classification-random_forest-v1.joblib"

    # get the project file name: "<your_project_path>/titanic_streamlit"
    this_file_path = os.path.abspath(__file__)
    project_path = "/".join(this_file_path.split("/")[:-3])

    # display an image of the Titanic
    st.image("notebooks/7-deploy/images/RMS_Titanic.jpg")

    # title
    st.header(body="Would you have survived the Titanic?🚢")

    # get the data from the user
    df_user_data = get_user_data()

    # load the model and predict the outcome for the given user data
    model = load_model(model_file_path=project_path + "/models/" + model_name)
    state = model.predict(df_user_data)[0]

    emojis = ["😕", "😀"]

    st.write("")
    st.title(f"chance to survive! {emojis[state]}")
    if state == 0:
        st.error("Bad news my friend, you will be food for sharks! 🦈")

    else:
        st.success("Congratulations! You can rest assured, you will be fine! 🤩")


if __name__ == "__main__":
    main()
