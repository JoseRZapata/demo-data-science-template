import os

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

# https://docs.streamlit.io/library/api-reference

# HOW TO RUN THE APP:
# streamlit run notebooks/7-deploy/titanic-streamlit-batch.py


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
        user_data["sex"] = st.radio(
            label="Sex:", options=["Woman", "Man"], horizontal=False
        )
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
    df["embarked"] = df["embarked"].map({
        "Cherbourg": "C",
        "Queenstown": "Q",
        "Southampton": "S",
    })

    return df


def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess batch data from a CSV file to match the format expected by the model.

    Args:
        df (pd.DataFrame): The original dataframe from CSV

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for prediction
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Convert sex values if needed
    if "sex" in processed_df.columns:
        processed_df["sex"] = processed_df["sex"].map(
            lambda x: "male"
            if x.lower() in ["man", "male", "m"]
            else "female"
            if x.lower() in ["woman", "female", "f"]
            else x
        )

    # Convert pclass values if needed
    if "pclass" in processed_df.columns:
        if processed_df["pclass"].dtype == "object":
            pclass_mapping = {
                "1st": 1,
                "2nd": 2,
                "3rd": 3,
                "first": 1,
                "second": 2,
                "third": 3,
                "1": 1,
                "2": 2,
                "3": 3,
            }
            processed_df["pclass"] = processed_df["pclass"].map(
                lambda x: pclass_mapping.get(str(x).lower(), x)
            )
            processed_df["pclass"] = pd.to_numeric(
                processed_df["pclass"], errors="coerce"
            )

    # Convert embarked values if needed
    if "embarked" in processed_df.columns:
        embarked_mapping = {
            "cherbourg": "C",
            "queenstown": "Q",
            "southampton": "S",
            "c": "C",
            "q": "Q",
            "s": "S",
        }
        processed_df["embarked"] = processed_df["embarked"].map(
            lambda x: embarked_mapping.get(str(x).lower(), x)
        )

    # Ensure required columns exist and have numeric types
    numeric_columns = ["age", "sibsp", "parch", "fare"]
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

    return processed_df


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


def individual_prediction_tab(model: Pipeline) -> None:
    """
    Display the individual prediction interface and make prediction for a single person.

    Args:
        model (Pipeline): The trained model
    """
    # get the data from the user
    df_user_data = get_user_data()

    # predict the outcome for the given user data
    state = model.predict(df_user_data)[0]

    emojis = ["😕", "😀"]

    st.write("")
    st.title(f"Chance to survive! {emojis[state]}")
    if state == 0:
        st.error("Bad news my friend, you will be food for sharks! 🦈")
    else:
        st.success("Congratulations! You can rest assured, you will be fine! 🤩")


def batch_prediction_tab(model: Pipeline) -> None:
    """
    Display the batch prediction interface and make predictions for a CSV file.

    Args:
        model (Pipeline): The trained model
    """
    st.subheader("Upload your CSV file with passenger data")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load and display the data
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Check if required columns exist
            required_cols = [
                "pclass",
                "sex",
                "age",
                "sibsp",
                "parch",
                "fare",
                "embarked",
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.warning(
                    f"Warning: Your data is missing these required columns: {', '.join(missing_cols)}"
                )
                st.info(
                    "Required columns: pclass, sex, age, sibsp, parch, fare, embarked"
                )
            else:
                # Add a button to make predictions
                if st.button("Predict Survival"):
                    with st.spinner("Processing data and making predictions..."):
                        # Make predictions
                        predictions = model.predict(df)

                        # Add predictions to the dataframe
                        result_df = df.copy()
                        result_df["Predicted_Survival"] = predictions
                        result_df["Survival_Status"] = result_df[
                            "Predicted_Survival"
                        ].map({0: "Did not survive", 1: "Survived"})

                        # Display results
                        st.success("Predictions completed!")
                        st.subheader("Prediction Results")
                        st.dataframe(result_df)

                        # Calculate survival rate
                        survival_rate = predictions.mean() * 100
                        st.metric("Overall Survival Rate", f"{survival_rate:.2f}%")

                        # Option to download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download results as CSV",
                            data=csv,
                            file_name="titanic_predictions.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            st.info("Please make sure your CSV file is properly formatted.")
    else:
        st.info("Please upload a CSV file with passenger information.")

        # Show sample format
        st.subheader("Sample CSV format:")
        sample_data = pd.DataFrame({
            "pclass": [1, 2, 3],
            "sex": ["female", "male", "female"],
            "age": [29, 35, 15],
            "sibsp": [0, 1, 0],
            "parch": [0, 0, 1],
            "fare": [211.3, 26.0, 7.75],
            "embarked": ["S", "C", "Q"],
        })
        st.dataframe(sample_data)


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

    # load the model
    model = load_model(model_file_path=project_path + "/models/" + model_name)

    # Create tabs for individual and batch prediction
    tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])

    with tab1:
        individual_prediction_tab(model)

    with tab2:
        batch_prediction_tab(model)


if __name__ == "__main__":
    main()
