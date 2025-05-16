import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


class TitanicInput(BaseModel):
    age: int = Field(default=30, ge=0, le=100)
    pclass: int = Field(default=1, ge=1, le=3)
    sex: str = Field(default="male", pattern="^(male|female)$")
    sibsp: int = Field(default=0, ge=0, le=15)
    parch: int = Field(default=0, ge=0, le=15)
    fare: float = Field(default=30, ge=0, le=300)
    embarked: str = Field(default="C", pattern="^(C|Q|S)$")


class TitanicOutput(BaseModel):
    survived: int


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting survival on the Titanic",
    version="1.0",
)


@app.get("/")
def read_root() -> str:
    """
    Root endpoint for the Titanic Survival Prediction API.
    Returns:
        dict: A simple message indicating the API is running.
    """
    return "Welcome to the Titanic Survival Prediction API! more info at /docs"


@app.post("/predict")
def predict(input: TitanicInput) -> TitanicOutput:
    """
    Predict survival on the Titanic based on input features.
    Args:
        input (TitanicInput): Input features for prediction.
    Returns:
        TitanicOutput: Prediction result indicating survival (1) or not (0).
    """
    input_dict = {
        "age": [input.age],
        "pclass": [input.pclass],
        "sex": [input.sex],
        "sibsp": [input.sibsp],
        "parch": [input.parch],
        "fare": [input.fare],
        "embarked": [input.embarked],
    }

    input_df = pd.DataFrame.from_dict(input_dict)

    # Load the model
    model_name = "titanic_classification-random_forest-v1.joblib"
    model = joblib.load("models/" + model_name)

    # Make prediction
    prediction = model.predict(input_df)[0]
    return TitanicOutput(survived=prediction)
