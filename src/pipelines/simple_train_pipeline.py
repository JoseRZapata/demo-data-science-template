# Train Model pipeline
#
# ## By:
# [Jose R. Zapata](https://joserzapata.github.io/)
#
# ## Date:
# 2025-03-14
#
# ## Description:
#
# Set all the code for the first model selected
#
# <https://github.com/JoseRZapata/demo-data-science-template/blob/main/notebooks/5-models/03-jrz-first_model-2025_02_27.ipynb>

# Import  libraries
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load data

URL_DATA = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
dataset = pd.read_csv(URL_DATA, low_memory=False, na_values="?")

# Data preparation

selected_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "survived",
]

dataset_features = dataset[selected_features]
# Convert data types

# numerical columns
cols_numeric_float = ["age", "fare"]
cols_numeric_int = ["sibsp", "parch"]
cols_numeric = cols_numeric_float + cols_numeric_int

# categorical columns
cols_categoric = ["sex", "embarked"]
cols_categoric_ord = ["pclass"]

# Categorical variables
dataset[cols_categoric] = dataset[cols_categoric].astype("category")

dataset["pclass"] = pd.Categorical(
    dataset["pclass"], categories=[3, 2, 1], ordered=True
)

# Numerical variables

dataset[cols_numeric_float] = dataset[cols_numeric_float].astype("float")
dataset[cols_numeric_int] = dataset[cols_numeric_int].astype("int8")

# Target variables

target = "survived"

dataset[target] = dataset[target].astype("int8")

# Data preprocessing
dataset = dataset.drop_duplicates()


# Train / Test split

X_features = dataset.drop(target, axis="columns")
Y_target = dataset[target]

# 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(
    X_features, Y_target, stratify=Y_target, test_size=0.2, random_state=42
)

# Feature Engineering
numeric_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)

categorical_ord_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OrdinalEncoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipe, cols_numeric),
        ("categoric", categorical_pipe, cols_categoric),
        ("categoric ordinal", categorical_ord_pipe, cols_categoric_ord),
    ]
)

data_model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier())]
)

# Hyperparameter tuning
score = "recall"

hyperparameters = {
    "model__max_depth": [4, 5, 7, 9, 10],
    "model__max_features": [2, 3, 4, 5, 6, 7, 8, 9],
    "model__criterion": ["gini", "entropy"],
}

grid_search = GridSearchCV(
    data_model_pipeline,
    hyperparameters,
    cv=5,
    scoring=score,
    n_jobs=8,
)
grid_search.fit(x_train, y_train)

best_data_model_pipeline = grid_search.best_estimator_

# Evaluation

y_pred = best_data_model_pipeline.predict(x_test)
metric_result = recall_score(y_test, y_pred)
print(f"evaluation metric: {metric_result}")

# Model Validation
# baseline score
BASELINE_SCORE = 0.7

model_validation = metric_result > BASELINE_SCORE

if model_validation:
    print("Model validation passed")
else:
    print("Model validation failed")
    raise ValueError("Model validation failed")

# Save model

DATA_MODEL = Path.cwd().resolve() / "models"

dump(
    best_data_model_pipeline,
    DATA_MODEL / "first_basic_model.joblib",
    protocol=5,
)
