# Train pipeline model
# 2025-02-27

# ## 📚 Import  libraries

# %%
# base libraries for data science
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

# ## 💾 Load data
url_data = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
dataset = pd.read_csv(url_data, low_memory=False, na_values="?")

# selected features
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
# ## Convert data types

# numerical columns
cols_numeric_float = ["age", "fare"]
cols_numeric_int = ["sibsp", "parch"]
cols_numeric = cols_numeric_float + cols_numeric_int

# categorical columns
cols_categoric = ["sex", "embarked"]
cols_categoric_ord = ["pclass"]
cols_categorical = cols_categoric + cols_categoric_ord

# Categorical variables

dataset[cols_categoric] = dataset[cols_categoric].astype("category")

dataset["pclass"] = pd.Categorical(dataset["pclass"], categories=[3, 2, 1], ordered=True)

# Numerical variables

dataset[cols_numeric_float] = dataset[cols_numeric_float].astype("float")
dataset[cols_numeric_int] = dataset[cols_numeric_int].astype("int8")

# ### target variables
target = "survived"

dataset[target] = dataset[target].astype("int8")

dataset = dataset.drop_duplicates()

# 👨‍🏭 Feature Engineering

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

# Train / Test split

# split data into features and target

X_features = dataset.drop(target, axis="columns")
Y_target = dataset[target]

# 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(
    X_features, Y_target, stratify=Y_target, test_size=0.2, random_state=42
)

# Create pipeline

data_model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier())]
)

# Hyperparameter tunning
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

# evaluation
y_pred = best_data_model_pipeline.predict(x_test)

metric_result = recall_score(y_test, y_pred)
print(f"evaluation metric: {metric_result}")

# Save the model
# %%
DATA_MODEL = Path.cwd().resolve().parents[1] / "models"
# %%
dump(
    best_data_model_pipeline,
    DATA_MODEL / "model.joblib",
    protocol=5,
)
