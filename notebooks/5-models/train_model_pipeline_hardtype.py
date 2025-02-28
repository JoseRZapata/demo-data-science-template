# Train pipeline model hardcode just for academic purposes
# 2025-02-27

# ## 📚 Import  libraries

# base libraries for data science

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# 💾 Load data

dataset = pd.read_csv(
    "https://www.openml.org/data/get_csv/16826755/phpMYEkMl",
    low_memory=False,
    na_values="?",
)


dataset_features = dataset[
    [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
        "survived",
    ]
]
# Convert data types

# Categorical variables

dataset[["sex", "embarked", "pclass"]] = dataset[["sex", "embarked", "pclass"]].astype("category")

dataset["pclass"] = pd.Categorical(dataset["pclass"], categories=[3, 2, 1], ordered=True)

# Numerical variables

dataset[["age", "fare"]] = dataset[["age", "fare"]].astype("float")
dataset[["sibsp", "parch"]] = dataset[["sibsp", "parch"]].astype("int8")

# ### target variables
dataset["survived"] = dataset["survived"].astype("int8")

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
        ("numeric", numeric_pipe, ["age", "fare", "sibsp", "parch"]),
        ("categoric", categorical_pipe, ["sex", "embarked"]),
        ("categoric ordinal", categorical_ord_pipe, ["pclass"]),
    ]
)

# Train / Test split
X_features = dataset.drop("survived", axis="columns")
Y_target = dataset["survived"]

x_train, x_test, y_train, y_test = train_test_split(
    X_features, Y_target, stratify=Y_target, test_size=0.2, random_state=42
)

# Create pipeline
data_model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier())]
)

# Hyperparameter tunning
hyperparameters = {
    "model__max_depth": [4, 5, 7, 9, 10],
    "model__max_features": [2, 3, 4, 5, 6, 7, 8, 9],
    "model__criterion": ["gini", "entropy"],
}

grid_search = GridSearchCV(
    data_model_pipeline,
    hyperparameters,
    cv=5,
    scoring="recall",
    n_jobs=8,
)
grid_search.fit(x_train, y_train)

best_data_model_pipeline = grid_search.best_estimator_

# evaluation
y_pred = best_data_model_pipeline.predict(x_test)

metric_result = recall_score(y_test, y_pred)
print(f"evaluation metric: {metric_result}")

# Save the model
dump(
    best_data_model_pipeline,
    "models/model.joblib",
    protocol=5,
)
