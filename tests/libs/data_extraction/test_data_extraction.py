import pandas as pd
import pytest
from pytest_mock import MockerFixture

from libs.data_extraction.data_extraction import data_type_conversion, download_csv_url


@pytest.fixture
def mock_read_csv(mocker: MockerFixture) -> None:
    def mock_function(
        url: str,
        usecols: list[str] | None = None,
        na_values: str | None = None,
        low_memory: bool | None = None,
    ) -> pd.DataFrame:
        # Crear un DataFrame de prueba
        data = {
            "pclass": [1, 2, 3],
            "name": ["John", "Jane", "Jack"],
            "sex": ["male", "female", "male"],
            "age": [30, 25, 35],
            "sibsp": [1, 0, 1],
            "parch": [0, 1, 1],
            "fare": [100.0, 50.0, 25.0],
            "embarked": ["C", "S", "Q"],
            "survived": [1, 0, 1],
        }
        return pd.DataFrame(data)

    mocker.patch(
        "src.libs.data_extraction.data_extraction.pd.read_csv",
        side_effect=mock_function,
    )


def test_download_csv_url(mocker: MockerFixture) -> None:
    url = "https://example.com/data.csv"
    use_columns = [
        "pclass",
        "name",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
        "survived",
    ]
    raw_path = "data/01_raw/test_raw.csv"
    na_value = ""

    # Mock pd.read_csv and DataFrame.to_csv
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=pd.DataFrame())
    mocker.patch("pandas.DataFrame.to_csv", return_value=None)

    download_csv_url(url, use_columns, raw_path, na_value)

    # Assert that pd.read_csv was called with the correct arguments
    mock_read_csv.assert_called_once_with(
        url, usecols=use_columns, na_values=na_value, low_memory=False
    )


def test_data_type_conversion(mock_read_csv: MockerFixture) -> None:
    # define column types
    cat_columns = ["pclass", "sex", "embarked"]
    float_columns = ["age", "fare"]
    int_columns = ["sibsp", "parch"]
    bool_columns = ["survived"]

    # call function
    test_data = pd.read_csv("data/01_raw/titanic_raw.csv")
    converted_data = data_type_conversion(
        test_data,
        cat_columns,
        float_columns,
        int_columns,
        bool_columns,
    )

    # check column data types
    assert converted_data["pclass"].dtype.name == "category"
    assert converted_data["sex"].dtype.name == "category"
    assert converted_data["embarked"].dtype.name == "category"
    assert converted_data["age"].dtype.name == "float64"
    assert converted_data["fare"].dtype.name == "float64"
    assert converted_data["sibsp"].dtype.name == "int64"
    assert converted_data["parch"].dtype.name == "int64"
    assert converted_data["survived"].dtype.name == "bool"
