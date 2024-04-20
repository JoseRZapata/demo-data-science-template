import pandas as pd
import pytest
from libs.data_etl.data_etl import data_type_conversion, download_csv_url
from pytest_mock import MockerFixture


@pytest.fixture
def mock_read_csv(mocker: MockerFixture) -> None:
    def mock_function(
        url: str,
        usecols: list[str] | None = None,
        na_values: str | None = None,
        low_memory: bool | None = None,
    ) -> pd.DataFrame:
        # Create a mock DataFrame for testing
        data = {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}
        return pd.DataFrame(data)

    mocker.patch("src.libs.data_etl.data_etl.pd.read_csv", side_effect=mock_function)


def test_download_csv_url(mock_read_csv: MockerFixture) -> None:
    url = "https://example.com/data.csv"
    use_columns = ["column1", "column2"]
    raw_path = "data/01_raw/titanic_raw.csv"
    na_value = ""

    download_csv_url(url, use_columns, raw_path, na_value)

    # Assert that the CSV file is saved correctly
    df_raw = pd.read_csv(raw_path, usecols=use_columns)
    assert df_raw.shape == (3, 2)
    assert df_raw.columns.tolist() == use_columns


def test_data_type_conversion() -> None:
    # Crear un DataFrame de prueba
    data = pd.DataFrame(
        {
            "cat_column": ["a", "b", "c"],
            "float_column": [1.1, 2.2, 3.3],
            "int_column": [1, 2, 3],
            "bool_column": [True, False, True],
        }
    )

    # Definir las columnas para cada tipo de datos
    cat_columns = ["cat_column"]
    float_columns = ["float_column"]
    int_columns = ["int_column"]
    bool_columns = ["bool_column"]

    # Llamar a la función data_type_conversion
    converted_data = data_type_conversion(
        data, cat_columns, float_columns, int_columns, bool_columns
    )

    # Verificar que los tipos de datos de las columnas son correctos
    assert converted_data["cat_column"].dtype.name == "category"
    assert converted_data["float_column"].dtype.name == "float64"
    assert converted_data["int_column"].dtype.name == "int64"
    assert converted_data["bool_column"].dtype.name == "bool"
