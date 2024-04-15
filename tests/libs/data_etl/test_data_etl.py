import pandas as pd
import pytest
from libs.data_etl.data_etl import download_csv_url
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
