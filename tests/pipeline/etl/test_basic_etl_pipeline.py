import pandas as pd
import pytest
from hydra import compose, initialize
from pytest_mock import MockerFixture

from src.pipelines.etl import basic_etl_pipeline


@pytest.fixture
def hydra_config_path() -> str:
    return "../../../conf"


def test_basic_etl_pipeline(hydra_config_path: str, mocker: MockerFixture) -> None:
    # Import mock for the download_csv_url function
    mock_download_csv_url = mocker.patch(
        "src.pipelines.etl.basic_etl_pipeline.download_csv_url"
    )

    # import mock for the data_type_conversion function
    mock_data_type_conversion = mocker.patch(
        "src.pipelines.etl.basic_etl_pipeline.data_type_conversion"
    )

    # Initialize Hydra and the config
    with initialize(config_path=hydra_config_path, version_base="1.1"):
        cfg = compose(config_name="config")

    # Execute the pipeline
    basic_etl_pipeline.get_data(cfg)
    basic_etl_pipeline.data_transformation(cfg)

    # check that the function was called with the right arguments
    mock_download_csv_url.assert_called_once_with(
        url=cfg.etl.url,
        use_columns=[str(column) for column in cfg.features] + [str(cfg.target_column)],
        raw_path=cfg.data.raw,
        na_value=cfg.etl.na_value,
    )

    args, kwargs = mock_data_type_conversion.call_args
    assert isinstance(
        args[0], pd.DataFrame
    )  # check that the first argument is a DataFrame
    assert kwargs == {
        "cat_columns": cfg.cols_categoric._content,
        "float_columns": cfg.cols_numeric_float._content,
        "int_columns": cfg.cols_numeric_int._content,
        "bool_columns": cfg.cols_boolean._content,
    }
