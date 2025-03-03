import pandas as pd
import pytest
from hydra import compose, initialize
from pytest_mock import MockerFixture

from src.pipelines.data_extraction import data_extraction_pipeline
from src.pipelines.data_extraction.data_extraction_pipeline import main


@pytest.fixture
def hydra_config_path() -> str:
    return "../../../conf/data_extraction"


def test_data_extraction_pipeline(hydra_config_path: str, mocker: MockerFixture) -> None:
    # Import mock for the download_csv_url function
    mock_download_csv_url = mocker.patch(
        "src.pipelines.data_extraction.data_extraction_pipeline.download_csv_url"
    )

    # import mock for the data_type_conversion function
    mock_data_type_conversion = mocker.patch(
        "src.pipelines.data_extraction.data_extraction_pipeline.data_type_conversion"
    )

    # Mock pd.read_csv to return a dummy DataFrame
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=pd.DataFrame())

    # Initialize Hydra and the config
    with initialize(config_path=hydra_config_path, version_base="1.1"):
        cfg = compose(config_name="data_extraction.yaml")

    # Execute the pipeline
    data_extraction_pipeline.get_data(cfg)
    data_extraction_pipeline.data_transformation(cfg)

    # check that the function was called with the right arguments
    mock_download_csv_url.assert_called_once_with(
        url=cfg.etl.url,
        use_columns=[str(column) for column in cfg.features] + [str(cfg.target_column)],
        raw_path=cfg.data.raw,
        na_value=cfg.etl.na_value,
    )

    args, kwargs = mock_data_type_conversion.call_args
    assert isinstance(args[0], pd.DataFrame)  # check that the first argument is a DataFrame
    assert kwargs == {
        "cat_columns": cfg.cols_categoric._content,
        "float_columns": cfg.cols_numeric_float._content,
        "int_columns": cfg.cols_numeric_int._content,
        "bool_columns": cfg.cols_boolean._content,
    }

    # Check that pd.read_csv was called with the correct arguments
    mock_read_csv.assert_called_once_with(cfg.data.raw)


def test_get_data(hydra_config_path: str, mocker: MockerFixture) -> None:
    # Import mock for the download_csv_url function
    mock_download_csv_url = mocker.patch(
        "src.pipelines.data_extraction.data_extraction_pipeline.download_csv_url"
    )
    # Initialize Hydra and the config
    with initialize(config_path=hydra_config_path, version_base="1.1"):
        cfg = compose(config_name="data_extraction.yaml")
    # Execute the get_data function
    data_extraction_pipeline.get_data(cfg)
    # check that the function was called with the right arguments
    mock_download_csv_url.assert_called_once_with(
        url=cfg.etl.url,
        use_columns=[str(column) for column in cfg.features] + [str(cfg.target_column)],
        raw_path=cfg.data.raw,
        na_value=cfg.etl.na_value,
    )


def test_data_transformation(hydra_config_path: str, mocker: MockerFixture) -> None:
    # import mock for the data_type_conversion function
    mock_data_type_conversion = mocker.patch(
        "src.pipelines.data_extraction.data_extraction_pipeline.data_type_conversion",
        return_value=pd.DataFrame(),  # Ensure it returns a DataFrame
    )
    # Mock pd.read_csv to return a dummy DataFrame
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=pd.DataFrame())
    # Mock pd.DataFrame.to_parquet to do nothing
    mock_to_parquet = mocker.patch("pandas.DataFrame.to_parquet")
    # Initialize Hydra and the config
    with initialize(config_path=hydra_config_path, version_base="1.1"):
        cfg = compose(config_name="data_extraction.yaml")
    # Execute the data_transformation function
    data_extraction_pipeline.data_transformation(cfg)
    args, kwargs = mock_data_type_conversion.call_args
    assert isinstance(args[0], pd.DataFrame)  # check that the first argument is a DataFrame
    assert kwargs == {
        "cat_columns": cfg.cols_categoric._content,
        "float_columns": cfg.cols_numeric_float._content,
        "int_columns": cfg.cols_numeric_int._content,
        "bool_columns": cfg.cols_boolean._content,
    }
    # Check that pd.read_csv was called with the correct arguments
    mock_read_csv.assert_called_once_with(cfg.data.raw)
    # Check that to_parquet was called with the correct arguments
    mock_to_parquet.assert_called_once_with(cfg.data.intermediate, engine="pyarrow", index=False)


def test_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.pipelines.data_extraction.data_extraction_pipeline.get_data", lambda: None
    )
    monkeypatch.setattr(
        "src.pipelines.data_extraction.data_extraction_pipeline.data_transformation",
        lambda: None,
    )

    main()
