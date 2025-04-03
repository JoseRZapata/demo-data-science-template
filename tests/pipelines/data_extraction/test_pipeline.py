"""Tests for data extraction pipeline."""

import os
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

# Add the project root to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from src.pipelines.data_extraction.pipeline import data_extraction


@pytest.fixture
def sample_config_file():
    """Create a sample config file for testing."""
    # Create a temp file with our test config
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "test_extraction_config.yaml")

    # Create the config content
    config_content = """
    etl:
      url: https://example.com/dataset.csv
      na_value: '?'
    features:
      target_column: target
      features:
        - feature1
        - feature2
        - feature3
    """

    # Write the config to a temp file
    with open(config_path, "w") as f:
        f.write(config_content)

    # Return the path to the config file
    yield config_path

    # Cleanup
    os.unlink(config_path)
    os.rmdir(temp_dir)


@patch("src.pipelines.data_extraction.pipeline.download_csv_url")
def test_data_extraction_success(mock_download_csv_url, sample_config_file):
    """Test that data_extraction correctly extracts data when successful."""
    # Create a mock dataframe to return
    mock_df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": ["a", "b", "c"],
        "feature3": [True, False, True],
        "target": [0, 1, 0],
    })

    # Set up the mock to return our dataframe
    mock_download_csv_url.return_value = mock_df

    # Call the function
    result = data_extraction(sample_config_file)

    # Verify the expected return value
    assert isinstance(result, pd.DataFrame)
    assert result.equals(mock_df)

    # Verify the mock was called with the correct arguments
    mock_download_csv_url.assert_called_once()
    args, kwargs = mock_download_csv_url.call_args
    assert args[0] == "https://example.com/dataset.csv"
    assert sorted(kwargs["use_columns"]) == sorted([
        "feature1",
        "feature2",
        "feature3",
        "target",
    ])
    assert kwargs["na_value"] == "?"


@patch("src.pipelines.data_extraction.pipeline.download_csv_url")
def test_data_extraction_empty_result(mock_download_csv_url, sample_config_file):
    """Test that data_extraction raises an error when no data is returned."""
    # Set up the mock to return an empty dataframe
    mock_download_csv_url.return_value = pd.DataFrame()

    # Call the function and expect an error
    with pytest.raises(ValueError, match="Data extraction returned no data"):
        data_extraction(sample_config_file)


@patch("src.pipelines.data_extraction.pipeline.download_csv_url")
def test_data_extraction_none_result(mock_download_csv_url, sample_config_file):
    """Test that data_extraction raises an error when None is returned."""
    # Set up the mock to return None
    mock_download_csv_url.return_value = None

    # Call the function and expect an error
    with pytest.raises(ValueError, match="Data extraction returned no data"):
        data_extraction(sample_config_file)
