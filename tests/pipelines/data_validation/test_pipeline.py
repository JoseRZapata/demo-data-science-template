"""Tests for data validation pipeline."""

import os
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

# Add the project root to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from src.pipelines.data_validation.pipeline import data_validation


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": ["a", "b", "c"],
        "feature3": [True, False, True],
        "target": [0, 1, 0],
    })


@pytest.fixture
def sample_validation_config_file():
    """Create a sample validation config file for testing."""
    # Create a temp file with our test config
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "test_validation_config.yaml")

    # Create the config content
    config_content = """
    columns:
      feature1:
        dtype: Int64
        nullable: false
      feature2:
        dtype: String
        nullable: false
      feature3:
        dtype: Bool
        nullable: false
      target:
        dtype: Int64
        nullable: false
    """

    # Write the config to a temp file
    with open(config_path, "w") as f:
        f.write(config_content)

    # Return the path to the config file
    yield config_path

    # Cleanup
    os.unlink(config_path)
    os.rmdir(temp_dir)


@patch("src.pipelines.data_validation.pipeline.validate_data")
@patch("src.pipelines.data_validation.pipeline.OmegaConf.load")
def test_data_validation_success(
    mock_load, mock_validate_data, sample_dataframe, sample_validation_config_file
):
    """Test that data_validation correctly processes valid data."""
    # Mock OmegaConf load to return our config
    mock_config = OmegaConf.create({
        "columns": {
            "feature1": {"dtype": "Int64", "nullable": False},
            "feature2": {"dtype": "String", "nullable": False},
            "feature3": {"dtype": "Bool", "nullable": False},
            "target": {"dtype": "Int64", "nullable": False},
        }
    })
    mock_load.return_value = mock_config

    # Mock the validate_data function to return our dataframe
    mock_validate_data.return_value = sample_dataframe

    # Call the function
    result = data_validation(sample_dataframe, sample_validation_config_file)

    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_dataframe)

    # Verify the mocks were called with the right arguments
    mock_load.assert_called_once_with(sample_validation_config_file)
    mock_validate_data.assert_called_once_with(sample_dataframe, mock_config)


@patch("src.pipelines.data_validation.pipeline.validate_data")
@patch("src.pipelines.data_validation.pipeline.OmegaConf.load")
def test_data_validation_error(
    mock_load, mock_validate_data, sample_dataframe, sample_validation_config_file
):
    """Test that data_validation correctly raises errors."""
    # Mock OmegaConf load to return our config
    mock_config = OmegaConf.create({
        "columns": {
            "feature1": {"dtype": "Int64", "nullable": False},
            "feature2": {"dtype": "String", "nullable": False},
            "feature3": {"dtype": "Bool", "nullable": False},
            "target": {"dtype": "Int64", "nullable": False},
        }
    })
    mock_load.return_value = mock_config

    # Mock the validate_data function to raise an error
    mock_validate_data.side_effect = ValueError("Validation failed")

    # Call the function and expect an error
    with pytest.raises(ValueError, match="Validation failed"):
        data_validation(sample_dataframe, sample_validation_config_file)
