"""Test module for the data preparation preprocessing pipeline."""
# Disable Bandit assert warning for pytest test files

import os
import tempfile
from collections.abc import Generator

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.pipelines.data_preparation.preprocess_pipeline import (
    data_preprocess,
    data_type_conversion,
)


# pylint: disable=redefined-outer-name
@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame for testing.

    Returns:
        pd.DataFrame: Sample data for testing
    """
    return pd.DataFrame(
        {
            "pclass": [1, 2, 3, 1, 1],  # duplicated row at the end
            "age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "sibsp": [1, 1, 0, 1, 1],
            "embarked": ["S", "C", "S", "S", "S"],
        }
    )


@pytest.fixture
def config_file() -> Generator[str, None, None]:
    """Create a temporary config file for testing.

    Yields:
        str: Path to the temporary config file
    """
    config = {
        "data_types": {
            "pclass": "category",
            "age": "float32",
            "sibsp": "Int8",
            "embarked": "category",
        }
    }

    # Create a temporary file for the config
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "test_config.yaml")

    # Save the config using OmegaConf
    conf = OmegaConf.create(config)
    OmegaConf.save(conf, config_path)

    yield config_path

    # Cleanup
    os.unlink(config_path)
    os.rmdir(temp_dir)


def test_data_preprocess(sample_data: pd.DataFrame, config_file: str) -> None:
    """Test that data_preprocess function correctly preprocesses data.

    Args:
        sample_data: Test DataFrame
        config_file: Path to test config file
    """
    # Test preprocessing
    result = data_preprocess(sample_data, config_file)

    # Check that duplicates are removed
    assert len(result) == 4  # One duplicate row should be removed

    # Check that data types are set correctly
    assert result["pclass"].dtype.name == "category"
    assert result["age"].dtype.name == "float32"
    assert result["sibsp"].dtype.name == "Int8"
    assert result["embarked"].dtype.name == "category"


def test_data_type_conversion(sample_data: pd.DataFrame) -> None:
    """Test the data_type_conversion function.

    Args:
        sample_data: Test DataFrame
    """
    # Define a dictionary of data types for conversion
    data_types_dict = {
        "pclass": "category",
        "age": "float32",
        "sibsp": "int8",
        "embarked": "category",
    }

    # Test basic type conversion
    result = data_type_conversion(sample_data, data_types_dict)

    # Check that each column has the correct data type
    assert result["pclass"].dtype.name == "category"
    assert result["age"].dtype.name == "float32"
    assert result["sibsp"].dtype.name == "int8"
    assert result["embarked"].dtype.name == "category"

    # Verify original DataFrame is not modified
    assert sample_data["pclass"].dtype.name != "category"
    assert id(sample_data) != id(result)

    # Test with a column that doesn't exist in the DataFrame
    data_types_with_nonexistent = data_types_dict.copy()
    data_types_with_nonexistent["nonexistent_column"] = "float64"

    result_with_nonexistent = data_type_conversion(sample_data, data_types_with_nonexistent)

    # Should not throw an error and should convert the existing columns
    assert result_with_nonexistent["pclass"].dtype.name == "category"
    assert "nonexistent_column" not in result_with_nonexistent.columns

    # Test with an empty dictionary
    result_empty_dict = data_type_conversion(sample_data, {})

    # Should return a copy with the same data types
    assert id(sample_data) != id(result_empty_dict)
    assert result_empty_dict["pclass"].dtype.name == sample_data["pclass"].dtype.name
