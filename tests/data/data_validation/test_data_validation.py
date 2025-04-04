"""Tests for data validation functions."""

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pandera as pa
import pytest
from omegaconf import OmegaConf

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.data.data_validation.data_validation import (
    create_pandera_schema,
    validate_data,
)


@pytest.fixture
def sample_validation_config():
    """Create a sample validation configuration."""
    config_yaml = """
    columns:
      age:
        dtype: Float64
        nullable: false
        checks:
          - in_range:
              min: 0
              max: 120
      name:
        dtype: String
        nullable: true
      category:
        dtype: String
        nullable: false
        checks:
          - isin:
              allowed_values:
                - A
                - B
                - C
    """
    return OmegaConf.create(config_yaml)


@pytest.fixture
def valid_dataframe():
    """Create a dataframe that should pass validation."""
    return pd.DataFrame(
        {
            "age": [25.0, 40.0, 60.0],
            "name": ["John", "Jane", "Bob"],
            "category": ["A", "B", "C"],
        }
    )


@pytest.fixture
def invalid_dataframe():
    """Create a dataframe that should fail validation."""
    return pd.DataFrame(
        {
            "age": [25.0, -5.0, 200.0],  # Invalid: negative value and > 120
            "name": ["John", None, "Bob"],  # Valid because nullable is true
            "category": ["A", "D", "C"],  # Invalid: 'D' not in allowed values
        }
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe."""
    return pd.DataFrame(
        {
            "age": [25.0, 40.0, 60.0],
            "name": ["John", "Jane", "Bob"],
            "category": ["A", "B", "C"],
        }
    )


def test_create_pandera_schema(sample_validation_config):
    """Test that a pandera schema is correctly created from configuration."""
    schema = create_pandera_schema(sample_validation_config)

    # Check that it's the right type
    assert isinstance(schema, pa.DataFrameSchema)

    # Check the columns were correctly added
    assert "age" in schema.columns
    assert "name" in schema.columns
    assert "category" in schema.columns

    # Check the types were correctly set - use string comparison for robustness
    assert str(schema.columns["age"].dtype) == "float64"
    assert str(schema.columns["name"].dtype) == "str"
    assert str(schema.columns["category"].dtype) == "str"

    # Check nullable settings
    assert schema.columns["age"].nullable is False
    assert schema.columns["name"].nullable is True
    assert schema.columns["category"].nullable is False

    # Check that checks were added (age and category have checks)
    assert len(schema.columns["age"].checks) == 1
    assert len(schema.columns["category"].checks) == 1


@patch("src.data.data_validation.data_validation.create_pandera_schema")
def test_validate_data_valid(mock_create_schema, sample_validation_config, valid_dataframe):
    """Test that validation succeeds with valid data."""
    # Create a schema that will validate successfully
    schema = pa.DataFrameSchema(
        {
            "age": pa.Column(pa.dtypes.Float64, nullable=False),
            "name": pa.Column(pa.dtypes.String, nullable=True),
            "category": pa.Column(pa.dtypes.String, nullable=False),
        }
    )

    # Mock schema to avoid actually creating it
    mock_create_schema.return_value = schema

    # Should not raise an exception
    result = validate_data(valid_dataframe, sample_validation_config)

    # Should return a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Should have the same shape as input
    assert result.shape == valid_dataframe.shape


@patch("src.data.data_validation.data_validation.create_pandera_schema")
def test_validate_data_invalid(mock_create_schema, sample_validation_config, invalid_dataframe):
    """Test that validation fails with invalid data."""
    # Create a schema that will fail validation
    schema = pa.DataFrameSchema(
        {
            "age": pa.Column(
                pa.dtypes.Float64,
                nullable=False,
                checks=[pa.Check.in_range(min_value=0, max_value=120)],
            ),
            "name": pa.Column(pa.dtypes.String, nullable=True),
            "category": pa.Column(
                pa.dtypes.String, nullable=False, checks=[pa.Check.isin(["A", "B", "C"])]
            ),
        }
    )

    # Mock schema
    mock_create_schema.return_value = schema

    # Should raise ValueError
    with pytest.raises(ValueError):
        validate_data(invalid_dataframe, sample_validation_config)


def test_validate_unexpected_error(sample_dataframe, sample_validation_config):
    """Test handling of unexpected errors during validation."""
    # Create a mock schema for the validate call
    schema_mock = MagicMock()
    schema_mock.validate.side_effect = AttributeError("Schema validation error")

    # Patch the create_pandera_schema to return our mock schema
    with patch(
        "src.data.data_validation.data_validation.create_pandera_schema",
        return_value=schema_mock,
    ):
        # The validate_data function should catch AttributeError and convert to ValueError
        with pytest.raises(ValueError):
            validate_data(sample_dataframe, sample_validation_config)
