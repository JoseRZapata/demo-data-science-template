"""Tests for data type conversion function in data extraction pipeline."""

import os
import sys

import numpy as np
import pandas as pd

# Add the project root to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from src.pipelines.data_extraction.data_extraction_pipeline import data_type_conversion


def test_data_type_conversion_all_columns():
    """Test data type conversion for all column types."""
    # Create a test DataFrame with different data types
    df = pd.DataFrame({
        "cat_col": ["A", "B", "C", None],
        "float_col": ["1.1", "2.2", "not_float", None],
        "int_col": ["1", "2", "not_int", None],
        "bool_col": [True, False, None, None],
    })

    # Define column types
    cat_columns = ["cat_col"]
    float_columns = ["float_col"]
    int_columns = ["int_col"]
    bool_columns = ["bool_col"]

    # Call the function
    result = data_type_conversion(
        df,
        cat_columns=cat_columns,
        float_columns=float_columns,
        int_columns=int_columns,
        bool_columns=bool_columns,
    )

    # Verify the column types
    assert pd.api.types.is_categorical_dtype(result["cat_col"])
    assert pd.api.types.is_float_dtype(result["float_col"])
    assert pd.api.types.is_integer_dtype(result["int_col"])
    assert pd.api.types.is_bool_dtype(result["bool_col"])

    # Verify that conversion errors are handled correctly
    assert np.isnan(result["float_col"][2])  # "not_float" should be converted to NaN
    assert pd.isna(
        result["int_col"][2]
    )  # "not_int" should be converted to NA for Int64


def test_data_type_conversion_with_missing_columns():
    """Test data type conversion when specified columns don't exist in the DataFrame."""
    # Create a test DataFrame
    df = pd.DataFrame({"existing_col": [1, 2, 3]})

    # Define column types with columns that don't exist in the DataFrame
    cat_columns = ["nonexistent_cat_col"]
    float_columns = ["nonexistent_float_col"]
    int_columns = ["nonexistent_int_col"]
    bool_columns = ["nonexistent_bool_col"]

    # Call the function
    result = data_type_conversion(
        df,
        cat_columns=cat_columns,
        float_columns=float_columns,
        int_columns=int_columns,
        bool_columns=bool_columns,
    )

    # Verify that the function doesn't modify the DataFrame
    assert result.equals(df)
    assert "nonexistent_cat_col" not in result.columns


def test_data_type_conversion_empty_lists():
    """Test data type conversion with empty column lists."""
    # Create a test DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Call the function with empty lists
    result = data_type_conversion(
        df, cat_columns=[], float_columns=[], int_columns=[], bool_columns=[]
    )

    # Verify that the function doesn't modify the DataFrame
    assert result.equals(df)
    assert result.equals(df)
