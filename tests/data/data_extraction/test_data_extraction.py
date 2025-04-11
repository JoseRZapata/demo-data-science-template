"""Tests for data extraction functions."""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.data.data_extraction.data_extraction import download_csv_url


def test_download_csv_url_success():
    """Test successful download of a CSV file."""
    # Mock pandas read_csv to avoid making actual HTTP requests
    mock_df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        }
    )

    with patch("pandas.read_csv", return_value=mock_df) as mock_read_csv:
        # Test with default parameters
        url = "https://example.com/data.csv"
        use_columns = ["col1", "col2"]

        result = download_csv_url(url, use_columns)

        # Assert the result is the expected DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.equals(mock_df)

        # Assert read_csv was called with the right parameters
        mock_read_csv.assert_called_once()
        call_args = mock_read_csv.call_args[1]
        assert call_args["usecols"] == use_columns
        assert call_args["low_memory"] is False


def test_download_csv_url_with_na_value():
    """Test CSV download with custom NA value."""
    # Mock pandas read_csv to avoid making actual HTTP requests
    mock_df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        }
    )

    with patch("pandas.read_csv", return_value=mock_df) as mock_read_csv:
        # Test with custom NA value
        url = "https://example.com/data.csv"
        use_columns = ["col1", "col2"]
        na_value = "N/A"

        result = download_csv_url(url, use_columns, na_value)

        # Assert the result is the expected DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.equals(mock_df)

        # Assert read_csv was called with the right parameters
        mock_read_csv.assert_called_once()
        call_args = mock_read_csv.call_args[1]
        assert call_args["usecols"] == use_columns
        assert call_args["na_values"] == na_value


def test_download_csv_url_error():
    """Test error handling during CSV download."""
    with patch("pandas.read_csv", side_effect=Exception("Download failed")) as mock_read_csv:
        # Test case where read_csv raises an exception
        url = "https://example.com/data.csv"
        use_columns = ["col1", "col2"]

        with pytest.raises(Exception, match="Download failed"):
            download_csv_url(url, use_columns)

        # Assert read_csv was called once
        mock_read_csv.assert_called_once()
