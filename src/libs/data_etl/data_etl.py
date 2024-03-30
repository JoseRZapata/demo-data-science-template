"""Functions for data extraction, transformation, and loading (ETL)

Author: Jose R. Zapata <https://joserzapata.github.io/>
"""

import pandas as pd


def extract_data(url: str, use_columns: list, raw: str) -> None:
    """Extract csv data from a given URL and save it as a parquet file.
    use_columns is a list of column names to use from the data.

    Args:
        url (str):          URL from which to extract the data.
        use_columns (list): List of column names to use from the data.
        raw (str):          Path to save the raw data.

    Returns:
        None

    """
    df_raw = pd.read_csv(url, usecols=use_columns, low_memory=False)
    df_raw.to_parquet(raw, engine="pyarrow")  # no parsing of mixed
