"""Functions for data extraction, transformation, and loading (ETL)

Author: Jose R. Zapata <https://joserzapata.github.io/>
"""

import pandas as pd


# @beartype
def download_csv_url(url: str, use_columns: list, na_value: str = "") -> pd.DataFrame:
    """Extract csv data from a given URL and save it as a parquet file.
    use_columns is a list of column names to use from the data.

    Args:
        url (str):          URL from which to extract the data.
        use_columns (list): List of column names to use from the data.
        na_value (str):     Additional strings to recognize as NA/NaN.

    Returns:
        pd.DataFrame: The extracted dataset.

    """
    df_raw = pd.read_csv(url, usecols=use_columns, na_values=na_value, low_memory=False)
    return df_raw
