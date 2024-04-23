"""Functions for data extraction, transformation, and loading (ETL)

Author: Jose R. Zapata <https://joserzapata.github.io/>
"""

import pandas as pd


def download_csv_url(
    url: str, use_columns: list, raw_path: str, na_value: str = ""
) -> None:
    """Extract csv data from a given URL and save it as a parquet file.
    use_columns is a list of column names to use from the data.

    Args:
        url (str):          URL from which to extract the data.
        use_columns (list): List of column names to use from the data.
        raw_path (str):     Path to save the raw data.
        na_value (str):     Additional strings to recognize as NA/NaN.

    Returns:
        None

    """
    df_raw = pd.read_csv(url, usecols=use_columns, na_values=na_value, low_memory=False)
    df_raw.to_csv(raw_path)  # no parsing of mixed


def data_type_conversion(
    data: pd.DataFrame,
    cat_columns: list,
    float_columns: list,
    int_columns: list,
    bool_columns: list,
) -> pd.DataFrame:
    """Converts the data type of a column in a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing features and target.
        cat_columns (list): List of columns to convert to category data type.
        float_columns (list): List of columns to convert to float data type.
        int_columns (list): List of columns to convert to integer data type.
        bool_columns (list): List of columns to convert to boolean data type.

    Returns:
        pd.DataFrame: DataFrame with the specified specified data types.
    """
    data[cat_columns] = data[cat_columns].astype("category")
    data[float_columns] = data[float_columns].astype("float")
    data[int_columns] = data[int_columns].astype("int")
    data[bool_columns] = data[bool_columns].astype("bool")
    return data
