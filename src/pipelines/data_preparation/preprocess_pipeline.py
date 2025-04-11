"""
Module for preprocessing data before model training.
"""

import pandas as pd
from omegaconf import OmegaConf


def data_preprocess(data: pd.DataFrame, config_path: str) -> pd.DataFrame:
    """Preprocess data by removing duplicates and setting data types.

    Args:
        data (pd.DataFrame): Input dataset
        config_path (str): Path to the configuration file

    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    # Convert OmegaConf to Python dictionary
    data_types_dict = OmegaConf.to_container(config.data_types)
    # Remove duplicates
    data_no_duplicates = data.drop_duplicates().reset_index(drop=True)
    # Set data types for each column
    data_no_duplicates = data_type_conversion(data_no_duplicates, data_types_dict)
    return data_no_duplicates


def data_type_conversion(data: pd.DataFrame, data_types_dict: dict) -> pd.DataFrame:
    """Converts the data type of a column in a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing features and target.
        data_types_dict (dict): Dictionary of column names and their data types.

    Returns:
        pd.DataFrame: DataFrame with the specified data types.
    """
    data_copy = data.copy()
    for column, dtype in data_types_dict.items():
        if column in data_copy.columns:
            data_copy[column] = data_copy[column].astype(dtype)
    return data_copy
