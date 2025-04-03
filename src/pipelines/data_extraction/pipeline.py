"""Logic Layer - DataExtractor Pipe"""

from typing import cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.data_extraction.data_extraction import download_csv_url


def data_extraction(data_extractor_config_path: str) -> pd.DataFrame:
    """Extract data using the provided configuration.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        pd.DataFrame: The extracted dataset.

    Raises:
        ValueError: If data extraction fails or returns None.
    """

    cfg = cast(DictConfig, OmegaConf.load(data_extractor_config_path))
    # Convert Hydra config to our Config type

    url = cfg.etl.url
    na_value = cfg.etl.na_value
    features = cfg.features

    # Initialize and execute data extraction

    dataset_raw = download_csv_url(
        url,
        use_columns=list(features.features + [features.target_column]),
        na_value=na_value,
    )

    if dataset_raw is None or dataset_raw.empty:
        raise ValueError("Data extraction returned no data")

    return dataset_raw
