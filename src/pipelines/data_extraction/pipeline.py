"""Logic Layer - DataExtractor Pipe"""

from typing import cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from app.interfaces.data.extractors.base import (
    Config,
    DataConfig,
    EtlConfig,
    FeatureConfig,
)
from app.logic.data.extractor.data_extractor import DataExtraction



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
    config = Config(
        type=cfg.type,
        etl=EtlConfig(**cfg.etl),
        data=DataConfig(**cfg.data),
        features=FeatureConfig(**cfg.features),
    )

    # Initialize and execute data extraction
    data_extractor = DataExtraction(config)
    dataset_raw = data_extractor.extract()

    if dataset_raw is None or dataset_raw.empty:
        logger.error("DataExtractor  - Dataset is empty")
        raise ValueError("Data extraction returned no data")
    return dataset_raw
