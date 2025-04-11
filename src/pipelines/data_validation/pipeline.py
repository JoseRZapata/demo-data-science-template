"""DataValidation Pipeline"""

from typing import cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.data_validation.data_validation import validate_data


def data_validation(dataset: pd.DataFrame, validation_config_path: str) -> pd.DataFrame:
    """Validate data using Pandera schemas defined in the configuration.

    Args:
        dataset (pd.DataFrame): Dataset to validate.
        validation_config_path (str): Path to the validation configuration file.

    Returns:
        pd.DataFrame: The validated dataset if validation passes.

    Raises:
        ValueError: If validation fails or if there are unexpected errors.
    """
    # Load validation configuration
    validation_rules = cast(DictConfig, OmegaConf.load(validation_config_path))

    # Validate data using pandera schema
    validated_data = validate_data(dataset, validation_rules)

    return validated_data
