"""Data extraction pipeline implementation."""

import pandas as pd
from omegaconf import DictConfig

from src.data.data_extraction.data_extraction import download_csv_url


def data_type_conversion(
    df: pd.DataFrame,
    cat_columns: list,
    float_columns: list,
    int_columns: list,
    bool_columns: list,
) -> pd.DataFrame:
    """Convert DataFrame columns to appropriate data types.

    Args:
        df: Input DataFrame
        cat_columns: List of categorical columns
        float_columns: List of float columns
        int_columns: List of integer columns
        bool_columns: List of boolean columns

    Returns:
        DataFrame with converted data types
    """
    # Convert categorical columns
    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Convert float columns
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Convert integer columns
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                "Int64"
            )  # nullable integer type

    # Convert boolean columns
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    return df


def get_data(cfg: DictConfig) -> None:
    """Download raw data using the configuration.

    Args:
        cfg: Hydra configuration
    """
    # Extract values from config
    url = cfg.etl.url
    use_columns = [str(column) for column in cfg.features.features] + [
        str(cfg.features.target_column)
    ]
    raw_path = cfg.data.raw
    na_value = cfg.etl.na_value

    # Download the data
    download_csv_url(
        url=url,
        use_columns=use_columns,
        raw_path=raw_path,
        na_value=na_value,
    )


def data_transformation(cfg: DictConfig) -> None:
    """Transform the raw data and save to intermediate format.

    Args:
        cfg: Hydra configuration
    """
    # Read the raw data
    df = pd.read_csv(cfg.data.raw)

    # Since we don't have column type specifications in the config,
    # we'll infer types based on column names or content
    # This is a simplified approach - in production, you would want explicit type definitions

    # Infer categorical columns (non-numeric columns except target)
    all_columns = set(df.columns)
    numeric_columns = set(df.select_dtypes(include=["number"]).columns)
    categorical_columns = list(all_columns - numeric_columns)

    # Split numeric columns into float and int based on current types
    float_columns = list(df.select_dtypes(include=["float"]).columns)
    int_columns = list(df.select_dtypes(include=["int"]).columns)

    # Boolean columns - could be inferred from values, but we'll leave empty for now
    bool_columns = []

    # Convert data types
    df = data_type_conversion(
        df,
        cat_columns=categorical_columns,
        float_columns=float_columns,
        int_columns=int_columns,
        bool_columns=bool_columns,
    )

    # Save to parquet format
    df.to_parquet(cfg.data.intermediate, engine="pyarrow", index=False)


def run() -> None:
    """Run the data extraction pipeline."""
    import os
    from pathlib import Path

    from hydra import compose, initialize

    # Change to the project root directory
    repo_root = Path(__file__).parent.parent.parent.parent
    os.chdir(repo_root)

    # Now use a relative path for the config
    config_path = "conf"

    # Initialize Hydra configuration
    with initialize(config_path=config_path, version_base="1.1"):
        cfg = compose(config_name="data_extraction.yaml")

    # Run the pipeline steps
    get_data(cfg)
    data_transformation(cfg)
    data_transformation(cfg)
