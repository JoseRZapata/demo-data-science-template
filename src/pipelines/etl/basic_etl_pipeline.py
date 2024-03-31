"""
This basic script performs pipeline of data extraction, transformation,
and loading (ETL) for the Titanic dataset.
It reads the dataset from a URL and stores it in a pandas DataFrame.

A better approach would be to use a orchestrator like: prefect, kedro, zenml, etc.

Author: Jose R. Zapata <https://joserzapata.github.io/>
"""

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.libs.data_etl.data_etl import data_type_conversion, download_csv_url


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def get_data(cfg: DictConfig) -> None:
    download_csv_url(
        url=cfg.etl.url,
        use_columns=cfg.etl.use_columns,
        raw_path=cfg.data.raw,
        na_value=cfg.etl.na_value,
    )


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def data_transformation(cfg: DictConfig) -> None:
    """Transform the raw data into a more suitable format for machine learning.

    steps:
    - Read raw data from cvs
    - Convert data types (categorical, float, int, bool)
    - Save the intermediate data as parquet

    """
    # Pandas indexing needs a list of columns, so cfg.etl.cols_categoric has to be
    # converted to a list adding ._content -> cfg.etl.cols_categoric._content
    (
        pd.read_csv(cfg.data.raw)
        .pipe(
            data_type_conversion,
            cat_columns=cfg.etl.cols_categoric._content,
            float_columns=cfg.etl.cols_numeric_float._content,
            int_columns=cfg.etl.cols_numeric_int._content,
            bool_columns=cfg.etl.cols_boolean._content,
        )
        .to_parquet(cfg.data.intermediate, engine="pyarrow", index=False)
    )


if __name__ == "__main__":
    # download raw data
    get_data()
    # transform data in correct format
    data_transformation()
    print("ETL pipeline completed.")
