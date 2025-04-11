"""ELT Pipeline - Executes Data pipes"""

from pathlib import Path

import pyarrow as pa

from src.pipelines.data_extraction.pipeline import data_extraction
from src.pipelines.data_preparation.preprocess_pipeline import data_preprocess
from src.pipelines.data_validation.pipeline import data_validation

if __name__ == "__main__":
    # data directory path
    DATA_DIR = Path.cwd().resolve() / "data"

    # data extraction
    dataset = data_extraction("conf/data_extraction.yaml")  # pragma: no cover

    # data validation
    data_validation(dataset, "conf/data_validation.yaml")  # pragma: no cover

    # data pre-processing
    dataset_preprocess = data_preprocess(dataset, "conf/data_preparation.yaml")  # pragma: no cover

    # save preprocessed dataset
    schema = pa.Table.from_pandas(dataset_preprocess).schema
    dataset_preprocess.to_parquet(
        DATA_DIR / "02_intermediate/dataset_type_fixed.parquet",
        index=False,
        schema=schema,
    )
    print("Data pre-processing completed successfully.")
