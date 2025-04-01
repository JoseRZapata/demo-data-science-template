"""ELT Pipeline - Executes Data pipes"""

from app.logic.data.preparator.pipe import data_preparation
from app.logic.data.validator.pipe import data_validation
from app.pipelines.constants import (
    DATA_EXTRACTOR_CONFIG_PATH,
    DATA_PREPARATION_CONF_PATH,
    DATA_VALIDATION_CONFIG_PATH,
)

from pipelines.data_extraction.pipeline import data_extraction

if __name__ == "__main__":
    dataset = data_extraction(DATA_EXTRACTOR_CONFIG_PATH)  # pragma: no cover
    data_validation(dataset, DATA_VALIDATION_CONFIG_PATH)  # pragma: no cover
    data_preparation(dataset, DATA_PREPARATION_CONF_PATH)  # pragma: no cover
