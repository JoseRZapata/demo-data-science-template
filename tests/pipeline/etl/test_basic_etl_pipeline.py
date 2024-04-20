import pandas as pd
import pytest
from hydra import compose, initialize

from src.pipelines.etl.basic_etl_pipeline import data_transformation, get_data


@pytest.fixture
def hydra_config_path() -> str:
    return "../../../conf"


def test_basic_etl_pipeline(hydra_config_path: str) -> None:
    # Inicializar Hydra y componer la configuración
    with initialize(config_path=hydra_config_path):
        cfg = compose(config_name="config")

    # Ejecutar la primera parte del pipeline: descargar datos
    get_data(cfg)

    # Ejecutar la segunda parte del pipeline: transformar datos
    data_transformation(cfg)

    # Verificar que se haya creado el archivo intermedio correctamente
    intermediate_file = cfg.data.intermediate
    assert pd.read_parquet(intermediate_file).shape[0] > 0
