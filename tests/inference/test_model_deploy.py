from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response

# Importar la app desde el módulo correcto
from src.inference.model_deploy import app

# Constantes para códigos de estado HTTP
HTTP_200_OK = 200
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_500_INTERNAL_SERVER_ERROR = 500

# Crear un cliente de prueba para la aplicación FastAPI
client: TestClient = TestClient(app)


def test_read_root() -> None:
    """Prueba el endpoint raíz ("/").

    Verifica que el endpoint raíz devuelve un código de estado 200
    y el mensaje de bienvenida esperado.
    """
    response: Response = client.get("/")
    assert response.status_code == HTTP_200_OK
    assert response.json() == "Welcome to the Titanic Survival Prediction API! more info at /docs"


def test_predict_success() -> None:
    """Prueba el endpoint de predicción ("/predict") con datos válidos.

    Simula la carga exitosa de un modelo y una predicción. Verifica que
    la respuesta del endpoint sea 200 OK, contenga la clave 'survived',
    y que el modelo simulado haya sido llamado correctamente.
    """
    # Datos de entrada válidos
    valid_input_data: dict[str, int | str | float] = {
        "age": 30,
        "pclass": 1,
        "sex": "male",
        "sibsp": 1,
        "parch": 0,
        "fare": 70.0,
        "embarked": "C",
    }
    # Simular que el modelo existe y devuelve una predicción
    with patch("joblib.load") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]  # Simular que predice 'survived'
        mock_load.return_value = mock_model

        response: Response = client.post("/predict", json=valid_input_data)

        assert response.status_code == HTTP_200_OK
        response_data: dict[str, int] = response.json()
        assert "survived" in response_data
        assert response_data["survived"] in [0, 1]
        mock_load.assert_called_once_with(
            "models/titanic_classification-random_forest-v1.joblib"  # ACTUALIZADO
        )  # Verificar que se llamó a load con la ruta correcta
        mock_model.predict.assert_called_once()


def test_predict_model_not_found() -> None:
    """Prueba el endpoint de predicción cuando el modelo no se encuentra.

    Simula que `joblib.load` devuelve None (modelo no encontrado) y verifica
    que se lance una excepción `ValueError` con el mensaje esperado.
    """
    input_data: dict[str, int | str | float] = {
        "age": 40,
        "pclass": 2,
        "sex": "female",
        "sibsp": 0,
        "parch": 1,
        "fare": 100.0,
        "embarked": "S",
    }
    # Simular que joblib.load devuelve None (modelo no encontrado)
    with patch("joblib.load") as mock_load:
        mock_load.return_value = None
        # Verificar que se lanza un ValueError cuando el modelo no se encuentra
        with pytest.raises(ValueError, match="Model not found"):
            client.post("/predict", json=input_data)


# Pruebas de validación de Pydantic (FastAPI las maneja automáticamente, pero podemos añadir algunas)
@pytest.mark.parametrize(
    "invalid_payload, expected_status_code",
    [
        ({"age": -5}, HTTP_422_UNPROCESSABLE_ENTITY),  # Edad inválida (menor que 0)
        ({"pclass": 4}, HTTP_422_UNPROCESSABLE_ENTITY),  # Clase inválida (mayor que 3)
        ({"sex": "other"}, HTTP_422_UNPROCESSABLE_ENTITY),  # Sexo inválido
        (
            {"embarked": "X"},
            HTTP_422_UNPROCESSABLE_ENTITY,
        ),  # Puerto de embarque inválido
        # El caso ({}, 422) se elimina porque un payload vacío es válido
        # debido a los valores por defecto en el modelo TitanicInput.
    ],
)
def test_predict_invalid_input_data(
    invalid_payload: dict[str, int | str | float], expected_status_code: int
) -> None:
    """Prueba el endpoint de predicción con varios datos de entrada inválidos.

    Args:
        invalid_payload: Un diccionario que representa un payload JSON inválido.
        expected_status_code: El código de estado HTTP esperado para el payload inválido.
    """
    # Completar el payload con valores por defecto para los campos no especificados en invalid_payload
    # para asegurar que la validación falle solo por el campo que queremos probar.
    base_payload: dict[str, int | str | float] = {
        "age": 30,
        "pclass": 1,
        "sex": "male",
        "sibsp": 1,
        "parch": 0,
        "fare": 70.0,
        "embarked": "C",
    }
    payload_to_send: dict[str, int | str | float] = {**base_payload, **invalid_payload}

    response: Response = client.post("/predict", json=payload_to_send)
    assert response.status_code == expected_status_code
