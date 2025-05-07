"""Functions for data validation using Pandera.

Author: Jose R. Zapata <https://joserzapata.github.io/>
"""

from typing import Any

import pandas as pd
import pandera as pa
from beartype import beartype
from omegaconf import DictConfig


@beartype
def create_pandera_schema(validation_rules: DictConfig) -> pa.DataFrameSchema:
    """Create a pandera schema from configuration rules.

    Args:
        validation_rules (DictConfig): Configuration containing validation rules for columns.

    Returns:
        pa.DataFrameSchema: Schema for data validation.
    """
    columns: dict[str, Any] = {}

    for column_name, rules in validation_rules.columns.items():
        # Extract validation rules
        dtype = getattr(pa.dtypes, rules.dtype)
        nullable = rules.get("nullable", False)

        # Get checks if they exist
        checks: list[Any] = []
        if "checks" in rules:
            for check in rules.checks:
                check_name = next(iter(check))
                check_params = check[check_name]

                check_fn = getattr(pa.Check, check_name)
                if check_name == "in_range":
                    checks.append(check_fn(min_value=check_params.min, max_value=check_params.max))
                elif check_name == "isin":
                    checks.append(check_fn(check_params.allowed_values))
                elif check_name == "not_null":
                    checks.append(check_fn())

        # Create column schema
        columns[column_name] = pa.Column(dtype, nullable=nullable, checks=checks)

    return pa.DataFrameSchema(columns)


@beartype
def validate_data(dataset: pd.DataFrame, validation_rules: DictConfig) -> pd.DataFrame:
    """Validate the dataset against pandera schema rules.

    Args:
        dataset (pd.DataFrame): Data to validate.
        validation_rules (DictConfig): Configuration containing validation rules.

    Returns:
        pd.DataFrame: The validated dataset if validation passes.

    Raises:
        ValueError: If validation fails or if there are unexpected errors.
    """
    schema = create_pandera_schema(validation_rules)

    try:
        # Validate data and return the validated DataFrame
        validated_data = schema.validate(dataset)
        print("Data validation completed successfully.")
        return validated_data
    except pa.errors.SchemaError as exc:
        print("Schema validation failed!")
        raise ValueError(str(exc)) from None
    except (AttributeError, TypeError, ValueError) as exc:
        print("Data validation failed due to unexpected error!")
        raise ValueError(str(exc)) from None
