import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.data.schema import minute_schema


def _valid_row(**overrides):
    row = {
        "datetime": pd.Timestamp("2020-01-01 00:00:00"),
        "global_active_power": 1.5,
        "global_reactive_power": 0.1,
        "voltage": 230.0,
        "global_intensity": 5.0,
        "sub_metering_1": 0.0,
        "sub_metering_2": 0.0,
        "sub_metering_3": 0.0,
    }
    row.update(overrides)
    return row


def test_valid_dataframe_passes():
    df = pd.DataFrame([_valid_row(), _valid_row()])
    validated = minute_schema.validate(df)
    assert len(validated) == 2


def test_nullable_measurement_columns_allow_nan():
    df = pd.DataFrame([_valid_row(global_active_power=None)])
    validated = minute_schema.validate(df)
    assert pd.isna(validated["global_active_power"].iloc[0])


def test_null_datetime_is_rejected():
    df = pd.DataFrame([_valid_row(datetime=None)])
    with pytest.raises((SchemaError, SchemaErrors)):
        minute_schema.validate(df, lazy=True)


def test_negative_power_is_rejected():
    df = pd.DataFrame([_valid_row(global_active_power=-1.0)])
    with pytest.raises((SchemaError, SchemaErrors)):
        minute_schema.validate(df, lazy=True)


def test_voltage_out_of_range_is_rejected():
    df = pd.DataFrame([_valid_row(voltage=400.0)])
    with pytest.raises((SchemaError, SchemaErrors)):
        minute_schema.validate(df, lazy=True)


def test_negative_sub_metering_is_rejected():
    df = pd.DataFrame([_valid_row(sub_metering_1=-0.5)])
    with pytest.raises((SchemaError, SchemaErrors)):
        minute_schema.validate(df, lazy=True)


def test_numeric_strings_are_coerced():
    df = pd.DataFrame([_valid_row(global_active_power="1.5")])
    validated = minute_schema.validate(df)
    assert validated["global_active_power"].dtype == float
