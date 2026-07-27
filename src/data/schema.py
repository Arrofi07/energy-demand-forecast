"""Pandera schema for the cleaned minute-level power consumption data."""

import pandera as pa
from pandera import Column, DataFrameSchema, Check

minute_schema = DataFrameSchema(
    {
        "datetime": Column(pa.DateTime, nullable=False),
        "global_active_power": Column(float, Check.ge(0), nullable=True),
        "global_reactive_power": Column(float, Check.ge(0), nullable=True),
        "voltage": Column(float, Check.in_range(150, 280), nullable=True),
        "global_intensity": Column(float, Check.ge(0), nullable=True),
        "sub_metering_1": Column(float, Check.ge(0), nullable=True),
        "sub_metering_2": Column(float, Check.ge(0), nullable=True),
        "sub_metering_3": Column(float, Check.ge(0), nullable=True),
    },
    coerce=True,
)