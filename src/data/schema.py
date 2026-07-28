"""Pandera schema for validating cleaned minute-level power consumption data."""

import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check


# Define the expected structure and quality constraints
# for the cleaned minute-level dataset.
minute_schema = DataFrameSchema(
    {
        # Timestamp used as the time index for forecasting and aggregation.
        # Missing timestamps are not allowed.
        "datetime": Column(
            pa.DateTime,
            nullable=False,
        ),

        # Total household active power consumption (kW).
        # Negative consumption values are physically impossible,
        # but missing values are allowed before final preprocessing.
        "global_active_power": Column(
            float,
            Check.ge(0),
            nullable=True,
        ),

        # Reactive power consumption (kVAR).
        # Must be non-negative because power magnitude cannot be negative.
        "global_reactive_power": Column(
            float,
            Check.ge(0),
            nullable=True,
        ),

        # Household voltage measurement (V).
        # The range check removes physically unrealistic sensor values.
        "voltage": Column(
            float,
            Check.in_range(150, 280),
            nullable=True,
        ),

        # Total current intensity (A).
        "global_intensity": Column(
            float,
            Check.ge(0),
            nullable=True,
        ),

        # Individual appliance energy consumption measurements.
        # Sub-meter values represent energy usage and must be non-negative.
        "sub_metering_1": Column(
            float,
            Check.ge(0),
            nullable=True,
        ),

        "sub_metering_2": Column(
            float,
            Check.ge(0),
            nullable=True,
        ),

        "sub_metering_3": Column(
            float,
            Check.ge(0),
            nullable=True,
        ),
    },

    # Automatically convert columns to the declared data types.
    # This ensures consistent input format before validation.
    coerce=True,
)