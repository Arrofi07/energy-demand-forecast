"""Smoke tests for every dashboard page using Streamlit's AppTest framework.

These are integration-style tests -- each page loads the real production
model bundle and the real saved backtest results from disk (via
`src/dashboard/data_access.py`), so they're marked `slow` like the other
real-data integration tests in this project (e.g.
`tests/pipeline/test_inference_features.py`'s full-history check).
"""

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.slow


def test_home_page_renders_without_error():
    at = AppTest.from_file("src/dashboard/app.py")
    at.run(timeout=60)

    assert not at.exception
    assert "Energy Demand Forecasting" in at.title[0].value
    assert len(at.dataframe) >= 1


class TestLiveForecastPage:
    def test_renders_without_error_before_generating(self):
        at = AppTest.from_file("src/dashboard/pages/1_Live_Forecast.py")
        at.run(timeout=60)

        assert not at.exception
        assert len(at.info) >= 1  # "Pick a forecast origin..." prompt

    def test_generate_button_produces_a_forecast_chart(self):
        at = AppTest.from_file("src/dashboard/pages/1_Live_Forecast.py")
        at.run(timeout=60)

        at.button[0].click().run(timeout=60)

        assert not at.exception
        assert len(at.get('plotly_chart')) >= 1


class TestModelComparisonPage:
    def test_24h_benchmark_view_renders_without_error(self):
        at = AppTest.from_file("src/dashboard/pages/2_Model_Comparison.py")
        at.run(timeout=60)

        assert not at.exception
        assert len(at.dataframe) >= 1
        assert len(at.get('plotly_chart')) >= 2  # bar chart + horizon-step line chart

    def test_horizon_sensitivity_view_renders_without_error(self):
        at = AppTest.from_file("src/dashboard/pages/2_Model_Comparison.py")
        at.run(timeout=60)

        at.radio[0].set_value("Horizon sensitivity (1h / 24h / 7d)").run(timeout=60)

        assert not at.exception
        assert len(at.get('plotly_chart')) >= 1


def test_anomaly_detection_page_renders_without_error():
    at = AppTest.from_file("src/dashboard/pages/3_Anomaly_Detection.py")
    at.run(timeout=60)

    assert not at.exception
    assert len(at.get('plotly_chart')) >= 1
    assert len(at.dataframe) >= 1


class TestBusinessImpactPage:
    def test_renders_with_default_assumptions(self):
        at = AppTest.from_file("src/dashboard/pages/4_Business_Impact.py")
        at.run(timeout=60)

        assert not at.exception
        assert len(at.metric) >= 4  # MAE, RMSE, MAE/household, illustrative savings

    def test_changing_portfolio_size_updates_the_savings_estimate(self):
        at = AppTest.from_file("src/dashboard/pages/4_Business_Impact.py")
        at.run(timeout=60)
        savings_before = at.metric[-1].value

        at.slider[0].set_value(100_000).run(timeout=60)

        assert not at.exception
        savings_after = at.metric[-1].value
        assert savings_after != savings_before
