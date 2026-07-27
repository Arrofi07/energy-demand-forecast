energy-demand-forecasting/
├── data/
│   ├── raw/                  # original UCI txt file (gitignored, large)
│   ├── processed/            # hourly.parquet, daily.parquet
├── notebooks/
│   ├── 01_eda_trends_seasonality.ipynb
│   ├── 02_anomaly_detection.ipynb
│   ├── 03_model_classical.ipynb
│   ├── 04_model_prophet_statsforecast.ipynb
│   ├── 05_model_lstm.ipynb
│   ├── 06_model_comparison.ipynb
├── src/
│   ├── data/                 # ingestion, resampling, feature engineering
│   ├── models/                # arima.py, prophet_model.py, statsforecast_model.py, lstm.py
│   ├── evaluation/            # backtesting, metrics, business_impact.py
│   ├── api/                    # FastAPI app
│   └── monitoring/            # drift checks, alerting logic (even if simulated)
├── dashboard/                  # Streamlit app
├── tests/
├── docs/
│   ├── business_case.md
│   ├── production_design.md    # retraining cadence, alerting, monitoring
├── docker-compose.yml
├── pyproject.toml
└── README.md