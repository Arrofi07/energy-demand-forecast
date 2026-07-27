# Energy Demand Forecasting — Project Roadmap

> **Portfolio-complete checkpoint:** Phases 1–10 form a complete, story-worthy
> portfolio piece on their own (data → EDA → anomalies → features → baselines →
> classical → ML/DL → evaluation → business impact). Phases 11–16 are the
> "production-grade" follow-up that demonstrates engineering maturity — treat
> them as a second milestone, not a blocker to sharing the work.

---

## Phase 1 — Data Engineering

- [ ] Download and validate the UCI Household Energy Consumption dataset
- [ ] Build a reproducible data loading pipeline
- [ ] Data cleaning and preprocessing pipeline
- [ ] Missing value analysis and handling
- [ ] Convert raw CSV to Parquet for efficient storage
- [ ] Build a DuckDB analytical database
- [ ] Data validation with **Pandera** (schema/type/range checks)

---

## Phase 2 — Exploratory Data Analysis

- [ ] Data quality report (missing values, duplicates, outliers)
- [ ] Exploratory data analysis notebook
- [ ] Daily consumption pattern analysis
- [ ] Weekly seasonality analysis
- [ ] Monthly and yearly trend analysis
- [ ] STL decomposition (trend, seasonality, residual)
- [ ] Autocorrelation (ACF) and Partial ACF analysis
- [ ] Holiday and weekend consumption analysis
- [ ] Correlation analysis between electrical variables
- [ ] Interactive Plotly visualizations

---

## Phase 3 — Anomaly Detection

- [ ] Z-score anomaly detection
- [ ] Rolling IQR anomaly detection
- [ ] STL residual anomaly detection
- [ ] Isolation Forest anomaly detection (multivariate)
- [ ] Investigate major anomaly events (root-cause each one)
- [ ] Generate anomaly report

---

## Phase 4 — Feature Engineering

- [ ] Time-based features (hour, weekday, month, season)
- [ ] Lag features (1h, 6h, 24h, 168h)
- [ ] Rolling statistics (mean, std, min, max)
- [ ] Difference and percentage change features
- [ ] Cyclical encoding (sin/cos for time variables)
- [ ] Feature importance analysis

---

## Phase 5 — Forecasting Baselines

- [ ] Naive Forecast
- [ ] Seasonal Naive Forecast
- [ ] Moving Average Forecast
- [ ] Benchmark all baseline methods

---

## Phase 6 — Statistical Forecasting

- [ ] Stationarity testing (ADF & KPSS)
- [ ] ARIMA implementation
- [ ] SARIMA implementation
- [ ] Hyperparameter tuning
- [ ] Residual diagnostics
- [ ] MLflow experiment tracking

---

## Phase 7 — Machine Learning Models

- [ ] Prophet implementation
- [ ] `statsforecast` AutoARIMA/AutoETS (modern Prophet-replacement comparison)
- [ ] **LightGBM forecasting (primary gradient-boosting model)**
- [ ] XGBoost / Random Forest — optional ablation only (brief note: "tried X,
      saw negligible difference vs LightGBM, here's why") rather than a fully
      parallel track
- [ ] Hyperparameter tuning with Optuna + MLflow
- [ ] Feature importance comparison
- [ ] **SHAP values for LightGBM** — surface concrete, specific drivers
      (e.g., "lag-24h and weekday dominate; temperature-correlated features
      matter less than expected")

---

## Phase 8 — Deep Learning Models

- [ ] LSTM forecasting model
- [ ] Hyperparameter tuning with Optuna + MLflow
- [ ] GRU model (optional)
- [ ] Temporal Fusion Transformer (optional stretch goal)
- [ ] Compare learning curves

---

## Phase 9 — Model Evaluation

- [ ] Rolling-origin backtesting
- [ ] Time-series cross-validation
- [ ] Compare MAE, RMSE, MAPE, and sMAPE
- [ ] **Diebold-Mariano test** for statistical significance between model pairs
- [ ] Error analysis by season
- [ ] Failure case analysis
- [ ] Model comparison report

---

## Phase 10 — Business Impact

- [ ] Translate forecast accuracy into operational impact
- [ ] Estimate potential energy cost savings
- [ ] Discuss utility demand planning improvements
- [ ] Explain operational trade-offs
- [ ] Recommend the best production model

---

### 🏁 Portfolio-complete checkpoint — pause here, write it up, share it

---

## Phase 11 — Production Pipeline

- [ ] Build reusable preprocessing pipeline
- [ ] Feature generation pipeline for inference
- [ ] Model serialization
- [ ] MLflow Model Registry
- [ ] Batch prediction pipeline
- [ ] Forecast scheduling simulation

---

## Phase 12 — API Development

- [ ] FastAPI REST API
- [ ] Pydantic request validation
- [ ] Prediction endpoint
- [ ] Health check endpoint
- [ ] Dockerized API

---

## Phase 13 — Dashboard

- [ ] Streamlit dashboard
- [ ] Interactive forecast visualization
- [ ] Actual vs prediction comparison
- [ ] Confidence interval visualization
- [ ] Anomaly visualization
- [ ] Model comparison dashboard
- [ ] Business KPI dashboard

---

## Phase 14 — Testing & MLOps

- [ ] Unit tests
- [ ] Integration tests
- [ ] Data validation tests
- [ ] GitHub Actions CI/CD
- [ ] Docker Compose
- [ ] Model versioning with MLflow
- [ ] Experiment reproducibility

---

## Phase 15 — Deployment

- [ ] Deploy FastAPI on Railway or Render
- [ ] Deploy Streamlit on Streamlit Community Cloud
- [ ] Publish MLflow experiment artifacts
- [ ] Configure monitoring dashboard
- [ ] Production architecture documentation

---

## Phase 16 — Portfolio & Documentation

- [ ] Professional README
- [ ] Architecture diagram
- [ ] Model comparison report
- [ ] Business impact report
- [ ] Production system design document
- [ ] Future improvements section
- [ ] LinkedIn project write-up