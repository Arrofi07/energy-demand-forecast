# Energy Demand Forecasting — Project Roadmap

> **Notebook numbering:** `01_eda_trends_seasonality.ipynb`,
> `02_anomaly_detection.ipynb`, `03_feature_engineering.ipynb` (done). Next:
> `04_forecasting_baselines.ipynb`, `05_model_sarima.ipynb`,
> `06_model_prophet_lightgbm.ipynb`, `07_model_lstm.ipynb`,
> `08_model_comparison.ipynb`.

> **Portfolio-complete checkpoint:** Phases 1–10 form a complete, story-worthy
> portfolio piece on their own (data → EDA → anomalies → features → baselines →
> classical → ML/DL → evaluation → business impact). Phases 11–16 are the
> "production-grade" follow-up that demonstrates engineering maturity — treat
> them as a second milestone, not a blocker to sharing the work.

---

## Phase 1 — Data Engineering ✅

- [x] Download and validate the UCI Household Energy Consumption dataset
- [x] Build a reproducible data loading pipeline
- [x] Data cleaning and preprocessing pipeline
- [x] Missing value analysis and handling
- [x] Convert raw CSV to Parquet for efficient storage
- [x] Build a DuckDB analytical database
- [x] Data validation with **Pandera** (schema/type/range checks)

**Findings (missing-value gap analysis):** 1.25% of rows (25,979) have at least
one missing value, split across 71 distinct gaps. Bimodal distribution: 64
short gaps (<1h) account for only 1.7% of missing rows and are safely linearly
interpolated; 7 long gaps (14h–5 days) account for 98.3% of missing rows and
are left as `NaN` rather than interpolated, since they most likely represent
meter/logger outages, not zero consumption. Longest gaps: Aug 17–22 2010 (5
days), Sep 25–28 2010 (3.5 days), Apr 28–30 2007 (2.5 days). These 7 dates are
flagged as inputs to Phase 3 anomaly investigation.

---

## Phase 2 — Exploratory Data Analysis ✅

- [x] Data quality report (missing values, duplicates, outliers)
- [x] Exploratory data analysis notebook (`01_eda_trends_seasonality.ipynb`)
- [x] Daily consumption pattern analysis
- [x] Weekly seasonality analysis
- [x] Monthly and yearly trend analysis
- [x] STL decomposition (trend, seasonality, residual)
- [x] Autocorrelation (ACF) and Partial ACF analysis
- [x] Holiday and weekend consumption analysis
- [x] Correlation analysis between electrical variables
- [x] Interactive Plotly visualizations

**Findings:** No multi-year trend — consumption oscillates seasonally between
~0.7 kW (summer) and ~1.8-2.0 kW (winter) every year. Strong, stable weekly
(weekend > weekday) and monthly (heating-season) seasonality. Holiday effect
was essentially null (1.075 kW vs 1.092 kW, holiday vs regular). ACF/PACF
support a seasonal SARIMA order with m=7. STL residual spikes lined up
exactly with the known Phase 1 data gaps, not genuine anomalies — a useful
cross-check carried into Phase 3.

---

## Phase 3 — Anomaly Detection ✅

- [x] Z-score anomaly detection
- [x] Rolling IQR anomaly detection
- [x] STL residual anomaly detection
- [x] Isolation Forest anomaly detection (multivariate)
- [x] Investigate major anomaly events (root-cause each one)
- [x] Generate anomaly report (`02_anomaly_detection.ipynb`)

**Findings:** 27 days (~1.9%) flagged by 2+ of the 4 methods, none overlapping
the 8 known infrastructure gaps — clean separation between "missing data" and
"unusual-but-present data." Two distinct anomaly flavors: a high-power
weekend/winter cluster (cold snaps / gatherings) and a scattered low-power
cluster (likely short absences). Isolation Forest confirmed the high cluster
but not the low one. Decision: keep these days in training data, add an
`is_flagged_anomaly` indicator feature rather than dropping them.

---

## Phase 4 — Feature Engineering ✅

- [x] Time-based features (hour, weekday, month, season)
- [x] Lag features (1h, 6h, 24h, 168h) — hourly grain (`src/features/build_features.py`)
- [x] Rolling statistics (mean, std, min, max)
- [x] Difference and percentage change features (built leak-free, from
      already-lagged values only — not the current/target row)
- [x] Cyclical encoding (sin/cos for time variables)
- [x] Feature importance analysis (`03_feature_engineering.ipynb`, LightGBM +
      mutual information)

**Findings:** `lag_1h` dominates both importance measures by a wide margin
(short-horizon persistence effect). LightGBM split-count and mutual
information diverge meaningfully in two informative ways: cyclical hour
features (hour_sin/cos) rank high in LightGBM but low in MI (trees combine
them jointly; MI only sees each alone), while rolling min/max features rank
high in MI but low in LightGBM (informative but redundant given other
features already used). `is_flagged_anomaly` wasn't useful for 1-hour-ahead
prediction — worth re-testing at longer horizons. All lag/rolling/diff
features confirmed leak-free (built only from `shift()`-past values), so this
feature set is reusable as-is for the direct multi-horizon models in Phases
6-8 below.

---

## 📐 Forecast Horizon Design (decided before Phase 5)

**Phases 5-8 build a single trustworthy benchmark: 24-hour-ahead forecasting
on hourly data ("day-ahead" forecasting).** One question drives all of it:
*Can we construct a trustworthy forecasting benchmark for the actual business
problem — day-ahead demand forecasting?* No other horizons are introduced
until Phase 9 — this keeps Phases 5-8 focused on building each model well,
rather than splitting attention across horizons before any single model is
solid.

**Phase 9 introduces horizon sensitivity as its own investigation**, adding
1-hour and 7-day horizons specifically to answer two new questions:
*How does each model behave as the forecasting horizon increases? How robust
are different forecasting paradigms across increasingly difficult horizons?*
This is where the "comparison and reasoning over complexity" goal from the
start of the project really pays off — by then every model already has a
track record at 24h, so the horizon experiment is testing something real
rather than being tacked on for its own sake.

**Per-model multi-step strategy** (this distinction is itself worth a
paragraph in the writeup):
- **SARIMA / Prophet:** native multi-step support, no extra engineering.
  Prophet doesn't depend on recent lags, so it doesn't compound error the
  way autoregressive methods can — a direct point of contrast with SARIMA.
- **LightGBM:** **direct** multi-horizon strategy — train 24 separate models
  (one per horizon offset h=1..24), each using only origin-time features
  (already leak-free from Phase 4). Chosen over **recursive** (predict h=1,
  feed forward, repeat) specifically to avoid compounding error and keep the
  comparison against SARIMA/Prophet fair.
- **LSTM:** direct sequence-to-sequence — encode the recent window, decode
  all 24 output values in one shot, same reasoning as LightGBM's direct
  approach.

**Evaluation (Phases 5-8):** rolling-origin backtesting — many forecast
origins across the test period, 24 steps ahead from each, error recorded
**both aggregated and broken down by horizon step (h=1...24)**. The per-step
error curve, plotted per model, is the centerpiece chart for the 24h
benchmark itself.

**Evaluation (Phase 9 addition):** re-run the same models at 1h and 7d
horizons, then compare error *and model ranking* across all three horizons —
does the model that wins at 24h still win at 1h and 7d, or does the best
choice change with horizon? That comparison is the actual deliverable of
Phase 9's horizon sensitivity work.

---

## Phase 5 — Forecasting Baselines ✅

- [x] Naive Forecast (persistence: last known value)
- [x] Seasonal Naive Forecast (same hour, 24h and 168h ago)
- [x] Moving Average Forecast — superseded by testing both seasonal naive
      variants directly (see finding below)
- [x] Benchmark all baseline methods **at the 24h centerpiece horizon**
      (`04_forecasting_baselines.ipynb`)

**Findings:** Contrary to the hypothesis, **daily seasonal naive (lag 24h)
beat weekly seasonal naive (lag 168h) on every metric** (MAE 0.530 vs 0.563,
RMSE 0.786 vs 0.813, MAPE 67.3% vs 76.7%, sMAPE 49.5% vs 53.7%) — recency
apparently matters more than exact day-of-week matching at this horizon,
since the week-old reference point has more time to drift than the day-old
one. **Daily seasonal naive (MAE 0.530, RMSE 0.786) is the actual bar every
later model must beat**, not weekly as originally expected. MAPE/sMAPE are
unreliable for this dataset (49-77%, inflated by low-consumption overnight
hours near zero) — MAE/RMSE are the trustworthy metrics going forward.
Shared `src/evaluation/` infrastructure (train/test split, backtest harness,
metrics) built here will be reused by every model through Phase 9.

---

## Phase 6 — Statistical Forecasting ✅

- [x] Stationarity testing (ADF & KPSS)
- [x] ARIMA implementation
- [x] SARIMA implementation — 24h-ahead centerpiece, native multi-step
- [x] Hyperparameter tuning (auto_arima)
- [x] Residual diagnostics
- [x] MLflow experiment tracking

**Findings:** auto_arima selected ARIMA(3,1,0)(2,0,0)[24] (d=1, no seasonal
differencing needed). SARIMA's AIC (8531.4) decisively beat plain ARIMA's
(8994.5) — the seasonal term clearly earns its complexity on fit quality.
Against the Phase 5 baseline (daily seasonal naive: MAE 0.530, RMSE 0.786),
results were genuinely mixed: SARIMA's RMSE (0.776) edged out the baseline
(fewer large misses) but its MAE (0.560) was worse (less precise on typical
hours) — an honest interpretability/robustness trade-off, not a clean win.
Plain ARIMA was worse than both on every metric. Ljung-Box test showed
residuals still significantly autocorrelated (likely the un-modeled weekly
seasonality, m=24 only for tractability) — a known, accepted limitation
carried forward into the Phase 7/8 comparison.

---

## Phase 7 — Machine Learning Models

- [ ] Prophet implementation — 24h-ahead centerpiece, native multi-step
- [ ] `statsforecast` AutoARIMA/AutoETS (modern Prophet-replacement comparison)
- [ ] **LightGBM forecasting (primary gradient-boosting model)** — direct
      multi-horizon strategy, 24 models (one per h=1..24), origin-time
      features only (reuses Phase 4 feature set as-is)
- [ ] XGBoost / Random Forest — optional ablation only (brief note: "tried X,
      saw negligible difference vs LightGBM, here's why") rather than a fully
      parallel track
- [ ] Hyperparameter tuning with Optuna + MLflow
- [ ] Feature importance comparison
- [ ] **SHAP values for LightGBM** — surface concrete, specific drivers

---

## Phase 8 — Deep Learning Models

- [ ] LSTM forecasting model — direct sequence-to-sequence (encode recent
      window, decode all 24 hourly values in one shot), 24h-ahead centerpiece
- [ ] Hyperparameter tuning with Optuna + MLflow
- [ ] GRU model (optional)
- [ ] Temporal Fusion Transformer (optional stretch goal)
- [ ] Compare learning curves

---

## Phase 9 — Model Evaluation

**Part A — 24h benchmark evaluation:**
- [ ] Rolling-origin backtesting — many forecast origins, 24 steps ahead each
- [ ] **Error-vs-horizon-step curve** (h=1...24, per model) within the 24h
      benchmark itself
- [ ] Time-series cross-validation
- [ ] Compare MAE, RMSE, MAPE, and sMAPE
- [ ] **Diebold-Mariano test** for statistical significance between model pairs
- [ ] Error analysis by season
- [ ] Failure case analysis

**Part B — Horizon sensitivity investigation** (new horizons introduced here,
not before): re-run each model (or the strongest subset — SARIMA, Prophet,
LightGBM required; LSTM at 7d/daily grain optional given setup cost) at 1h
and 7d horizons, guided by two questions:
- [ ] *How does each model's error behave as the forecasting horizon
      increases?* — 1h vs 24h vs 7d error comparison per model
- [ ] *How robust are different forecasting paradigms across increasingly
      difficult horizons?* — does the model that wins at 24h still win at 1h
      and 7d, or does the best choice change with horizon?
- [ ] Model comparison report (24h benchmark + horizon sensitivity findings
      together)

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