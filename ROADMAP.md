# Energy Demand Forecasting — Project Roadmap

> **Notebook numbering:** `01_eda_trends_seasonality.ipynb`,
> `02_anomaly_detection.ipynb`, `03_feature_engineering.ipynb`,
> `04_forecasting_baselines.ipynb`, `05_model_sarima.ipynb`,
> `06_model_prophet_lightgbm.ipynb`, `07_model_lstm.ipynb`,
> `08_model_comparison.ipynb`, `09_horizon_sensitivity.ipynb`,
> `10_business_impact.ipynb` (done — Phases 1–10 complete),
> `11_production_pipeline.ipynb` (done — Phase 11 complete).

> **Portfolio-complete checkpoint reached:** Phases 1–10 form a complete,
> story-worthy portfolio piece (data → EDA → anomalies → features → baselines →
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

## Phase 7 — Machine Learning Models ✅ (core scope; 2 items deferred)

- [x] Prophet implementation — 24h-ahead centerpiece, native multi-step
- [ ] `statsforecast` AutoARIMA/AutoETS — deferred, not blocking; Prophet +
      LightGBM already give two structurally different paradigms to compare
- [x] **LightGBM forecasting (primary gradient-boosting model)** — direct
      multi-horizon strategy, 24 models (one per h=1..24), origin-time
      features only (reuses Phase 4 feature set as-is)
- [ ] XGBoost / Random Forest — optional ablation, deferred
- [x] Hyperparameter tuning with Optuna + MLflow — tuned once on h=24 (20
      trials, time-based validation split), reused across all 24 direct
      models
- [x] Feature importance comparison — LightGBM gain, h=1 vs h=24
- [x] **SHAP values for LightGBM** — h=24 model, surfaces concrete drivers

**Findings (`06_model_prophet_lightgbm.ipynb`):** **LightGBM is the first
model in this project to decisively beat the Phase 5 baseline on every
metric** (MAE 0.432 vs. 0.530, RMSE 0.593 vs. 0.786, sMAPE 44.4% vs. 49.5%)
— an ~18% MAE / ~25% RMSE cut, not the mixed win SARIMA managed. Prophet also
beat the baseline on MAE/RMSE (0.485 / 0.637) and both ML models beat SARIMA
outright. **Caveat:** LightGBM's backtest covers only 89.1% of test origins
(6,156/6,912) vs. 96.7% for Prophet/SARIMA — `direct_multi_horizon_backtest`
drops any origin with an NaN engineered feature, and the 168h-rolling-window
features stay contaminated for up to a week after each Phase-1 data gap. The
margin is large enough that this almost certainly doesn't flip the ranking,
but a shared gap-free origin set should be used before Phase 9 treats these
numbers as final. Feature importance confirms the model changes strategy
with horizon exactly as hypothesized: `lag_1h` dominates at h=1 (matching
Phase 4), while `lag_24h` and weekly-cycle features (`roll_mean_168h`,
`lag_168h`) take over at h=24 — the weekly signal SARIMA structurally lacked
(m=24 only). **Non-obvious finding:** MAE vs. horizon-step is not monotonic
for any of the three models — a sharp double-peaked curve with spikes at
h=7 (7am) and h=19-21 (7-9pm), which Prophet's own daily-seasonality
component explains: those are exactly its two humps (morning
routine/breakfast, evening cooking/heating) — the most behavior-driven,
least routine-predictable hours of the day, hard for every paradigm alike.

---

## Phase 8 — Deep Learning Models ✅ (core scope; 2 optional items deferred)

- [x] **LSTM forecasting model** — direct sequence-to-sequence (encoder LSTM
      reads the recent lookback window once, a linear head maps its final
      hidden state to all 24 hourly values in one shot — no autoregressive
      decoding), 24h-ahead centerpiece. Inputs deliberately minimal: raw
      target + cyclical calendar encodings only, no hand-engineered lags —
      the model is expected to learn temporal structure itself.
- [x] Hyperparameter tuning with Optuna (30 trials: lookback, hidden size,
      depth, dropout, learning rate) + MLflow
- [ ] GRU model — deferred, not blocking; LSTM already gives a second DL
      paradigm to compare against LightGBM's feature-engineered approach
- [ ] Temporal Fusion Transformer — optional stretch goal, deferred
- [x] Compare learning curves — train vs. validation loss per epoch

**Findings (`07_model_lstm.ipynb`):** LSTM clearly beats the Phase 5
baseline, Prophet, and SARIMA on every metric (MAE 0.440 vs. 0.530, RMSE
0.605 vs. 0.786, sMAPE 45.6% vs. 49.5%) — a genuine win in the same tier as
LightGBM, not SARIMA's mixed result — but LightGBM still edges it out on
MAE/RMSE/sMAPE (0.429 / 0.591 / 44.0%); LSTM actually posts the best MAPE of
all five methods (58.7%), a flip worth noting given this project's standing
distrust of MAPE/sMAPE on this dataset. Optuna (30 trials) settled on a 48h
lookback, 1 layer, hidden size 105, dropout 0.256 — the short lookback
winning over 72h/168h options echoes Phase 5's finding that daily lag beat
weekly lag for the naive baselines too; widening the search from an earlier
12-trial run to 30 trials changed the winning configuration but not the
model's standing versus the other four methods. **Honest caveat:** the
learning curve shows near-immediate overfitting — validation loss bottoms
out within 2-3 epochs, then climbs steadily while training loss keeps
falling — a real limitation of applying even a single-layer, 105-hidden-unit
LSTM to a single ~27k-hour series, unlike LightGBM's tree-based resistance
to overfitting. Early stopping correctly restored the best-seen (early)
checkpoint, which is why the backtest numbers above still hold up. Coverage
caveat, smaller than LightGBM's: LSTM's 48h lookback only needs to be
NaN-free 48h before an origin (vs. up to 168h for LightGBM's rolling
features), so it covers 94.1% of test origins (6,504/6,912) — between
LightGBM's 89.1% and Prophet/SARIMA's 96.7%. The error-vs-horizon-step
double-peak (spikes at h=7 and h=19-21) reappears for LSTM too, now
confirmed across four structurally different models — strong evidence it's
a property of the data (behavior-driven morning/evening variability), not
any one model's architecture. **Bottom line for Phase 9:** a paradigm
needing zero manual feature engineering gets within ~3% MAE of the
feature-engineered gradient-boosting model on this dataset, comfortably
clearing every classical/baseline method. Phase 9's Diebold-Mariano test
later confirmed the LightGBM-vs-LSTM gap is not statistically significant
(p=0.098) — LightGBM remains the practical pick, but the margin is closer
than the point-estimate MAE alone suggests.

---

## Phase 9 — Model Evaluation ✅

**Part A — 24h benchmark evaluation:**
- [x] Rolling-origin backtesting — many forecast origins, 24 steps ahead each
      (reuses the Phase 5-8 saved results as-is, no refitting)
- [x] **Error-vs-horizon-step curve** (h=1...24, per model) within the 24h
      benchmark itself
- [x] Time-series cross-validation — 4 contiguous chronological folds over
      the test period, as a walk-forward ranking-stability check
- [x] Compare MAE, RMSE, MAPE, and sMAPE
- [x] **Diebold-Mariano test** for statistical significance between model pairs
- [x] Error analysis by season
- [x] Failure case analysis

**Findings (`08_model_comparison.ipynb`):** LightGBM (MAE 0.429) and LSTM
(0.440) are well clear of Prophet (0.485), the Phase 5 baseline (0.530),
SARIMA (0.560), weekly seasonal naive (0.563), ARIMA (0.631), and flat
persistence (0.664) — but the LightGBM-vs-LSTM gap itself is **not**
statistically significant (Diebold-Mariano p=0.098); every other pairwise
comparison is significant at p<0.001. Time-series CV (4 chronological
folds) shows LightGBM #1 and LSTM #2 in *every* fold with no reversal, but
Prophet/SARIMA/baseline reshuffle by season — most strikingly, the naive
baseline beats both SARIMA and Prophet in the summer fold, when low,
stable demand makes yesterday's value a near-unbeatable forecast. Per-
season MAE sharpens Phase 6's "mixed" SARIMA verdict: SARIMA loses to the
trivial baseline in 3 of 4 seasons (winter, spring, summer), not just
"sometimes." Failure case analysis found all five methods missing the same
handful of real events (a Nov 20 evening spike, the Feb 21 and Oct 18
flagged-anomaly days), but LightGBM and the baseline have proportionally
fewer worst-case failures landing on already-flagged anomaly days (20%)
than LSTM/Prophet (40%) — concrete evidence that LightGBM's Phase 4
`is_flagged_anomaly` feature earns its keep. New reusable module:
`src/evaluation/comparison.py` (season labeling, CV-fold assignment,
Diebold-Mariano test, worst-case extraction).

**Part B — Horizon sensitivity investigation ✅** (new horizons introduced
here, not before): re-run each model (or the strongest subset — SARIMA,
Prophet, LightGBM required; LSTM at 7d skipped, exactly as flagged optional
given setup cost — a 168-step direct decode head is a different
architecture, not a parameter change) at 1h and 7d horizons, guided by two
questions:
- [x] *How does each model's error behave as the forecasting horizon
      increases?* — 1h vs 24h vs 7d error comparison per model
- [x] *How robust are different forecasting paradigms across increasingly
      difficult horizons?* — does the model that wins at 24h still win at 1h
      and 7d, or does the best choice change with horizon?
- [x] Model comparison report (24h benchmark + horizon sensitivity findings
      together)

**Findings (`09_horizon_sensitivity.ipynb`):** 1h reuses the existing 24h-
benchmark results at `horizon_step == 1` (zero refitting); 7d (168h) needed
genuinely new backtests, reusing Phases 6-7's tuned hyperparameters/order
(no re-search) with `origin_freq="7d"` (41 non-overlapping origins).
**LightGBM and LSTM are the only methods that stay near the top at every
horizon** (LSTM 0.207 vs. LightGBM's 0.211 at 1h, essentially tied at 24h,
LightGBM 0.430 at 7d with LSTM not evaluated there) — the clearest answer to
"does the 24h winner keep winning": yes. **SARIMA is the sharpest
reversal:** competitive-ish at 1h, unremarkable at 24h, but the outright
*worst* method of any kind at 7d (0.800, worse than trivial persistence) —
its daily-only (m=24) seasonal structure doesn't extrapolate a full week
out. **Prophet moves the opposite direction:** one of the weakest methods
at 1h, but clearly 2nd-best (behind only LightGBM) at 7d. Each naive
baseline peaks almost exactly at the horizon matching its own lag. **A real
bug was caught and fixed along the way:** `daily_seasonal_naive`'s fixed
24h lag silently produces future-leaking forecasts whenever horizon > 24h
(its farthest-step lookback lands after the origin) — this only surfaced
because Part B is the first time this project asked a method for a horizon
longer than its own lag. `src/evaluation/baselines.py`'s `_lagged_forecast`
now raises `ValueError` if `horizon > lag_hours`. Rather than just drop
`daily_seasonal_naive` from the 7d comparison, a leak-free
`extended_daily_seasonal_naive` closes the gap (tiles the most recent 24h
block forward instead of looking up a different day-ago point per step) —
mathematically identical to the original whenever horizon<=24 (confirmed to
6 decimal places, not just approximated), so it's a genuine extension of
Phase 5's "recency" baseline, not a new one. It answers the question Phase
5 couldn't: does "daily lag beats weekly lag" hold at 7d too, or was it a
24h-specific artifact? **It holds** — its aggregate 7d MAE (0.552) still
edges out `weekly_seasonal_naive`'s (0.564), extending Phase 5's finding
rather than overturning it. (At the single exact h=168 step, `naive`,
`weekly_seasonal_naive`, and `extended_daily_seasonal_naive` are
numerically tied — a neat, confirmed structural fact: 168=24×7, so every
lag-based baseline whose lag divides the horizon reduces to "the value at
the origin" at that one specific step; their aggregate curves still
differ.) LSTM was deliberately not extended to 7d: unlike SARIMA/Prophet/
LightGBM, its output head is sized at construction time, so a 7d version
needs a new architecture and a full retrain, not a parameter change —
disproportionate cost given Part A already found LightGBM and LSTM
statistically indistinguishable at 24h, and LightGBM's own 7d extension
answers the "does the ML/DL tier hold at 7d" question on its own. **Bottom
line for Phase 10:** the practical recommendation is horizon-dependent —
LightGBM/LSTM are the safe default throughout, but if a single classical
fallback were needed, Prophet is the better pick at longer horizons and
SARIMA the worse one, the opposite of what their similar 24h standing alone
would have suggested.

---

## Phase 10 — Business Impact ✅

- [x] Translate forecast accuracy into operational impact
- [x] Estimate potential energy cost savings
- [x] Discuss utility demand planning improvements
- [x] Explain operational trade-offs
- [x] Recommend the best production model

**Findings (`10_business_impact.ipynb`):** reframed Phase 9's two error
metrics by what they actually govern operationally — MAE drives expected
imbalance settlement cost, RMSE drives required reserve margin (it
penalizes the large-miss tail, which Phase 9's failure-case analysis
already showed clusters on real anomalous days) — rather than treating them
as interchangeable "accuracy" numbers. LightGBM's real, recomputed-live
improvement over the Phase 5 baseline (MAE −19.2%, RMSE −24.8%) was scaled,
through an explicit, adjustable assumption cell (10,000-household
portfolio, $20/MWh imbalance premium — clearly labeled illustrative, not a
validated utility ROI figure, and likely conservative since it doesn't
model error-cancellation across a real pooled portfolio), into an
illustrative ≈$178k/year avoided-imbalance-cost estimate. Demand-planning
implications tie directly back to earlier phases: Phase 9B's finding that
SARIMA collapses at 7d means week-ahead planning should never lean on it
despite being usable at 24h; Phase 9A's per-season breakdown means reserve
margins sized on annual-average error would under-cover winter and
over-cover summer. Operational trade-offs center on one point from Phase
9A: since the LightGBM-vs-LSTM accuracy gap isn't statistically significant
(DM p=0.098), LSTM's materially higher retraining cost and its need for a
new architecture (not just a parameter change) per horizon are costs paid
for no proven accuracy benefit. **Recommendation:** LightGBM as the primary
production model (best/tied-best accuracy at every horizon, cheapest to
retrain, only one proven to extend cleanly across horizons, built-in SHAP
explainability); Prophet as a secondary choice specifically for
longer-horizon planning conversations with non-technical stakeholders;
SARIMA not recommended for production; LSTM judged a validated research
result (raw sequence learning matches feature-engineered GBM) rather than
a production pick, given no significant edge over LightGBM's lower
operational cost.

---

### 🏁 Portfolio-complete checkpoint reached — Phases 1–10 done

---

## Phase 11 — Production Pipeline ✅

- [x] Build reusable preprocessing pipeline (`src/pipeline/preprocessing.py`
      -- orchestrates the existing Phase 1-4 download/clean/DuckDB/feature
      steps into one idempotent call)
- [x] Feature generation pipeline for inference (`src/pipeline/inference_features.py`
      -- builds one origin's feature row from a bounded ~192h window instead
      of the full history)
- [x] Model serialization (`src/pipeline/forecaster.py` -- `ModelBundle` +
      joblib round-trip for the 24 direct-horizon LightGBM models)
- [x] MLflow Model Registry (`src/pipeline/registry.py` -- trains and
      registers a single pyfunc-wrapped model under the `champion` alias)
- [x] Batch prediction pipeline (`src/pipeline/predict.py`)
- [x] Forecast scheduling simulation (`src/pipeline/schedule_simulation.py`)

**Findings (`11_production_pipeline.ipynb`):** **the production pipeline
reproduces the Phase 7 research result closely, not approximately** -- MAE
0.432 (production, via `simulate_daily_schedule` walking the real DuckDB ->
inference-features -> serialized-model -> forecast path) vs. 0.429
(notebook 06's in-memory backtest), a 0.9% relative gap, identical
`n=6,156` and identical 29/288 skipped origins for the same NaN-feature
reason. The gap is fully explained, not just close-enough: `registry.py`
deliberately pins `random_state=42` where the Optuna-tuned notebook left
LightGBM's row/column subsampling unseeded, so the two are honestly
different (but equally valid) fits of the same hyperparameters, not a
research/production logic mismatch -- both call the same `prepare_features`
function underneath. **A real, previously-unenumerated bug was caught
while building the validation itself**: picking a timestamp to test
`build_latest_features` against the full-history `hourly_features` table
required finding one genuinely free of NaN inputs, which surfaced a
~51-hour gap (2009-06-13 04:00 to 2009-06-15 06:00) that Phase 1's findings
never listed individually (only the 3 *longest* of 7 total long gaps were
named) -- a small, honest reminder that "documented" and "exhaustively
documented" aren't the same thing, four phases later. **A second real bug
was caught and fixed mid-phase**: `DEFAULT_BUNDLE_PATH` and MLflow's
tracking URI were both bare relative paths, so training from the notebook
(cwd=`notebooks/`) silently created a second, disconnected model bundle and
registry under `notebooks/` instead of reusing the repo-root one a script
would use -- fixed by anchoring every pipeline path to the project root
(`Path(__file__).resolve().parent.parent.parent`, matching
`results_store.py`'s existing pattern) and pinning MLflow's tracking URI
explicitly in `registry.py`. **A third, related bug turned up the same way:**
`run_preprocessing_pipeline()` orchestrates the pre-existing Phase 1-4
download/clean/DuckDB/feature modules, which all use paths relative to the
project root under the same "always run from repo root" assumption --
calling it from the notebook silently wrote a complete second copy of the
entire dataset (raw zip through `hourly_features`) under `notebooks/data/`.
Fixed with a `_in_project_root()` context manager that `os.chdir`s to the
project root for the duration of the four orchestrated steps and always
restores the caller's original directory afterward, even on error --
scoped to `preprocessing.py` itself rather than retrofitting path-anchoring
into four Phase 1-4 modules that have worked correctly under the
CLI-from-repo-root convention for ten phases already. **Known, stated
limitation carried forward:**
`is_flagged_anomaly` is Phase 3's retrospective, hardcoded gap-investigation
list -- always 0 for any date that didn't exist when Phase 3 ran, i.e. every
real future production date. Phase 4 already found this feature wasn't a
strong 1-hour-ahead predictor, so it doesn't undermine current numbers, but
a genuinely live system would need an online anomaly detector for this
feature to mean anything going forward. 23 new tests added under
`tests/pipeline/`, including a `slow`-marked integration test that verifies
the bounded-window feature reconstruction is bit-for-bit identical to the
full-history table, and a categorical-dtype regression test confirming
LightGBM correctly realigns category codes when an inference window sees
fewer category values than training did.

---

## Phase 12 — API Development ✅

- [x] FastAPI REST API (`src/api/main.py`)
- [x] Pydantic request validation (`src/api/schemas.py` -- `ForecastRequest`,
      malformed `as_of` rejected with a 422 before the route body even runs)
- [x] Prediction endpoint (`POST /forecast`)
- [x] Health check endpoint (`GET /health`)
- [x] Dockerized API (`Dockerfile`, built and run for real -- see findings)

**Findings:** deliberately thin -- the API duplicates none of Phase 11's
forecasting logic, it just translates HTTP into calls against
`src/pipeline/inference_features.py` and `forecaster.py`. `get_bundle` /
`get_forecaster` are FastAPI dependencies (not plain module state), so the
9 tests in `tests/api/test_main.py` inject a tiny in-memory bundle via
`app.dependency_overrides` instead of needing the real ~47MB production
bundle on disk -- the test suite stays hermetic regardless of whether
`registry.py` has ever been run in a given environment. **Verified live,
not just via `TestClient`:** started the real app with `uvicorn`, then
`curl`'d every path by hand -- `/health` returned the actual registered
model's `trained_through` and the dataset's real latest timestamp;
`POST /forecast` returned a live 24h forecast (peaking in the evening hours,
same double-peak shape found back in Phase 7/9); a malformed `as_of` came
back 422 before reaching any forecasting code; an `as_of` too early in the
dataset for a full lookback window came back 400 with the exact NaN-column
message `inference_features.py` raises, not a generic error. **Dockerized
and actually run, not just written -- which caught a real bug on the
first attempt:** `docker build` succeeded cleanly, but the container
crashed immediately on startup with `OSError: libgomp.so.1: cannot open
shared object file` -- `python:3.12-slim` doesn't ship the GNU OpenMP
runtime LightGBM's compiled extension links against at import time. A
successful build says nothing about whether the image actually runs; only
starting the container and curling it caught this. Fixed with one line
(`apt-get install libgomp1`); after the fix, `docker run -p 8000:8000`
serves byte-identical `/health` and `/forecast` responses to the bare-metal
`uvicorn` run, confirmed by curling both directly. Deliberate scope choice:
the model bundle and `energy.duckdb` are baked
into the image (self-contained `docker run`, no volume setup needed for
this project's scope) rather than mounted or pulled from the MLflow
registry at container startup -- the natural upgrade once Phase 15
(Deployment) needs the served model to be updatable without a rebuild.

---

## Phase 13 — Dashboard ✅

- [x] Streamlit dashboard (`src/dashboard/app.py` + 4 pages, multipage app)
- [x] Interactive forecast visualization (`pages/1_Live_Forecast.py`)
- [x] Actual vs prediction comparison (same page -- overlays known actuals
      where the forecast window falls inside the historical data)
- [x] Confidence interval visualization (empirical 10th-90th percentile
      band from the real Phase 7 backtest residuals, shaded around the
      live forecast -- labeled clearly as empirical, not model-native,
      since this LightGBM setup produces point forecasts only)
- [x] Anomaly visualization (`pages/3_Anomaly_Detection.py` -- reuses Phase
      3's already-flagged 27 dates from `build_features.py` rather than
      re-running detection live)
- [x] Model comparison dashboard (`pages/2_Model_Comparison.py` -- 24h
      leaderboard + error-vs-horizon chart, plus a toggle for the Phase 9B
      1h/24h/7d horizon-sensitivity comparison)
- [x] Business KPI dashboard (`pages/4_Business_Impact.py` -- an
      interactive version of Phase 10's illustrative savings estimate,
      with portfolio size and imbalance premium as live sliders instead of
      a fixed notebook cell)

**Findings:** every number on every page is computed live from
`data/results/` or the production model bundle via
`src/dashboard/data_access.py` (cached with `st.cache_data`/
`st.cache_resource`) -- nothing is hardcoded from the README, so the
dashboard can't silently drift from the actual saved artifacts. Confirmed
directly: at the default assumptions (10,000 households, $20/MWh), the
Business Impact page's live-computed estimate landed on **$178,394/year**,
matching the README's Phase 10 figure exactly. **A real bug was caught by
actually launching the app, not by trusting the 8 passing `AppTest`-based
unit tests** (`tests/dashboard/test_pages.py`): `streamlit run
src/dashboard/app.py` executes the file directly rather than via `python
-m`, so Python only puts `src/dashboard/` on `sys.path`, not the project
root -- every `from src...` import failed with `ModuleNotFoundError` the
moment a real browser hit the real server, even though `AppTest` (which
runs under pytest, where the project root is already importable) never
caught it. Fixed with an explicit `sys.path` bootstrap at the top of
`app.py` and every page file, anchored to `Path(__file__).resolve()` rather
than assuming a launch-time cwd. Verified end-to-end with a real headless
Chromium session (Playwright) driving the actual running server: navigated
every page, clicked "Generate forecast," toggled both Model Comparison
views, and captured screenshots -- zero console/page errors, and every
chart/table matches the numbers already reported in earlier phases.

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