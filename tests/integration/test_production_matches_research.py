"""End-to-end integration test: does the real production code path
(train -> save -> DuckDB -> bounded-window inference features -> serialized
model -> forecast) reproduce Phase 7's research backtest?

This formalizes the validation done by hand in
`11_production_pipeline.ipynb` as a permanent regression test. That
notebook found production MAE (0.432) within 0.9% of the research MAE
(0.429), with the gap fully explained by `registry.py` pinning
`random_state=42` where the Optuna-tuned notebook left LightGBM's
row/column subsampling unseeded. If a future change to `src/pipeline/` or
`src/features/` ever breaks that agreement -- silently, in a way no unit
test in isolation would catch -- this test is what catches it, not a
manual re-check.
"""

import duckdb
import pandas as pd
import pytest

from src.evaluation.baselines import HORIZON, compute_metrics, get_forecast_origins
from src.evaluation.results_store import load_results
from src.evaluation.splits import get_train_test_split
from src.pipeline.forecaster import save_bundle
from src.pipeline.inference_features import DB_PATH
from src.pipeline.registry import PRODUCTION_PARAMS, train_production_bundle
from src.pipeline.schedule_simulation import simulate_daily_schedule

# The notebook found a 0.9% gap explained entirely by an unseeded vs. pinned
# random_state -- 2% gives headroom for that expected noise without masking
# a genuine regression, which would be a much larger jump.
MAX_ALLOWED_RELATIVE_MAE_GAP = 0.02


@pytest.mark.slow
def test_schedule_simulation_reproduces_research_backtest_closely(tmp_path):
    bundle = train_production_bundle(lgb_params=PRODUCTION_PARAMS)
    bundle_path = tmp_path / "bundle.joblib"
    save_bundle(bundle, bundle_path)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    hourly = con.execute("SELECT datetime, global_active_power FROM hourly ORDER BY datetime").df()
    con.close()
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    series = hourly.set_index("datetime")["global_active_power"].asfreq("h")

    test_start = get_train_test_split(series, test_frac=0.2)
    origins = get_forecast_origins(series, test_start, horizon=HORIZON)

    sim_results = simulate_daily_schedule(origins, bundle_path=bundle_path)
    sim_results = sim_results.copy()
    sim_results["actual"] = series.reindex(
        sim_results["origin"] + pd.to_timedelta(sim_results["horizon_step"], unit="h")
    ).values

    production_metrics = compute_metrics(sim_results)
    research_metrics = compute_metrics(load_results("lightgbm_direct"))

    # Same NaN-feature guard logic, applied one origin at a time instead of
    # vectorized over the whole test period -- coverage should match exactly.
    assert production_metrics["n"] == research_metrics["n"]

    relative_gap = abs(production_metrics["MAE"] - research_metrics["MAE"]) / research_metrics["MAE"]
    assert relative_gap < MAX_ALLOWED_RELATIVE_MAE_GAP, (
        f"production MAE {production_metrics['MAE']:.4f} diverged from research MAE "
        f"{research_metrics['MAE']:.4f} by {relative_gap:.1%} -- expected < "
        f"{MAX_ALLOWED_RELATIVE_MAE_GAP:.0%}. If this fails, check whether a change "
        "to src/pipeline/inference_features.py or src/features/build_features.py "
        "has made the production feature computation diverge from the research path, "
        "rather than assuming it's just random_state noise."
    )
