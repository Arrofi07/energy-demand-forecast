"""Train the production LightGBM direct multi-horizon bundle and register it
in the MLflow Model Registry as one pyfunc model.

Reuses the exact hyperparameters Optuna selected in
`06_model_prophet_lightgbm.ipynb` (tuned on h=24, 20 trials) -- but pins
`random_state` here, unlike the tuning notebook, which left LightGBM's row/
column subsampling unseeded. That means this script's bundle won't be
bit-for-bit identical to the notebook's models (subsample/colsample_bytree
< 1 introduce randomness the notebook never controlled for), but its
metrics should land very close -- and this version *is* exactly
reproducible from this script alone, which the research notebook never
needed to guarantee.
"""

from pathlib import Path

import duckdb
import mlflow
import pandas as pd

from src.evaluation.baselines import HORIZON
from src.evaluation.splits import get_train_test_split
from src.models.lightgbm_direct import fit_direct_models, prepare_features
from src.pipeline.forecaster import DirectMultiHorizonForecaster, ModelBundle, save_bundle

# Anchored to the project root, not the caller's cwd -- this module is
# called both from repo-root scripts and from notebooks/ (whose cwd is the
# notebooks/ directory). Without this, training from the notebook silently
# wrote a second, disconnected MLflow store to notebooks/mlruns/ instead of
# registering against the same registry a repo-root script would use.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "energy.duckdb"
MODEL_NAME = "energy-demand-lightgbm-direct"
EXPERIMENT_NAME = "energy-demand-forecasting"
TRACKING_URI = f"file:{PROJECT_ROOT / 'mlruns'}"

PRODUCTION_PARAMS = {
    "n_estimators": 371,
    "learning_rate": 0.02641663667830087,
    "num_leaves": 61,
    "max_depth": 8,
    "min_child_samples": 24,
    "subsample": 0.7290233432213763,
    "colsample_bytree": 0.995734826483326,
    "objective": "regression",
    "verbosity": -1,
    "random_state": 42,
}


def load_features_and_split(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.Timestamp]:
    con = duckdb.connect(str(db_path), read_only=True)
    hourly = con.execute("SELECT datetime, global_active_power FROM hourly ORDER BY datetime").df()
    features_raw = con.execute("SELECT * FROM hourly_features ORDER BY datetime").df()
    con.close()

    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    series = hourly.set_index("datetime")["global_active_power"].asfreq("h")

    features_raw["datetime"] = pd.to_datetime(features_raw["datetime"])
    features_df = features_raw.set_index("datetime")

    test_start = get_train_test_split(series, test_frac=0.2)
    return features_df, test_start


def train_production_bundle(lgb_params: dict = PRODUCTION_PARAMS) -> ModelBundle:
    """Trains through the same `test_start` boundary Phases 5-9 all used --
    not through "today" -- so this bundle can be validated against known,
    held-out actuals (see `src/pipeline/schedule_simulation.py`) before it's
    ever trusted on genuinely unseen data.
    """
    features_df, test_start = load_features_and_split()
    horizons = list(range(1, HORIZON + 1))

    models = fit_direct_models(features_df, train_end=test_start, horizons=horizons, lgb_params=lgb_params)
    _, _, feature_cols = prepare_features(features_df, horizons[0])

    return ModelBundle(
        models=models,
        horizons=horizons,
        feature_cols=feature_cols,
        trained_through=test_start,
        lgb_params=lgb_params,
    )


def register_bundle(bundle: ModelBundle, alias: str = "champion") -> str:
    """Log the bundle as one pyfunc model and register it under `MODEL_NAME`,
    so `mlflow.pyfunc.load_model('models:/energy-demand-lightgbm-direct@champion')`
    works from anywhere without needing the local joblib file.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="lightgbm_direct_production"):
        mlflow.log_params(bundle.lgb_params)
        mlflow.log_param("trained_through", str(bundle.trained_through))
        mlflow.log_param("horizons", f"1-{max(bundle.horizons)}")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DirectMultiHorizonForecaster(bundle),
            registered_model_name=MODEL_NAME,
        )

    version = model_info.registered_model_version
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, alias, version)

    model_uri = f"models:/{MODEL_NAME}@{alias}"
    print(f"Registered {MODEL_NAME} v{version}, alias '{alias}' -> {model_uri}")
    return model_uri


def train_and_register(save_local: bool = True, alias: str = "champion") -> str:
    bundle = train_production_bundle()
    if save_local:
        save_bundle(bundle)
    return register_bundle(bundle, alias=alias)


if __name__ == "__main__":
    train_and_register()
