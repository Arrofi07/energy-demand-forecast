"""Model bundle serialization plus a single deployable forecaster wrapping
all 24 direct multi-horizon LightGBM models behind one `.predict()` call.

`fit_direct_models` (Phase 7, `src/models/lightgbm_direct.py`) returns a
plain `{horizon: LGBMRegressor}` dict -- convenient for a notebook, but not
something a batch job or an MLflow-registered model can hand around on its
own (it doesn't carry the feature-column order or which horizons it covers).
`ModelBundle` packages that dict with everything needed to use it correctly
elsewhere; `DirectMultiHorizonForecaster` is the MLflow-pyfunc-compatible
wrapper that turns "24 models" into "one model" from a caller's perspective.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src.models.lightgbm_direct import prepare_features

# Anchored to the project root (not the caller's cwd) -- this module is
# imported from both repo-root scripts and notebooks/ (whose cwd is the
# notebooks/ directory), and a bare relative path silently produced two
# disconnected bundle files depending on which one called it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / "lightgbm_direct_bundle.joblib"


@dataclass
class ModelBundle:
    """Everything needed to reproduce Phase 7's LightGBM predictions later,
    outside the notebook that trained them.
    """

    models: dict          # {horizon: LGBMRegressor}
    horizons: list[int]
    feature_cols: list[str]
    trained_through: pd.Timestamp
    lgb_params: dict
    target_col: str = "global_active_power"


def save_bundle(bundle: ModelBundle, path: Path = DEFAULT_BUNDLE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"Saved model bundle ({len(bundle.models)} horizons, trained through "
          f"{bundle.trained_through}) -> {path}")


def load_bundle(path: Path = DEFAULT_BUNDLE_PATH) -> ModelBundle:
    return joblib.load(Path(path))


class DirectMultiHorizonForecaster(mlflow.pyfunc.PythonModel):
    """Wraps a `ModelBundle` so the 24 independent per-horizon models behave
    as one deployable forecasting model with a single `.predict()` call.

    `model_input` is expected to have one row per forecast origin (a
    DatetimeIndex) with columns matching the `hourly_features` schema --
    exactly what `src/pipeline/inference_features.py` produces. Internally
    this reuses `prepare_features` from Phase 7 for every horizon, so the
    origin-time -> per-horizon feature transform is defined in exactly one
    place, whether it's being trained, backtested, or served.
    """

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        records = []
        for h in self.bundle.horizons:
            X, _, _ = prepare_features(model_input, h)
            X = X[self.bundle.feature_cols]
            preds = self.bundle.models[h].predict(X)
            for origin, forecast in zip(model_input.index, preds):
                records.append({
                    "origin": origin,
                    "horizon_step": h,
                    "forecast": float(forecast),
                })
        return (
            pd.DataFrame.from_records(records)
            .sort_values(["origin", "horizon_step"])
            .reset_index(drop=True)
        )
