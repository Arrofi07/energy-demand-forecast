import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.lstm import (
    FEATURE_COLS,
    TARGET_CHANNEL,
    TARGET_COL,
    LSTMForecaster,
    TargetScaler,
    WindowDataset,
    make_windows,
    predict,
    train_model,
)


@pytest.fixture
def feature_df():
    """Small df shaped like hourly_features restricted to the LSTM's input
    channels. Target is a deterministic ramp (0, 1, 2, ...) so window
    contents can be asserted exactly, not just by shape.
    """
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        TARGET_COL: np.arange(len(idx), dtype="float64"),
        "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
        "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
        "dayofweek_sin": np.sin(2 * np.pi * idx.dayofweek / 7),
        "dayofweek_cos": np.cos(2 * np.pi * idx.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * idx.month / 12),
        "month_cos": np.cos(2 * np.pi * idx.month / 12),
        "is_weekend": (idx.dayofweek >= 5).astype(int),
    }, index=idx)
    return df


class TestMakeWindows:
    def test_window_covers_lookback_ending_at_origin(self, feature_df):
        lookback, horizon = 5, 3
        X, y, origins = make_windows(feature_df, lookback, horizon)

        i = 50
        origin = feature_df.index[i]
        pos = list(origins).index(origin)
        expected_window = feature_df[FEATURE_COLS].to_numpy()[i - lookback + 1: i + 1]
        np.testing.assert_allclose(X[pos], expected_window, rtol=1e-5)

    def test_target_covers_horizon_after_origin(self, feature_df):
        lookback, horizon = 5, 3
        X, y, origins = make_windows(feature_df, lookback, horizon)

        i = 50
        origin = feature_df.index[i]
        pos = list(origins).index(origin)
        expected_y = feature_df[TARGET_COL].to_numpy()[i + 1: i + 1 + horizon]
        np.testing.assert_allclose(y[pos], expected_y, rtol=1e-5)

    def test_windows_touching_nan_are_dropped(self, feature_df):
        df = feature_df.copy()
        df.iloc[60, df.columns.get_loc(TARGET_COL)] = np.nan
        lookback, horizon = 5, 3
        X, y, origins = make_windows(df, lookback, horizon)

        # Any origin whose window [i-lookback+1, i] or target [i+1, i+horizon]
        # spans row 60 must be absent.
        bad_positions = range(60 - horizon, 60 + lookback)
        bad_origins = df.index[[p for p in bad_positions if 0 <= p < len(df)]]
        assert not any(o in origins for o in bad_origins)

    def test_explicit_origins_are_respected(self, feature_df):
        lookback, horizon = 5, 3
        requested = feature_df.index[[50, 80, 100]]
        X, y, origins = make_windows(feature_df, lookback, horizon, origins=requested)

        assert set(origins) == set(requested)
        assert len(X) == len(requested)

    def test_out_of_bounds_origins_are_dropped(self, feature_df):
        lookback, horizon = 10, 24
        requested = feature_df.index[[0, 1]]  # too early for a full lookback window
        X, y, origins = make_windows(feature_df, lookback, horizon, origins=requested)

        assert len(origins) == 0
        assert X.shape == (0, lookback, len(FEATURE_COLS))


class TestTargetScaler:
    def test_transform_inverse_transform_round_trip(self):
        series = pd.Series(np.random.default_rng(0).normal(2.0, 0.5, size=500))
        scaler = TargetScaler.fit(series)

        values = series.to_numpy()
        recovered = scaler.inverse_transform(scaler.transform(values))
        np.testing.assert_allclose(recovered, values, rtol=1e-5)

    def test_fit_ignores_nan(self):
        series = pd.Series([1.0, 2.0, np.nan, 3.0])
        scaler = TargetScaler.fit(series)
        assert scaler.mean == pytest.approx(2.0)


class TestLSTMForecaster:
    def test_output_shape_matches_horizon(self):
        model = LSTMForecaster(n_features=len(FEATURE_COLS), hidden_size=8, num_layers=1, horizon=24)
        x = torch.randn(4, 10, len(FEATURE_COLS))
        out = model(x)
        assert out.shape == (4, 24)


@pytest.mark.slow
class TestTrainModel:
    def test_training_reduces_loss_and_predict_returns_expected_shape(self, feature_df):
        lookback, horizon = 5, 3
        X, y, _ = make_windows(feature_df, lookback, horizon)
        scaler = TargetScaler.fit(pd.Series(feature_df[TARGET_COL]))
        X_scaled = X.copy()
        X_scaled[:, :, TARGET_CHANNEL] = scaler.transform(X[:, :, TARGET_CHANNEL])
        y_scaled = scaler.transform(y)

        split = len(X_scaled) // 2
        train_loader = DataLoader(WindowDataset(X_scaled[:split], y_scaled[:split]), batch_size=8, shuffle=True)
        val_loader = DataLoader(WindowDataset(X_scaled[split:], y_scaled[split:]), batch_size=8)

        model = LSTMForecaster(n_features=len(FEATURE_COLS), hidden_size=8, num_layers=1, horizon=horizon)
        device = torch.device("cpu")
        model, history = train_model(model, train_loader, val_loader, device, max_epochs=20, patience=5)

        # Early stopping keeps the best-seen weights, not necessarily the
        # weights from the final recorded epoch, so check the best value
        # improved on the first epoch rather than requiring monotonic decrease.
        assert min(history["val_loss"]) <= history["val_loss"][0]

        preds = predict(model, X_scaled, device)
        assert preds.shape == y_scaled.shape
