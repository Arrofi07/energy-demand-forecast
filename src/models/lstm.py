"""Direct sequence-to-sequence LSTM forecasting for the 24h-ahead benchmark.

Same "direct" philosophy as `lightgbm_direct.py`, but here it's built into the
architecture rather than into 24 separate models: an LSTM encoder reads the
recent lookback window once, and a single linear head maps its final hidden
state straight to all 24 future hours in one shot. There is no autoregressive
decoding step, so the model can't compound its own errors forecast-step by
forecast-step the way a recursive RNN would.

Only the target channel is standardized before scoring — the other input
channels (hour/day/month sin-cos pairs, `is_weekend`) are already bounded in
[-1, 1] or {0, 1} and don't need it. The scaler must be fit on the training
slice only and reused everywhere else, exactly like every other leakage rule
in this project.

Windows that touch a NaN (either in the lookback window or in the 24 target
hours) are dropped entirely, the same honest policy `lightgbm_direct.py` uses
for its `X.notna().all(axis=1)` filter — both models pay the same "missing
coverage near data gaps" cost rather than silently imputing across it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

FEATURE_COLS = [
    "global_active_power",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]
TARGET_COL = "global_active_power"
TARGET_CHANNEL = FEATURE_COLS.index(TARGET_COL)

HORIZON = 24


def get_device() -> torch.device:
    """MPS if this machine has it (Apple Silicon), otherwise CPU.

    LSTM ops run correctly on MPS as of the torch version pinned here, so
    this is a straightforward speedup, not a correctness trade-off.
    """
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Seed both numpy and torch so tuning/training runs are reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TargetScaler:
    """Standardizes only the target channel, fit on a training slice only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, series: pd.Series) -> "TargetScaler":
        clean = series.dropna()
        return cls(mean=float(clean.mean()), std=float(clean.std()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def make_windows(
    df: pd.DataFrame,
    lookback: int,
    horizon: int = HORIZON,
    origins: pd.DatetimeIndex | None = None,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build raw (unscaled) (X, y) windows from a regular hourly-indexed df.

    For a window ending at origin t: X covers [t - lookback + 1, t] (all
    `feature_cols`), y covers [t + 1, t + horizon] (`target_col` only) --
    exactly the encode-recent-window / decode-all-horizons-at-once split the
    model architecture expects.

    If `origins` is given, a window is built for exactly those timestamps
    (used to evaluate at the shared backtest origins); otherwise, a window is
    built for every valid position in `df` (used for training). Either way,
    any window touching a NaN is dropped, so the returned `origins` may be a
    subset of what was requested.
    """
    values = df[feature_cols].to_numpy(dtype="float64")
    target_values = df[target_col].to_numpy(dtype="float64")
    index = df.index
    n = len(df)

    if origins is None:
        positions = range(lookback - 1, n - horizon)
    else:
        loc = {ts: i for i, ts in enumerate(index)}
        positions = [
            loc[ts] for ts in origins
            if ts in loc and lookback - 1 <= loc[ts] <= n - horizon - 1
        ]

    X_list, y_list, origin_list = [], [], []
    for i in positions:
        window = values[i - lookback + 1: i + 1]
        target_window = target_values[i + 1: i + 1 + horizon]
        if np.isnan(window).any() or np.isnan(target_window).any():
            continue
        X_list.append(window)
        y_list.append(target_window)
        origin_list.append(index[i])

    X = np.stack(X_list).astype("float32") if X_list else np.empty((0, lookback, len(feature_cols)), dtype="float32")
    y = np.stack(y_list).astype("float32") if y_list else np.empty((0, horizon), dtype="float32")
    return X, y, pd.DatetimeIndex(origin_list)


def scale_windows(X: np.ndarray, y: np.ndarray, scaler: TargetScaler) -> tuple[np.ndarray, np.ndarray]:
    """Apply `scaler` to the target channel of X and to all of y."""
    X_scaled = X.copy()
    X_scaled[:, :, TARGET_CHANNEL] = scaler.transform(X[:, :, TARGET_CHANNEL])
    y_scaled = scaler.transform(y)
    return X_scaled, y_scaled


class WindowDataset(Dataset):
    """Thin tensor wrapper so `make_windows` output can feed a DataLoader."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    """Encoder LSTM -> linear head producing all `horizon` outputs at once.

    Deliberately not autoregressive: the head reads only the encoder's final
    hidden state, so nothing predicted for hour h+1 depends on what was
    predicted for hour h -- matching the direct multi-horizon strategy
    `lightgbm_direct.py` uses, this time expressed architecturally.
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, horizon: int = HORIZON, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])  # final layer's final hidden state


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[nn.Module, dict]:
    """Train with early stopping on validation MSE; restore the best weights.

    Returns the model plus a per-epoch `{"train_loss": [...], "val_loss":
    [...]}` history, used both for the early-stopping decision and for the
    learning-curve comparison the roadmap asks for.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for _ in range(max_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_losses.append(loss_fn(preds, y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Run inference in batches, returning scaled (not yet inverse-transformed) predictions."""
    model.to(device)
    model.eval()
    loader = DataLoader(torch.from_numpy(X), batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for X_batch in loader:
            preds.append(model(X_batch.to(device)).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0, model.head.out_features), dtype="float32")


def direct_seq2seq_backtest(
    df: pd.DataFrame,
    model: nn.Module,
    scaler: TargetScaler,
    origins: pd.DatetimeIndex,
    lookback: int,
    device: torch.device,
    horizon: int = HORIZON,
    method_name: str = "lstm",
) -> pd.DataFrame:
    """Evaluate the trained model at the shared backtest origins.

    Same long-format (origin, horizon_step, method, actual, forecast) output
    as every other model's backtest, so results are directly comparable and
    poolable with `src/evaluation/results_store.py`.
    """
    X, y, window_origins = make_windows(df, lookback, horizon, origins=origins)
    if len(X) == 0:
        return pd.DataFrame(columns=["origin", "horizon_step", "method", "actual", "forecast"])

    X_scaled = X.copy()
    X_scaled[:, :, TARGET_CHANNEL] = scaler.transform(X[:, :, TARGET_CHANNEL])

    preds_scaled = predict(model, X_scaled, device)
    preds = scaler.inverse_transform(preds_scaled)

    records = []
    for origin, actual_row, forecast_row in zip(window_origins, y, preds):
        for h, (actual, forecast) in enumerate(zip(actual_row, forecast_row), start=1):
            records.append({
                "origin": origin,
                "horizon_step": h,
                "method": method_name,
                "actual": actual,
                "forecast": forecast,
            })
    return pd.DataFrame.from_records(records)
