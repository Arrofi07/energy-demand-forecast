"""Business KPI dashboard -- an interactive version of
`10_business_impact.ipynb`'s illustrative cost-savings estimate. The
accuracy numbers (MAE/RMSE reduction) are recomputed live from the same
saved backtest results the notebook uses; the two assumption inputs
(portfolio size, imbalance premium) are sliders here instead of a fixed
notebook cell, so the "change the assumptions, get a different number"
point the notebook makes explicitly is something you can actually do.
"""

import sys
from pathlib import Path

# See the matching comment in src/dashboard/app.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.dashboard.data_access import summarize_24h_benchmark

st.set_page_config(page_title="Business Impact", page_icon="💰", layout="wide")
st.title("💰 Business Impact")

st.caption(
    "Scope honesty, same as the notebook: this dataset is one household's "
    "consumption, not a utility's aggregate load. The accuracy comparison below is "
    "real and directly usable; any dollar estimate requires scaling up to a "
    "hypothetical portfolio, which needs the explicit assumptions below."
)

summary = summarize_24h_benchmark()
baseline_mae, baseline_rmse = summary.loc["daily_seasonal_naive", ["MAE", "RMSE"]]
lgbm_mae, lgbm_rmse = summary.loc["lightgbm", ["MAE", "RMSE"]]

mae_reduction_kw = baseline_mae - lgbm_mae
mae_reduction_pct = mae_reduction_kw / baseline_mae * 100
rmse_reduction_pct = (baseline_rmse - lgbm_rmse) / baseline_rmse * 100

col1, col2, col3 = st.columns(3)
col1.metric("LightGBM MAE", f"{lgbm_mae:.3f} kW", f"-{mae_reduction_pct:.1f}% vs. baseline", delta_color="inverse")
col2.metric("LightGBM RMSE", f"{lgbm_rmse:.3f} kW", f"-{rmse_reduction_pct:.1f}% vs. baseline", delta_color="inverse")
col3.metric("MAE reduction / household", f"{mae_reduction_kw:.3f} kW")

st.markdown(
    """
    - **MAE → imbalance cost.** Day-ahead markets settle the gap between
      scheduled and actual delivery at an imbalance price -- expected cost
      scales with *average absolute* error.
    - **RMSE → reserve margin.** Reserve capacity held on standby to cover
      forecast risk scales with the *spread* of the error distribution,
      which penalizes large misses more than MAE does.
    """
)

st.subheader("Illustrative utility-scale savings estimate")
st.caption("Change these assumptions -- the estimate below updates live. Not a validated ROI figure.")

col1, col2 = st.columns(2)
with col1:
    portfolio_households = st.slider(
        "Portfolio size (households)", min_value=1_000, max_value=100_000, value=10_000, step=1_000,
    )
with col2:
    imbalance_premium = st.slider(
        "Imbalance premium ($/MWh)", min_value=1.0, max_value=100.0, value=20.0, step=1.0,
    )

HOURS_PER_YEAR = 24 * 365
annual_mae_reduction_kwh_per_household = mae_reduction_kw * HOURS_PER_YEAR
portfolio_annual_mae_reduction_mwh = annual_mae_reduction_kwh_per_household * portfolio_households / 1000
illustrative_annual_savings = portfolio_annual_mae_reduction_mwh * imbalance_premium

st.metric("Illustrative avoided imbalance cost / year", f"${illustrative_annual_savings:,.0f}")

with st.expander("How this number is computed"):
    st.markdown(
        f"""
        1. MAE reduction per household: **{mae_reduction_kw:.3f} kW/hour** (LightGBM vs.
           daily seasonal naive baseline, recomputed live from `data/results/`)
        2. → **{annual_mae_reduction_kwh_per_household:,.0f} kWh/year** of reduced absolute
           forecast error, per household ({mae_reduction_kw:.3f} kW × {HOURS_PER_YEAR:,} hours)
        3. → **{portfolio_annual_mae_reduction_mwh:,.0f} MWh/year** across a
           {portfolio_households:,}-household portfolio
        4. → **${illustrative_annual_savings:,.0f}/year** at ${imbalance_premium:.0f}/MWh

        Assumes per-household forecast error carries over unchanged to a pooled
        portfolio -- real aggregation effects (errors partially cancel across many
        independent households) would likely make this conservative, i.e. probably
        an upper bound rather than an underestimate.
        """
    )

st.subheader("Recommendation")
st.markdown(
    """
    **Primary production model: LightGBM** -- best or statistically-tied-best accuracy
    at every horizon tested, cheapest to retrain, the only model proven to extend
    cleanly across horizons, and the only one with a built-in explainability story (SHAP).

    **Secondary: Prophet** for longer-horizon planning conversations with non-technical
    stakeholders. **Not recommended: SARIMA** (loses to the naive baseline in most
    seasons, collapses at 7d). **LSTM**: a validated research result, not a production
    pick -- its accuracy gap with LightGBM isn't statistically significant (Phase 9,
    Diebold-Mariano p=0.098), so its higher training cost buys nothing.
    """
)
