"""
train_models.py — Retail Sales Forecasting
Trains on MONTHLY store-level data (48 months).
Models: Holt-Winters, SARIMA, Prophet
Best model per store = lowest MAPE on 6-month holdout.
Target: Overall MAPE ~8%
 
Run: python train_models.py   (~3-5 min for 52 stores)
Produces: model_results.csv, forecasts.csv
"""
 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
 
# ── Graceful imports ─────────────────────────────────────
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HW_OK = True
except ImportError:
    HW_OK = False
    print("⚠ statsmodels missing. pip install statsmodels")
 
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_OK = True
except ImportError:
    SARIMA_OK = False
 
try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_OK = True
    except ImportError:
        PROPHET_OK = False
        print("⚠ Prophet missing. pip install prophet")
 
if not HW_OK and not SARIMA_OK and not PROPHET_OK:
    print("ERROR: No modeling libraries found!")
    print("Run: pip install statsmodels prophet")
    exit(1)
 
 
def smape(actual, predicted):
    """Symmetric MAPE — handles zeros better."""
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    denom = (np.abs(a) + np.abs(p)) / 2
    mask  = denom > 0
    return np.mean(np.abs(a[mask] - p[mask]) / denom[mask]) * 100
 
 
def mape(actual, predicted):
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    mask = a > 0
    return np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100
 
 
def fit_holt_winters(series, holdout):
    """Holt-Winters on monthly data — seasonal_periods=12, damped trend."""
    if not HW_OK or len(series) < 24:
        return None, None, None
    try:
        model = ExponentialSmoothing(
            series, trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=12, initialization_method="estimated",
        )
        fit = model.fit(optimized=True, remove_bias=False)
        pred   = fit.forecast(holdout)
        future = fit.forecast(holdout + 6)[-6:]
        return pred.values, future.values, fit
    except Exception as e:
        return None, None, None
 
 
def fit_sarima(series, holdout):
    """SARIMA(1,1,1)(1,1,0)[12] on monthly data."""
    if not SARIMA_OK or len(series) < 30:
        return None, None, None
    try:
        model = SARIMAX(
            series, order=(1,1,1), seasonal_order=(1,1,0,12),
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fit    = model.fit(disp=False, maxiter=200)
        pred   = fit.forecast(holdout)
        future = fit.forecast(holdout + 6)[-6:]
        return np.array(pred), np.array(future), fit
    except Exception:
        return None, None, None
 
 
def fit_prophet(train_df, holdout):
    """Prophet on monthly data."""
    if not PROPHET_OK or len(train_df) < 24:
        return None, None, None
    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10,
        )
        m.fit(train_df, iter=500)
        future_df   = m.make_future_dataframe(periods=holdout + 6, freq="MS")
        forecast_df = m.predict(future_df)
        pred   = forecast_df.iloc[-(holdout + 6):-6]["yhat"].values
        future = forecast_df.iloc[-6:]["yhat"].values
        return pred, future, m
    except Exception:
        return None, None, None
 
 
def main():
    print("Loading store_monthly.csv...")
    try:
        df = pd.read_csv("store_monthly.csv", parse_dates=["date"])
    except FileNotFoundError:
        print("ERROR: store_monthly.csv not found. Run generate_data.py first.")
        return
 
    stores   = sorted(df["store_id"].unique())
    HOLDOUT  = 12  # 12 months holdout → realistic MAPE ~7-9%
    FORECAST = 6   # 6 months future
 
    results   = []
    forecasts = []
 
    print(f"Training 3 models for {len(stores)} stores (holdout={HOLDOUT}m)...\n")
 
    for i, store in enumerate(stores):
        sdf = df[df["store_id"] == store].sort_values("date").copy()
        sdf = sdf.dropna(subset=["sales"])
 
        if len(sdf) < 30:
            continue
 
        train_df = sdf.iloc[:-HOLDOUT]
        test_df  = sdf.iloc[-HOLDOUT:]
        series   = train_df.set_index("date")["sales"].asfreq("MS")
        test_y   = test_df["sales"].values
 
        future_dates = pd.date_range(
            sdf["date"].max() + pd.DateOffset(months=1),
            periods=FORECAST, freq="MS"
        )
 
        prop_train = train_df[["date","sales"]].rename(columns={"date":"ds","sales":"y"})
 
        model_mapes   = {}
        model_preds   = {}
        model_futures = {}
 
        # Holt-Winters
        hw_pred, hw_fut, _ = fit_holt_winters(series, HOLDOUT)
        if hw_pred is not None:
            m = mape(test_y, hw_pred)
            model_mapes["Holt-Winters"]    = m
            model_preds["Holt-Winters"]    = hw_pred
            model_futures["Holt-Winters"]  = hw_fut
 
        # SARIMA
        sa_pred, sa_fut, _ = fit_sarima(series, HOLDOUT)
        if sa_pred is not None:
            m = mape(test_y, sa_pred)
            model_mapes["SARIMA"]    = m
            model_preds["SARIMA"]    = sa_pred
            model_futures["SARIMA"]  = sa_fut
 
        # Prophet
        ph_pred, ph_fut, _ = fit_prophet(prop_train, HOLDOUT)
        if ph_pred is not None:
            m = mape(test_y, ph_pred)
            model_mapes["Prophet"]    = m
            model_preds["Prophet"]    = ph_pred
            model_futures["Prophet"]  = ph_fut
 
        if not model_mapes:
            print(f"  [{i+1}/{len(stores)}] {store} — all models failed, skipping")
            continue
 
        best    = min(model_mapes, key=model_mapes.get)
        best_m  = round(model_mapes[best], 2)
        others  = "  ".join([f"{k}={v:.1f}%" for k,v in model_mapes.items()])
        print(f"  [{i+1:2d}/{len(stores)}] {store}  best={best} MAPE={best_m:.1f}%  |  {others}")
 
        results.append({
            "store_id":     store,
            "best_model":   best,
            "mape":         best_m,
            "hw_mape":      round(model_mapes.get("Holt-Winters", np.nan), 2),
            "sarima_mape":  round(model_mapes.get("SARIMA", np.nan), 2),
            "prophet_mape": round(model_mapes.get("Prophet", np.nan), 2),
        })
 
        # Historical
        for _, row in sdf.iterrows():
            forecasts.append({"store_id": store, "ds": row["date"],
                               "type": "history", "value": row["sales"], "model": best})
 
        # Actuals vs Predicted (test period)
        for dt, act, pred in zip(test_df["date"], test_y, model_preds[best]):
            forecasts.append({"store_id":store,"ds":dt,"type":"actual","value":round(act,2),"model":best})
            forecasts.append({"store_id":store,"ds":dt,"type":"predicted","value":round(max(pred,0),2),"model":best})
 
        # Future
        for dt, val in zip(future_dates, model_futures[best]):
            forecasts.append({"store_id":store,"ds":dt,"type":"forecast","value":round(max(val,0),2),"model":best})
 
    # Save
    res_df  = pd.DataFrame(results)
    fore_df = pd.DataFrame(forecasts)
    res_df.to_csv("model_results.csv",  index=False)
    fore_df.to_csv("forecasts.csv",     index=False)
 
    overall = res_df["mape"].mean()
    hw_n  = (res_df["best_model"]=="Holt-Winters").sum()
    sa_n  = (res_df["best_model"]=="SARIMA").sum()
    ph_n  = (res_df["best_model"]=="Prophet").sum()
 
    print(f"\n{'='*55}")
    print(f"  Overall MAPE     : {overall:.1f}%")
    print(f"  Stores done      : {len(res_df)}")
    print(f"  Holt-Winters wins: {hw_n} stores")
    print(f"  SARIMA wins      : {sa_n} stores")
    print(f"  Prophet wins     : {ph_n} stores")
    print(f"{'='*55}")
    print("✅ Saved: model_results.csv | forecasts.csv")
    print("Now run: streamlit run app.py")
 
 
if __name__ == "__main__":
    main()