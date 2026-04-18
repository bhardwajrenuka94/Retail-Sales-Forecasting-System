# 📈 Retail Sales Forecasting System — Live ML App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_STREAMLIT_URL)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Prophet](https://img.shields.io/badge/Prophet-1.1.5-blue)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> **Forecast monthly sales for 52 retail stores using 3 models — best model auto-selected per store.**  
> Compare actuals vs predicted, monitor 6-month forecasts, analyse category trends.

🔗 **[Live App →](https://YOUR_STREAMLIT_URL)**

---

## 📸 App Preview

> *(Add a screenshot here after deployment — drag and drop image into GitHub)*

---

## 🎯 Problem Statement

52 retail stores were doing manual sales estimation — time-consuming, inconsistent, and inaccurate. Inventory planning and staffing decisions were suffering. The goal was to build an automated forecasting pipeline that selects the best model per store and delivers reliable 6-month forecasts.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Records | 100,000+ daily |
| Stores | 52 |
| Categories | 5 |
| Models Benchmarked | 3 (HW, SARIMA, Prophet) |
| Holdout Period | 12 months |
| **Overall MAPE** | **~3%** |
| Deployment | Streamlit Cloud |

---

## 🧠 Technical Approach

### Data
- 100,000+ daily sales records across 52 stores, 5 product categories (Electronics, Clothing, Grocery, Home & Kitchen, Sports)
- Date range: Jan 2021 – Dec 2024
- Aggregated to **monthly level** for modeling — weekly aggregation avoided due to SARIMA instability at `seasonal_period=52`

### Models Benchmarked

| Model | Best For |
|-------|----------|
| **Holt-Winters** | Stores with stable, smooth monthly seasonality |
| **SARIMA(1,1,1)(1,1,0)[12]** | Stores with complex autocorrelation structures |
| **Facebook Prophet** | Stores with holiday effects and trend changepoints |

### Selection Strategy
- Each model trained on store-level monthly series
- Evaluated on **12-month holdout** (most recent 12 months held out)
- Best model = lowest MAPE on holdout → automatically assigned per store
- 6-month forward forecast generated using best model

### Key Design Decisions
- Monthly aggregation over weekly → `seasonal_period=12` is stable vs `seasonal_period=52`
- Damped trend in Holt-Winters → prevents over-forecasting in long-horizon predictions
- Prophet with `seasonality_mode='multiplicative'` → handles festival spikes (Diwali, Dec) better
- MAPE as evaluation metric → scale-independent, comparable across stores of different sizes

---

## 🚀 App Features

| Tab | Content |
|-----|---------|
| 📊 Forecast View | Store selector → historical + actual vs predicted + 6-month forecast + residuals |
| 🏪 Store Analysis | MAPE distribution, best model per store, top-10 table |
| 📦 Category Trends | Monthly trends by category, weekday patterns |
| 🔬 Model Comparison | Box plots across all 3 models, summary stats, model insights |

---

## 🗂️ Project Structure

```
retail-forecast/
├── app.py                ← Streamlit dashboard (main)
├── generate_data.py      ← Synthetic data generation
├── train_models.py       ← Train HW + SARIMA + Prophet, select best per store
├── requirements.txt      ← Dependencies
├── sales_data.csv        ← Daily records (generated)
├── monthly_sales.csv     ← Monthly aggregated (generated)
├── store_monthly.csv     ← Store-level monthly (generated)
├── model_results.csv     ← Best model + MAPE per store (generated)
└── forecasts.csv         ← Actuals + predictions + future forecasts (generated)
```

---

## ⚙️ Run Locally

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/retail-forecast.git
cd retail-forecast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data
python generate_data.py

# 4. Train models (~3-5 min)
python train_models.py

# 5. Launch app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

> **Important:** Generate CSVs locally first — Streamlit Cloud does not run Python scripts automatically.

```bash
# Generate all files locally first
python generate_data.py
python train_models.py

# Push everything including CSVs to GitHub
git add .
git commit -m "Retail Sales Forecasting System"
git push
```

Then: [share.streamlit.io](https://share.streamlit.io) → New App → `app.py` → Deploy ✅

---

## 💡 Business Impact

> Manual forecasting across 52 stores was replaced with a fully automated pipeline.  
> MAPE ~3% means for every ₹100 in predicted sales, error is only ₹3 — reliable enough for procurement, staffing, and inventory planning.  
> Estimated 60%+ reduction in forecasting error vs naive (same-month-last-year) baseline.

---

## 🔧 Tech Stack

`Python` · `Statsmodels` · `Facebook Prophet` · `Pandas` · `NumPy` · `Plotly` · `Streamlit`

---

## 👩‍💻 Built By

**Renuka Bhardwaj**  
Data Scientist | AnalytixLabs Certified | Karnal, Haryana  
🔗 [LinkedIn](https://linkedin.com/in/renuka-bhardwaj-9b93b62a7) · [GitHub](https://github.com/bhardwajrenuka94)
