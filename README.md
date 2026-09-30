# Time Series Analysis and Forecasting of Daily Temperature in Delhi

A complete, beginner-to-intermediate time-series project: from raw data to a 90-day forecast with prediction intervals, following a documented and leakage-free workflow.

## Project Overview

The notebook analyses daily climate observations for Delhi, India (2013-2017) and forecasts the **daily mean temperature (°C)**. Every step is explained in a *Concept → Why → Code → Interpretation* format.

## Dataset

- **Name:** Daily Climate Time Series Data
- **Source:** https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
- **Files used:** `DailyDelhiClimateTrain.csv` (1,462 rows × 5 columns: `date`, `meantemp`, `humidity`, `wind_speed`, `meanpressure`; daily from 2013-01-01) for training, and `DailyDelhiClimateTest.csv` (114 rows, early 2017) as the hold-out test set
- **Target:** `meantemp`
- According to the dataset page, the data was collected from the Weather Underground API.

## Objectives

- Prepare and index time-series data correctly
- Identify trend and seasonality
- Test stationarity and apply differencing only when justified
- Read ACF/PACF plots
- Validate forecasts chronologically and compare against baselines
- Diagnose residuals and forecast with uncertainty

## Technologies Used

Python, pandas, NumPy, Matplotlib, Seaborn, statsmodels, scikit-learn, Jupyter

## Methodology

1. Load and inspect the data
2. Clean: datetime conversion, sorting, duplicate and gap checks, plausibility checks (abnormal values flagged, not invented)
3. Exploratory analysis: rolling mean/std, distribution, calendar patterns
4. Resampling (weekly, monthly)
5. Additive decomposition (period = 365)
6. Stationarity: ADF test, first-order differencing
7. ACF and PACF
8. Train on the training file, test on the provided test file (chronological; overlapping date removed from the test set)
9. Fit 11 models; ARIMA orders chosen by a small AIC search on training data only
10. Evaluation, residual analysis, model selection rule, 90-day forecast

## Models Used (11)

| Model | Family |
|---|---|
| Naive | baseline |
| Seasonal naive (365 days) | baseline |
| Seasonal mean (same date, all earlier years) | baseline |
| Moving average (30 days) | smoothing |
| ARIMA | classical, non-seasonal |
| Harmonic regression (trend + Fourier terms) | regression |
| SARIMAX + Fourier terms | classical, seasonal regressors with ARIMA errors |
| STL + ARIMA | decomposition |
| Theta (additive seasonal adjustment) | classical |
| Random Forest (lags + seasonal terms, recursive) | machine learning |
| Gradient Boosting (lags + seasonal terms, recursive) | machine learning |

A classic `SARIMA(...)[365]` is avoided on purpose: it is impractical for daily data with a yearly season.

## Evaluation Metrics

MAE, RMSE and MAPE on the same chronological test period. MAPE is reported with a caveat (the Celsius scale has an arbitrary zero).

## Key Findings

> Run the notebook and paste your own results here (the notebook prints a generated summary in section 21).

| Model | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| Naive | | | |
| Seasonal Naive (365d) | | | |
| Seasonal Mean | | | |
| Moving Average (30d) | | | |
| ARIMA | | | |
| Harmonic Regression | | | |
| SARIMAX + Fourier | | | |
| STL + ARIMA | | | |
| Theta | | | |
| Random Forest (lags) | | | |
| Gradient Boosting (lags) | | | |

Qualitative takeaways the notebook demonstrates: temperature has a strong yearly cycle and no weekly pattern; models without a seasonal component cannot follow the cycle over a multi-month horizon; the final model is selected by a rule stated in advance (lowest test RMSE).

## Repository Structure

```text
time-series-analysis-project/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── time_series_analysis.ipynb
├── data/
│   └── README.md
├── src/
│   └── README.md
└── figures/
    └── README.md
```

## How to Run

```bash
git clone https://github.com/<your-username>/time-series-analysis-project.git
cd time-series-analysis-project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# download both CSV files into data/ (see data/README.md)
jupyter notebook notebooks/time_series_analysis.ipynb
```

**Google Colab:** upload both CSV files to `/content/` and run the notebook; it looks there first.

**Note:** the test file is short (about 3.5 months, winter to early summer), so small differences between models are not reliable evidence.

## Dataset / License Note

The dataset belongs to its original provider and is not redistributed here. Check the licence and terms on the Kaggle page and the original source before any reuse.

## Future Improvements

- Rolling-origin (walk-forward) cross-validation
- Add humidity, pressure and wind as exogenous regressors
- Compare with ETS, Prophet and gradient-boosting models with lag features
- Add a separate validation period so the test file is not used for both comparison and selection
- Tune the machine-learning models
- Move helper functions into `src/` and add tests
