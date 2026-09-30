# Time Series Analysis and Forecasting of Daily Temperature in Delhi

A beginner-to-intermediate time-series project that takes Delhi's daily climate data from two raw CSV files to an 11-model forecasting comparison and a 90-day forecast. Every step is explained in a *Concept → Why → Code → Interpretation* format.

## Project Overview

The notebook analyses daily observations for Delhi, India (2013 – April 2017) and forecasts the **daily mean temperature (°C)**. The training file is used for exploration and model fitting. The provided test file is used as a chronological hold-out set. The workflow covers data preparation, exploratory analysis, decomposition, stationarity testing, ACF/PACF, baseline, classical and machine-learning models, residual diagnostics and a documented model-selection rule.

## Dataset

- **Name:** Daily Climate Time Series Data
- **Source:** https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
- **Training file:** `DailyDelhiClimateTrain.csv`, 1,462 rows × 5 columns (`date`, `meantemp`, `humidity`, `wind_speed`, `meanpressure`), daily from 2013-01-01 to 2017-01-01
- **Test file:** `DailyDelhiClimateTest.csv`, 114 rows, daily, early 2017
- **Target:** `meantemp` (daily mean temperature, °C)
- According to the dataset page, the data was collected from the Weather Underground API.

Both files contain the date 2017-01-01. The notebook removes that row from the test set, so the test period used for scoring is **2017-01-02 to 2017-04-24 (113 days)**.

## Objectives

- Prepare and index time-series data correctly (datetime handling, gap and duplicate checks, abnormal-value flagging)
- Identify trend and seasonality with rolling statistics, resampling and decomposition
- Test stationarity (ADF) and apply differencing only when justified
- Read ACF and PACF plots
- Validate forecasts chronologically with no data leakage
- Compare baseline, classical and machine-learning models
- Diagnose residuals and forecast with uncertainty

## Technologies Used

Python, pandas, NumPy, Matplotlib, Seaborn, statsmodels, scikit-learn, Jupyter

## Methodology

1. Load and inspect both files
2. Clean: datetime conversion, sorting, duplicate and gap checks, plausibility checks (abnormal values are flagged, not invented), overlap check between train and test
3. Explore the **training data only**: rolling mean and standard deviation, distribution, calendar patterns
4. Resample to weekly and monthly means
5. Additive decomposition (period = 365)
6. Stationarity: ADF test and first-order differencing
7. ACF and PACF
8. Fit 11 models on the training file; score them on the test file with a multi-step forecast of the whole test period
9. Residual analysis of the lowest-RMSE model
10. Refit on train + test and forecast the next 90 days

## Models Used (11)

| Model | Family |
|---|---|
| Naive | baseline |
| Seasonal Naive (365 days) | baseline |
| Seasonal Mean (same date, all earlier years) | baseline |
| Moving Average (30 days) | smoothing |
| ARIMA | classical, non-seasonal |
| Harmonic Regression (linear trend + Fourier terms) | regression |
| SARIMAX + Fourier terms | classical, seasonal regressors with ARIMA errors |
| STL + ARIMA | decomposition |
| Theta (additive seasonal adjustment) | classical |
| Random Forest (lags + seasonal terms, recursive) | machine learning |
| Gradient Boosting (lags + seasonal terms, recursive) | machine learning |

A classic `SARIMA(...)[365]` is avoided on purpose: it is impractical for daily data with a yearly season. ARIMA orders were chosen by a small AIC search (orders 0–2) on training data only. The machine-learning models use fixed, untuned hyperparameters.

## Evaluation Metrics

MAE, RMSE and MAPE on the same test period. MAPE is reported with a caveat: the Celsius scale has an arbitrary zero, so percentage errors are not fully meaningful.

## Key Findings

**Data and seasonality**

- Temperature follows a strong yearly cycle: seasonal strength from the decomposition is **0.94**.
- On average **June is the warmest month (33.7 °C)** and **January the coldest (13.3 °C)**.
- There is no meaningful weekly pattern, as expected for weather data.

**Trend**

- Yearly means of the complete training years: 2013: 24.8 °C, 2014: 25.0 °C, 2015: 25.1 °C, 2016: 27.1 °C.
- Trend strength is only **0.17**. 2016 was warmer, but one warm year in a four-year sample is not enough to claim a long-term trend.

**Stationarity**

- ADF p-value on the original series: **0.2774**, so the unit-root hypothesis is not rejected.
- ADF p-value after first-order differencing: **≈ 0.0000**.
- Differencing order used for ARIMA: **d = 1**.

**Model performance on the test file (113 days), sorted by RMSE**

| Model | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| Harmonic Regression | 2.028 | 2.456 | 10.439 |
| SARIMAX + Fourier | 2.302 | 2.904 | 10.823 |
| Seasonal Mean | 2.303 | 2.930 | 10.900 |
| Random Forest (lags) | 2.318 | 2.942 | 11.006 |
| STL + ARIMA | 2.596 | 3.233 | 13.856 |
| Seasonal Naive (365d) | 2.621 | 3.259 | 13.643 |
| Gradient Boosting (lags) | 2.676 | 3.344 | 11.910 |
| Theta | 3.547 | 4.153 | 16.546 |
| Moving Average (30d) | 5.830 | 7.753 | 23.524 |
| ARIMA | 8.740 | 10.696 | 35.336 |
| Naive | 11.764 | 13.362 | 50.027 |

- **Selected model (lowest RMSE): Harmonic Regression** (RMSE 2.456 °C, MAE 2.028 °C). Its mean residual is **−0.42 °C** (actual minus forecast), so it forecast slightly too warm on average over this window.
- Models that explicitly represent the yearly cycle (Harmonic Regression, SARIMAX + Fourier, Seasonal Mean, Random Forest) are clearly better than models that do not (Naive, Moving Average, non-seasonal ARIMA). The non-seasonal models cannot follow the seasonal climb over a multi-month horizon.
- Among the four best models, RMSE differs by about 0.5 °C. With only 113 test days from one part of the year, these differences are **weak evidence**, so the top four should be regarded as roughly comparable.
- The Naive, Moving Average and ARIMA errors are very large. These models depend heavily on the most recent training values, so they are sensitive to unusual observations at the end of the training period. The shared date 2017-01-01 appears in both files and is worth inspecting (the notebook prints both versions side by side).

**Forecast**

- The selected model is refitted on train + test and forecasts the 90 days after the test period (from 2017-04-25). The forecast continues the seasonal pattern into the pre-monsoon and monsoon months, which were **not** in the test window, so it should be treated with extra caution. See `figures/17_future_forecast.png` after running the notebook.

## Figures

After running the notebook, the plots are saved to `figures/`. Suggested ones for display:

![Model grid](figures/14_model_grid.png)
![Top models](figures/15_top_models.png)
![Future forecast](figures/17_future_forecast.png)

## Limitations

- **Short test window:** about 113 days (winter to early summer), so the ranking does not cover the monsoon or the hottest months.
- **Test set used twice:** it is used both to compare models and to choose one. A separate validation period or rolling-origin cross-validation would be more rigorous.
- **Short history:** only about four years of training data.
- **Univariate models:** humidity, wind and pressure are not used.
- **Untuned machine-learning models**, and recursive forecasting accumulates errors over long horizons.
- **Constant variance assumed** in prediction intervals.
- **Data quality:** abnormal values exist in other columns (for example pressure) and the data comes from a third-party API.

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

**Google Colab:** upload both CSV files to `/content/` and run the notebook. It looks there first.

## Dataset / License Note

The dataset belongs to its original provider and is not redistributed here. Check the licence and terms on the Kaggle page and the original source before any reuse.

## Future Improvements

- Rolling-origin (walk-forward) cross-validation
- A separate validation period, so the test file is used only once
- Tune the machine-learning models
- Add humidity, pressure and wind as exogenous regressors (their future values would need forecasting)
- Compare with ETS and Prophet
- Investigate the shared 2017-01-01 row and its effect on the baseline models
- Move helper functions into `src/` and add tests
