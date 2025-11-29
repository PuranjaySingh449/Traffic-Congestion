# Traffic Volume Forecasting

**LSTM BiLSTM R²=0.98** vs ARIMA/SARIMA/Holt-Winters on Metro Interstate Traffic (hourly).[1]

## 📊 Results

| Model                  | RMSE  | R²    | MAPE   |
|------------------------|-------|-------|--------|
| **LSTM (BiLSTM)**      | **281**| **0.98**| **10.5%** |
| ARIMA(1,0,1)           | 3799  | -2.73 | 99.98% |
| SARIMA(24h)            | 1987  | -0.02 | 180.9% |
| Holt-Winters           | 3556  | -2.27 | 119.7% |[1]

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python train_lstm.py  # ~20min, R²=0.98
```

## 🗂️ Datasets

Place CSV in `data/` (or update paths):

```
data/
└── Metro_Interstate_Traffic_Volume.csv
```

**Columns**: `traffic_volume`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `date_time`

## 📁 Outputs

```
✅ best_traffic_lstm_model.h5     (R²=0.98 - BEST)
✅ traffic_scaler.pkl            (Production scaler)
✅ all_models_comparison.csv     (4-model results)
✅ training_history.png          (Loss curves)
✅ predictions_vs_actual.png     (Test predictions)
✅ scatter_plot.png             (R² visual)
✅ residual_analysis.png        (Error analysis)
✅ future_predictions.png       (Next 24h)
```

## ⚙️ Requirements

```txt
tensorflow>=2.13.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib
```

## ✨ Features

- **BiLSTM**: 50+ features (24h lags, cyclical hour/day, weather, rush hour)
- **Auto-Stationarity**: ADF tests (data already stationary d=0)
- **Production**: `.h5` model + scaler saved
- **Visuals**: 8 plots (predictions, residuals, training, future)
- **Future Predict**: Next 24h traffic forecasting

## 🔧 Tips

- **OOM?** Reduce `BATCH_SIZE=16`
- **Slow?** Use GPU TensorFlow
- **Production?** Load `best_traffic_lstm_model.h5`

***

**⭐ LSTM 10x better than classical TS! R²=0.98 → Deploy ready 🚦**[1]
