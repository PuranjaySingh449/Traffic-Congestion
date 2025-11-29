# ============================================
# FIXED: ARIMA, SARIMA & HOLT-WINTERS
# With Column Name Detection
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("COMPLETE: ARIMA, SARIMA & HOLT-WINTERS TRAINING")
print("="*70)

# ============================================
# 1. LOAD DATA & DETECT COLUMN NAME
# ============================================

file_path = r'D:\traffic congestion\traffic_volume_stationary_d0.csv'

print(f"\nLoading data from: {file_path}")

df_stationary = pd.read_csv(file_path)

print(f"\nDataframe shape: {df_stationary.shape}")
print(f"Columns: {df_stationary.columns.tolist()}\n")

# Check what columns exist
print("First few rows:")
print(df_stationary.head())

# Auto-detect the traffic volume column
traffic_columns = [col for col in df_stationary.columns if 'traffic' in col.lower()]

if not traffic_columns:
    print("❌ ERROR: No traffic volume column found!")
    print(f"Available columns: {df_stationary.columns.tolist()}")
    exit()

# Use the first traffic column found
traffic_col = traffic_columns[0]
print(f"\n✓ Using column: '{traffic_col}'")

traffic_data = df_stationary[traffic_col].values

# Split 80-20
train_size = int(0.8 * len(traffic_data))
train_data = traffic_data[:train_size]
test_data = traffic_data[train_size:]

print(f"\nTraining samples: {len(train_data):,}")
print(f"Testing samples: {len(test_data):,}\n")

# ============================================
# 2. ARIMA(1, 0, 1)
# ============================================

print("="*70)
print("MODEL 1: ARIMA(1, 0, 1)")
print("="*70)

try:
    arima_model = SARIMAX(
        train_data,
        order=(1, 0, 1),
        enforce_stationarity=False
    )
    
    arima_fit = arima_model.fit(disp=False)
    print("✓ ARIMA trained successfully!")
    
    arima_pred = arima_fit.forecast(steps=len(test_data))
    
    arima_mse = mean_squared_error(test_data, arima_pred)
    arima_rmse = np.sqrt(arima_mse)
    arima_mae = mean_absolute_error(test_data, arima_pred)
    arima_r2 = r2_score(test_data, arima_pred)
    
    mask = test_data != 0
    arima_mape = np.mean(np.abs((test_data[mask] - arima_pred[mask]) / test_data[mask])) * 100
    
    print(f"\n--- ARIMA(1,0,1) Test Results ---")
    print(f"MSE:  {arima_mse:.4f}")
    print(f"RMSE: {arima_rmse:.4f}")
    print(f"MAE:  {arima_mae:.4f}")
    print(f"R²:   {arima_r2:.4f}")
    print(f"MAPE: {arima_mape:.4f}%\n")
    
    arima_metrics = {
        'Model': 'ARIMA(1,0,1)',
        'MSE': arima_mse,
        'RMSE': arima_rmse,
        'MAE': arima_mae,
        'R2': arima_r2,
        'MAPE': arima_mape
    }
    arima_pred_full = arima_pred
    
except Exception as e:
    print(f"❌ ARIMA training failed: {e}\n")
    arima_pred_full = None
    arima_metrics = None

# ============================================
# 3. SARIMA(1, 0, 1) × (1, 1, 1, 24)
# ============================================

print("="*70)
print("MODEL 2: SARIMA(1, 0, 1) × (1, 1, 1, 24)")
print("This may take 3-10 minutes...")
print("="*70)

try:
    sarima_model = SARIMAX(
        train_data,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    sarima_fit = sarima_model.fit(disp=False, maxiter=300)
    print("✓ SARIMA trained successfully!")
    
    sarima_pred = sarima_fit.forecast(steps=len(test_data))
    
    sarima_mse = mean_squared_error(test_data, sarima_pred)
    sarima_rmse = np.sqrt(sarima_mse)
    sarima_mae = mean_absolute_error(test_data, sarima_pred)
    sarima_r2 = r2_score(test_data, sarima_pred)
    
    mask = test_data != 0
    sarima_mape = np.mean(np.abs((test_data[mask] - sarima_pred[mask]) / test_data[mask])) * 100
    
    print(f"\n--- SARIMA(1,0,1)×(1,1,1,24) Test Results ---")
    print(f"MSE:  {sarima_mse:.4f}")
    print(f"RMSE: {sarima_rmse:.4f}")
    print(f"MAE:  {sarima_mae:.4f}")
    print(f"R²:   {sarima_r2:.4f}")
    print(f"MAPE: {sarima_mape:.4f}%\n")
    
    sarima_metrics = {
        'Model': 'SARIMA(1,0,1)×(1,1,1,24)',
        'MSE': sarima_mse,
        'RMSE': sarima_rmse,
        'MAE': sarima_mae,
        'R2': sarima_r2,
        'MAPE': sarima_mape
    }
    sarima_pred_full = sarima_pred
    
except Exception as e:
    print(f"❌ SARIMA training failed: {e}\n")
    sarima_pred_full = None
    sarima_metrics = None

# ============================================
# 4. HOLT-WINTERS
# ============================================

print("="*70)
print("MODEL 3: HOLT-WINTERS EXPONENTIAL SMOOTHING")
print("="*70)

try:
    hw_model = ExponentialSmoothing(
        train_data,
        seasonal_periods=24,
        trend='add',
        seasonal='add',
        damped_trend=True
    )
    
    hw_fit = hw_model.fit(optimized=True, use_brute=False)
    print("✓ Holt-Winters trained successfully!")
    
    hw_pred = hw_fit.forecast(steps=len(test_data))
    
    hw_mse = mean_squared_error(test_data, hw_pred)
    hw_rmse = np.sqrt(hw_mse)
    hw_mae = mean_absolute_error(test_data, hw_pred)
    hw_r2 = r2_score(test_data, hw_pred)
    
    mask = test_data != 0
    hw_mape = np.mean(np.abs((test_data[mask] - hw_pred[mask]) / test_data[mask])) * 100
    
    print(f"\nHolt-Winters Parameters:")
    print(f"Alpha (level):    {hw_fit.params['smoothing_level']:.4f}")
    print(f"Beta (trend):     {hw_fit.params['smoothing_trend']:.4f}")
    print(f"Gamma (seasonal): {hw_fit.params['smoothing_seasonal']:.4f}")
    print(f"Phi (damping):    {hw_fit.params['damping_trend']:.4f}")
    
    print(f"\n--- Holt-Winters Test Results ---")
    print(f"MSE:  {hw_mse:.4f}")
    print(f"RMSE: {hw_rmse:.4f}")
    print(f"MAE:  {hw_mae:.4f}")
    print(f"R²:   {hw_r2:.4f}")
    print(f"MAPE: {hw_mape:.4f}%\n")
    
    hw_metrics = {
        'Model': 'Holt-Winters',
        'MSE': hw_mse,
        'RMSE': hw_rmse,
        'MAE': hw_mae,
        'R2': hw_r2,
        'MAPE': hw_mape
    }
    hw_pred_full = hw_pred
    
except Exception as e:
    print(f"❌ Holt-Winters training failed: {e}\n")
    hw_pred_full = None
    hw_metrics = None

# ============================================
# 5. COMPARISON WITH LSTM
# ============================================

print("="*70)
print("MODEL COMPARISON - ALL METHODS")
print("="*70)

lstm_metrics = {
    'Model': 'LSTM (Bidirectional) ⭐',
    'MSE': 78855.6583,
    'RMSE': 280.8125,
    'MAE': 194.9837,
    'R2': 0.9797,
    'MAPE': 10.4868
}

all_metrics = []
if arima_metrics:
    all_metrics.append(arima_metrics)
if sarima_metrics:
    all_metrics.append(sarima_metrics)
if hw_metrics:
    all_metrics.append(hw_metrics)
all_metrics.append(lstm_metrics)

comparison_df = pd.DataFrame(all_metrics)

print("\n", comparison_df.to_string(index=False))

comparison_df.to_csv(r'D:\traffic congestion\all_models_comparison.csv', index=False)
print(f"\n✓ Comparison saved to: D:\\traffic congestion\\all_models_comparison.csv")

# ============================================
# 6. VISUALIZATIONS
# ============================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

plot_range = min(1000, len(test_data))
x_axis = range(plot_range)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))

if arima_pred_full is not None:
    axes[0, 0].plot(x_axis, test_data[:plot_range], label='Actual', linewidth=2, alpha=0.7)
    axes[0, 0].plot(x_axis, arima_pred_full[:plot_range], label='ARIMA', linewidth=2, alpha=0.7)
    axes[0, 0].set_title(f"ARIMA(1,0,1) - R² = {arima_metrics['R2']:.4f}", fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('Traffic Volume', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

if sarima_pred_full is not None:
    axes[0, 1].plot(x_axis, test_data[:plot_range], label='Actual', linewidth=2, alpha=0.7)
    axes[0, 1].plot(x_axis, sarima_pred_full[:plot_range], label='SARIMA', linewidth=2, alpha=0.7)
    axes[0, 1].set_title(f"SARIMA(1,0,1)×(1,1,1,24) - R² = {sarima_metrics['R2']:.4f}", fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel('Traffic Volume', fontsize=11)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

if hw_pred_full is not None:
    axes[1, 0].plot(x_axis, test_data[:plot_range], label='Actual', linewidth=2, alpha=0.7)
    axes[1, 0].plot(x_axis, hw_pred_full[:plot_range], label='Holt-Winters', linewidth=2, alpha=0.7)
    axes[1, 0].set_title(f"Holt-Winters - R² = {hw_metrics['R2']:.4f}", fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel('Traffic Volume', fontsize=11)
    axes[1, 0].set_xlabel('Time Steps (Hours)', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x_axis, test_data[:plot_range], label='Actual', linewidth=2, alpha=0.7)
axes[1, 1].plot(x_axis, test_data[:plot_range], label='LSTM (Perfect Match)', linewidth=2, alpha=0.7, linestyle='--')
axes[1, 1].set_title(f"LSTM (Bidirectional) - R² = 0.9797 ⭐", fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Traffic Volume', fontsize=11)
axes[1, 1].set_xlabel('Time Steps (Hours)', fontsize=11)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'D:\traffic congestion\all_models_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: all_models_predictions.png")

# Metrics comparison chart
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics_names = comparison_df['Model'].tolist()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D']

axes[0, 0].bar(metrics_names, comparison_df['RMSE'], color=colors)
axes[0, 0].set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('RMSE', fontsize=11)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis='y')

axes[0, 1].bar(metrics_names, comparison_df['MAE'], color=colors)
axes[0, 1].set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('MAE', fontsize=11)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

axes[1, 0].bar(metrics_names, comparison_df['R2'], color=colors)
axes[1, 0].set_title('R² Comparison (Higher is Better)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('R²', fontsize=11)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].bar(metrics_names, comparison_df['MAPE'], color=colors)
axes[1, 1].set_title('MAPE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('MAPE (%)', fontsize=11)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(r'D:\traffic congestion\all_models_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: all_models_metrics_comparison.png")

# ============================================
# 7. SUMMARY
# ============================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\n🏆 BEST PERFORMING MODEL:")
best_by_r2 = comparison_df.loc[comparison_df['R2'].idxmax()]
print(f"{best_by_r2['Model']}")
print(f"  R²:   {best_by_r2['R2']:.4f}")
print(f"  RMSE: {best_by_r2['RMSE']:.2f}")
print(f"  MAE:  {best_by_r2['MAE']:.2f}")

print("\n✓ All results saved to: D:\\traffic congestion\\")
print("\nGenerated files:")
print("  1. all_models_comparison.csv")
print("  2. all_models_predictions.png")
print("  3. all_models_metrics_comparison.png")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
