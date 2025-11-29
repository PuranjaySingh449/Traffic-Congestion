# ============================================
# Data Preprocessing for ARIMA & SARIMA
# Stationarity Check, Differencing, and Saving
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("ARIMA & SARIMA DATA PREPROCESSING")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================

input_path = r'D:\traffic congestion\Metro_Interstate_Traffic_Volume.csv'
output_dir = r'D:\traffic congestion'

print(f"\nLoading data from: {input_path}")
df = pd.read_csv(input_path)

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")

# Extract traffic volume
traffic_series = df['traffic_volume'].copy()

print(f"\nTraffic Volume Statistics:")
print(f"Min: {traffic_series.min():.0f} vehicles")
print(f"Max: {traffic_series.max():.0f} vehicles")
print(f"Mean: {traffic_series.mean():.0f} vehicles")
print(f"Std Dev: {traffic_series.std():.0f} vehicles")

# ============================================
# 2. STATIONARITY TEST (ORIGINAL DATA)
# ============================================

print("\n" + "="*60)
print("STEP 1: CHECK STATIONARITY (ORIGINAL DATA)")
print("="*60)

def perform_adf_test(series, name="Series"):
    """Perform Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna(), autolag='AIC')
    
    print(f"\n--- {name} ---")
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.3f}")
    
    if result[1] <= 0.05:
        print(f"✓ Result: STATIONARY (p-value = {result[1]:.6f} ≤ 0.05)")
        return True, result
    else:
        print(f"✗ Result: NON-STATIONARY (p-value = {result[1]:.6f} > 0.05)")
        return True, result

is_stationary_original, adf_original = perform_adf_test(traffic_series, "Original Traffic Data")

# ============================================
# 3. FIRST DIFFERENCING
# ============================================

print("\n" + "="*60)
print("STEP 2: FIRST DIFFERENCING")
print("="*60)

# First differencing
traffic_diff_1 = traffic_series.diff().dropna()

print(f"\nFirst differencing shape: {traffic_diff_1.shape}")
print(f"First differencing statistics:")
print(f"Min: {traffic_diff_1.min():.2f}")
print(f"Max: {traffic_diff_1.max():.2f}")
print(f"Mean: {traffic_diff_1.mean():.2f}")
print(f"Std Dev: {traffic_diff_1.std():.2f}")

is_stationary_diff1, adf_diff1 = perform_adf_test(traffic_diff_1, "1st Order Differenced Data")

# ============================================
# 4. SECOND DIFFERENCING (IF NEEDED)
# ============================================

print("\n" + "="*60)
print("STEP 3: SECOND DIFFERENCING (If needed)")
print("="*60)

traffic_diff_2 = traffic_diff_1.diff().dropna()

print(f"\nSecond differencing shape: {traffic_diff_2.shape}")
print(f"Second differencing statistics:")
print(f"Min: {traffic_diff_2.min():.2f}")
print(f"Max: {traffic_diff_2.max():.2f}")
print(f"Mean: {traffic_diff_2.mean():.2f}")
print(f"Std Dev: {traffic_diff_2.std():.2f}")

is_stationary_diff2, adf_diff2 = perform_adf_test(traffic_diff_2, "2nd Order Differenced Data")

# ============================================
# 5. DETERMINE DIFFERENCING ORDER
# ============================================

print("\n" + "="*60)
print("STEP 4: DETERMINE OPTIMAL DIFFERENCING ORDER")
print("="*60)

if is_stationary_original:
    d_order = 0
    final_data = traffic_series
    recommendation = "Use original data (d=0)"
    print(f"\n✓ Original data is stationary")
    print(f"RECOMMENDATION: {recommendation}")
elif adf_diff1[1] <= 0.05:
    d_order = 1
    final_data = traffic_diff_1
    recommendation = "Use 1st differencing (d=1)"
    print(f"\n✓ 1st differencing is stationary")
    print(f"RECOMMENDATION: {recommendation}")
else:
    d_order = 2
    final_data = traffic_diff_2
    recommendation = "Use 2nd differencing (d=2)"
    print(f"\n✓ 2nd differencing is stationary")
    print(f"RECOMMENDATION: {recommendation}")

# ============================================
# 6. CREATE DATAFRAMES FOR SAVING
# ============================================

print("\n" + "="*60)
print("STEP 5: CREATE PROCESSED DATASETS")
print("="*60)

# Original data with date_time
if 'date_time' in df.columns:
    df_original = df[['date_time', 'traffic_volume']].copy()
else:
    df_original = pd.DataFrame({
        'traffic_volume': traffic_series
    })

# First differencing data
df_diff_1 = pd.DataFrame({
    'traffic_volume_diff_1': traffic_diff_1.values
})

# Second differencing data
df_diff_2 = pd.DataFrame({
    'traffic_volume_diff_2': traffic_diff_2.values
})

# Final stationary data (based on recommendation)
if d_order == 0:
    df_stationary = df_original.copy()
    df_stationary.columns = ['date_time', 'traffic_volume_stationary'] if 'date_time' in df_original.columns else ['traffic_volume_stationary']
elif d_order == 1:
    df_stationary = df_diff_1.copy()
    df_stationary.columns = ['traffic_volume_stationary']
else:
    df_stationary = df_diff_2.copy()
    df_stationary.columns = ['traffic_volume_stationary']

print(f"\nDatasets created:")
print(f"1. Original data shape: {df_original.shape}")
print(f"2. 1st Differenced data shape: {df_diff_1.shape}")
print(f"3. 2nd Differenced data shape: {df_diff_2.shape}")
print(f"4. Stationary data shape (d={d_order}): {df_stationary.shape}")

# ============================================
# 7. SAVE DATASETS
# ============================================

print("\n" + "="*60)
print("STEP 6: SAVING PROCESSED DATASETS")
print("="*60)

# Save original data
original_path = f"{output_dir}\\traffic_volume_original.csv"
df_original.to_csv(original_path, index=False)
print(f"\n✓ Original data saved: {original_path}")

# Save 1st differenced data
diff_1_path = f"{output_dir}\\traffic_volume_diff_1st_order.csv"
df_diff_1.to_csv(diff_1_path, index=False)
print(f"✓ 1st differenced data saved: {diff_1_path}")

# Save 2nd differenced data
diff_2_path = f"{output_dir}\\traffic_volume_diff_2nd_order.csv"
df_diff_2.to_csv(diff_2_path, index=False)
print(f"✓ 2nd differenced data saved: {diff_2_path}")

# Save stationary data
stationary_path = f"{output_dir}\\traffic_volume_stationary_d{d_order}.csv"
df_stationary.to_csv(stationary_path, index=False)
print(f"✓ Stationary data (d={d_order}) saved: {stationary_path}")

# ============================================
# 8. VISUALIZATION: STATIONARITY COMPARISON
# ============================================

print("\n" + "="*60)
print("STEP 7: GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# Original data
axes[0].plot(traffic_series.values, linewidth=1.5, color='#FF6B6B')
axes[0].set_title(f'Original Traffic Data (Non-Stationary, p={adf_original[1]:.6f})', 
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('Traffic Volume', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(traffic_series.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {traffic_series.mean():.0f}')
axes[0].legend()

# 1st differenced
axes[1].plot(traffic_diff_1.values, linewidth=1.5, color='#4ECDC4')
axes[1].set_title(f'1st Order Differenced Data (p={adf_diff1[1]:.6f})', 
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel('Difference', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='red', linestyle='--', alpha=0.7)

# 2nd differenced
axes[2].plot(traffic_diff_2.values, linewidth=1.5, color='#45B7D1')
axes[2].set_title(f'2nd Order Differenced Data (p={adf_diff2[1]:.6f})', 
                  fontsize=13, fontweight='bold')
axes[2].set_ylabel('2nd Difference', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].axhline(0, color='red', linestyle='--', alpha=0.7)

# Final stationary data
final_name = f"Stationary Data (d={d_order})" if d_order > 0 else "Original (Already Stationary)"
axes[3].plot(final_data.values, linewidth=1.5, color='#FFA07A')
axes[3].set_title(final_name, fontsize=13, fontweight='bold')
axes[3].set_ylabel('Traffic Volume', fontsize=11)
axes[3].set_xlabel('Time Steps', fontsize=11)
axes[3].grid(True, alpha=0.3)
axes[3].axhline(0 if d_order > 0 else final_data.mean(), color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
visualization_path = f"{output_dir}\\stationarity_analysis.png"
plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
print(f"✓ Stationarity comparison plot saved: {visualization_path}")
plt.show()

# ============================================
# 9. ACF AND PACF PLOTS
# ============================================

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Original data
plot_acf(traffic_series.dropna(), lags=50, ax=axes[0, 0])
axes[0, 0].set_title('ACF - Original Data', fontsize=12, fontweight='bold')

plot_pacf(traffic_series.dropna(), lags=50, ax=axes[0, 1])
axes[0, 1].set_title('PACF - Original Data', fontsize=12, fontweight='bold')

# 1st differenced
plot_acf(traffic_diff_1.dropna(), lags=50, ax=axes[1, 0])
axes[1, 0].set_title('ACF - 1st Differenced', fontsize=12, fontweight='bold')

plot_pacf(traffic_diff_1.dropna(), lags=50, ax=axes[1, 1])
axes[1, 1].set_title('PACF - 1st Differenced', fontsize=12, fontweight='bold')

# Final stationary
plot_acf(final_data.dropna(), lags=50, ax=axes[2, 0])
axes[2, 0].set_title(f'ACF - Stationary Data (d={d_order})', fontsize=12, fontweight='bold')

plot_pacf(final_data.dropna(), lags=50, ax=axes[2, 1])
axes[2, 1].set_title(f'PACF - Stationary Data (d={d_order})', fontsize=12, fontweight='bold')

plt.tight_layout()
acf_pacf_path = f"{output_dir}\\acf_pacf_analysis.png"
plt.savefig(acf_pacf_path, dpi=300, bbox_inches='tight')
print(f"✓ ACF/PACF analysis plot saved: {acf_pacf_path}")
plt.show()

# ============================================
# 10. SUMMARY REPORT
# ============================================

print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

summary_report = f"""
ARIMA & SARIMA DATA PREPROCESSING SUMMARY
{'='*60}

INPUT FILE: {input_path}
OUTPUT DIRECTORY: {output_dir}

STATIONARITY ANALYSIS:
{'='*60}
Original Data:
  - ADF Statistic: {adf_original[0]:.6f}
  - p-value: {adf_original[1]:.6f}
  - Status: {'STATIONARY' if adf_original[1] <= 0.05 else 'NON-STATIONARY'}

1st Order Differencing:
  - ADF Statistic: {adf_diff1[0]:.6f}
  - p-value: {adf_diff1[1]:.6f}
  - Status: {'STATIONARY' if adf_diff1[1] <= 0.05 else 'NON-STATIONARY'}

2nd Order Differencing:
  - ADF Statistic: {adf_diff2[0]:.6f}
  - p-value: {adf_diff2[1]:.6f}
  - Status: {'STATIONARY' if adf_diff2[1] <= 0.05 else 'NON-STATIONARY'}

RECOMMENDATION FOR ARIMA/SARIMA:
{'='*60}
Differencing Order (d): {d_order}
{recommendation}

USE THIS DATA FOR ARIMA: {stationary_path}

ARIMA ORDER PARAMETERS:
  - p: Check PACF plot (number of significant spikes)
  - d: {d_order} (as determined above)
  - q: Check ACF plot (number of significant spikes)

EXAMPLE ARIMA MODELS TO TRY:
  - ARIMA(1, {d_order}, 1)
  - ARIMA(2, {d_order}, 1)
  - ARIMA(1, {d_order}, 2)
  - ARIMA(2, {d_order}, 2)

EXAMPLE SARIMA MODEL:
  - SARIMA(1, {d_order}, 1) x (1, 1, 1, 24)
    (The 1, 1, 1, 24 handles seasonal differencing with 24-hour period)

FILES GENERATED:
{'='*60}
1. traffic_volume_original.csv - Original data (shape: {df_original.shape})
2. traffic_volume_diff_1st_order.csv - 1st differencing (shape: {df_diff_1.shape})
3. traffic_volume_diff_2nd_order.csv - 2nd differencing (shape: {df_diff_2.shape})
4. traffic_volume_stationary_d{d_order}.csv - RECOMMENDED for ARIMA (shape: {df_stationary.shape})
5. stationarity_analysis.png - Visual comparison
6. acf_pacf_analysis.png - ACF/PACF plots for parameter selection
7. data_preprocessing_report.txt - This summary report

{'='*60}
Generated on: {pd.Timestamp.now()}
"""

report_path = f"{output_dir}\\data_preprocessing_report.txt"
with open(report_path, 'w') as f:
    f.write(summary_report)

print(summary_report)
print(f"\n✓ Summary report saved: {report_path}")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)

print(f"\n📊 Use this data for ARIMA/SARIMA:")
print(f"   {stationary_path}")

print(f"\n📈 Check ACF/PACF plots to determine ARIMA parameters:")
print(f"   {acf_pacf_path}")

print("\n✓ All files saved to: " + output_dir)
