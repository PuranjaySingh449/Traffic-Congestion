# ============================================
# LSTM Traffic Congestion Prediction
# Metro Interstate Traffic Volume Dataset
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================
# 1. DATA LOADING AND EXPLORATION
# ============================================

# Load the dataset
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

print("\n=== Dataset Information ===")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset info:\n{df.info()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nStatistical summary:\n{df.describe()}")

# ============================================
# 2. DATA PREPROCESSING
# ============================================

# Convert date_time to datetime
df['date_time'] = pd.to_datetime(df['date_time'])

# Sort by date_time
df = df.sort_values('date_time').reset_index(drop=True)

# Remove duplicates
df = df.drop_duplicates(subset=['date_time'], keep='first')

# Handle missing values
df = df.fillna(method='ffill').fillna(method='bfill')

print(f"\n=== After Cleaning ===")
print(f"Dataset shape: {df.shape}")

# ============================================
# 3. FEATURE ENGINEERING
# ============================================

print("\n=== Feature Engineering ===")

# Temporal features
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month
df['day_of_month'] = df['date_time'].dt.day
df['week_of_year'] = df['date_time'].dt.isocalendar().week
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding for hour (to capture 23->0 continuity)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Cyclical encoding for day of week
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Cyclical encoding for month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Lag features (previous traffic values)
for lag in [1, 2, 3, 6, 12, 24]:
    df[f'traffic_lag_{lag}'] = df['traffic_volume'].shift(lag)

# Rolling statistics
for window in [3, 6, 12, 24]:
    df[f'traffic_rolling_mean_{window}'] = df['traffic_volume'].rolling(window=window).mean()
    df[f'traffic_rolling_std_{window}'] = df['traffic_volume'].rolling(window=window).std()

# Weather interaction features
df['temp_rain'] = df['temp'] * df['rain_1h']
df['temp_snow'] = df['temp'] * df['snow_1h']

# Rush hour indicators
df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)

# Drop rows with NaN values created by lag/rolling features
df = df.dropna().reset_index(drop=True)

print(f"Features created: {df.shape[1]} features")
print(f"Final dataset shape: {df.shape}")

# ============================================
# 4. FEATURE SELECTION
# ============================================

# Select features for the model
feature_columns = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'temp', 'rain_1h', 'snow_1h', 'clouds_all',
    'traffic_lag_1', 'traffic_lag_2', 'traffic_lag_3', 'traffic_lag_6', 'traffic_lag_12', 'traffic_lag_24',
    'traffic_rolling_mean_3', 'traffic_rolling_mean_6', 'traffic_rolling_mean_12', 'traffic_rolling_mean_24',
    'traffic_rolling_std_3', 'traffic_rolling_std_6', 'traffic_rolling_std_12', 'traffic_rolling_std_24',
    'temp_rain', 'temp_snow',
    'is_morning_rush', 'is_evening_rush'
]

target_column = 'traffic_volume'

print(f"\nSelected features ({len(feature_columns)}):")
for i, col in enumerate(feature_columns, 1):
    print(f"{i}. {col}")

# ============================================
# 5. DATA NORMALIZATION
# ============================================

# Prepare data
data = df[feature_columns + [target_column]].values

# Split into train and test sets (80-20 split)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

print(f"\n=== Data Split ===")
print(f"Training data: {train_data.shape}")
print(f"Testing data: {test_data.shape}")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Save scaler for later use
import joblib
joblib.dump(scaler, 'traffic_scaler.pkl')

# ============================================
# 6. CREATE SEQUENCES FOR LSTM
# ============================================

def create_sequences(data, lookback=24):
    """
    Create sequences for LSTM input
    data: numpy array of shape (samples, features)
    lookback: number of previous time steps to use
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :-1])  # All features except target
        y.append(data[i, -1])  # Target (traffic_volume)
    return np.array(X), np.array(y)

# Define lookback window
LOOKBACK = 24  # Use past 24 hours to predict next hour

# Create sequences
X_train, y_train = create_sequences(train_data_scaled, LOOKBACK)
X_test, y_test = create_sequences(test_data_scaled, LOOKBACK)

print(f"\n=== Sequence Creation ===")
print(f"X_train shape: {X_train.shape}")  # (samples, lookback, features)
print(f"y_train shape: {y_train.shape}")  # (samples,)
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ============================================
# 7. BUILD LSTM MODEL
# ============================================

def build_lstm_model(input_shape, model_type='bidirectional'):
    """
    Build LSTM model for traffic prediction
    
    model_type: 'simple', 'stacked', 'bidirectional'
    """
    model = Sequential(name=f'LSTM_{model_type}')
    
    if model_type == 'simple':
        # Simple LSTM
        model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
    elif model_type == 'stacked':
        # Stacked LSTM
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        
    elif model_type == 'bidirectional':
        # Bidirectional LSTM (Best performance)
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(32, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Build the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (lookback, features)
model = build_lstm_model(input_shape, model_type='bidirectional')

print("\n=== Model Architecture ===")
model.summary()

# ============================================
# 8. DEFINE CALLBACKS
# ============================================

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Save best model
checkpoint = ModelCheckpoint(
    'best_traffic_lstm_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Reduce learning rate when loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stop, checkpoint, reduce_lr]

# ============================================
# 9. TRAIN THE MODEL
# ============================================

print("\n=== Training the Model ===")

BATCH_SIZE = 32
EPOCHS = 100

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# 10. EVALUATE THE MODEL
# ============================================

print("\n=== Model Evaluation ===")

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Denormalize predictions and actual values
# Create dummy array with correct shape for inverse transform
def denormalize(scaler, values):
    dummy = np.zeros((len(values), scaler.n_features_in_))
    dummy[:, -1] = values.flatten()
    denorm = scaler.inverse_transform(dummy)[:, -1]
    return denorm

y_train_actual = denormalize(scaler, y_train)
y_train_pred_denorm = denormalize(scaler, y_train_pred)

y_test_actual = denormalize(scaler, y_test)
y_test_pred_denorm = denormalize(scaler, y_test_pred)

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

train_metrics = calculate_metrics(y_train_actual, y_train_pred_denorm)
test_metrics = calculate_metrics(y_test_actual, y_test_pred_denorm)

print("\n--- Training Metrics ---")
for metric, value in train_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n--- Testing Metrics ---")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# ============================================
# 11. VISUALIZATION
# ============================================

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Training History
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_title('Model MAE During Training', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Predictions vs Actual (Test Set)
fig, ax = plt.subplots(figsize=(18, 6))

# Plot only first 1000 points for clarity
plot_range = min(1000, len(y_test_actual))
x_axis = range(plot_range)

ax.plot(x_axis, y_test_actual[:plot_range], label='Actual Traffic', linewidth=2, alpha=0.7)
ax.plot(x_axis, y_test_pred_denorm[:plot_range], label='Predicted Traffic', linewidth=2, alpha=0.7)
ax.set_title('Traffic Volume: Actual vs Predicted (Test Set)', fontsize=16, fontweight='bold')
ax.set_xlabel('Time Steps (Hours)', fontsize=12)
ax.set_ylabel('Traffic Volume', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Scatter Plot: Predicted vs Actual
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(y_test_actual, y_test_pred_denorm, alpha=0.5, s=10)
ax.plot([y_test_actual.min(), y_test_actual.max()], 
        [y_test_actual.min(), y_test_actual.max()], 
        'r--', linewidth=2, label='Perfect Prediction')
ax.set_title('Predicted vs Actual Traffic Volume', fontsize=16, fontweight='bold')
ax.set_xlabel('Actual Traffic Volume', fontsize=12)
ax.set_ylabel('Predicted Traffic Volume', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add R² score on plot
ax.text(0.05, 0.95, f'R² = {test_metrics["R2"]:.4f}', 
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Residual Plot
residuals = y_test_actual - y_test_pred_denorm

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].scatter(y_test_pred_denorm, residuals, alpha=0.5, s=10)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Traffic Volume', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Residual Value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Error Distribution
errors = np.abs(y_test_actual - y_test_pred_denorm)

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {errors.mean():.2f}')
ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median Error: {np.median(errors):.2f}')
ax.set_title('Distribution of Absolute Errors', fontsize=16, fontweight='bold')
ax.set_xlabel('Absolute Error', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 12. SAVE RESULTS
# ============================================

# Save model
model.save('final_traffic_lstm_model.h5')
print("\n=== Model saved as 'final_traffic_lstm_model.h5' ===")

# Save metrics to file
with open('model_metrics.txt', 'w') as f:
    f.write("=== LSTM Traffic Prediction Model Metrics ===\n\n")
    f.write("--- Training Metrics ---\n")
    for metric, value in train_metrics.items():
        f.write(f"{metric}: {value:.4f}\n")
    f.write("\n--- Testing Metrics ---\n")
    for metric, value in test_metrics.items():
        f.write(f"{metric}: {value:.4f}\n")

print("=== Metrics saved to 'model_metrics.txt' ===")

# ============================================
# 13. MAKE FUTURE PREDICTIONS (EXAMPLE)
# ============================================

def predict_future_traffic(model, scaler, last_sequence, n_hours=24):
    """
    Predict traffic for next n hours
    last_sequence: Last LOOKBACK hours of data (shape: lookback x features)
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_hours):
        # Predict next hour
        pred = model.predict(current_sequence.reshape(1, LOOKBACK, -1), verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence: remove first hour, add prediction at end
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, -1] = pred[0, 0]  # Update last position with prediction
    
    # Denormalize predictions
    predictions = np.array(predictions).reshape(-1, 1)
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, -1] = predictions.flatten()
    predictions_denorm = scaler.inverse_transform(dummy)[:, -1]
    
    return predictions_denorm

# Example: Predict next 24 hours
last_sequence = test_data_scaled[-LOOKBACK:, :-1]  # Last 24 hours of features
future_predictions = predict_future_traffic(model, scaler, last_sequence, n_hours=24)

print(f"\n=== Next 24 Hours Traffic Prediction ===")
for i, pred in enumerate(future_predictions, 1):
    print(f"Hour {i}: {pred:.0f} vehicles")

# Plot future predictions
plt.figure(figsize=(12, 6))
plt.plot(range(1, 25), future_predictions, marker='o', linewidth=2, markersize=8)
plt.title('Predicted Traffic Volume for Next 24 Hours', fontsize=16, fontweight='bold')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Predicted Traffic Volume', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('future_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== LSTM Traffic Prediction Complete! ===")
