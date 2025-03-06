# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.linear_model import Ridge

# Load Dataset (Replace 'your_dataset.csv' with the actual file path)
from google.colab import files
uploaded = files.upload()
data = pd.read_csv(list(uploaded.keys())[0])

# Data Preprocessing
# Handle missing values
data = data.dropna()

# Feature Engineering
data['Range'] = data['High'] - data['Low']
data['Volatility'] = (data['High'] - data['Low']) / data['Open']
data['Momentum'] = data['Close'] - data['Open']
data['Days_To_Expiry'] = (pd.to_datetime(data['Expiry Date']) - pd.to_datetime(data['Trade Date'])).dt.days

# Prepare features and target
X = data[['Open', 'High', 'Low', 'Volume', 'Range', 'Volatility', 'Momentum', 'Days_To_Expiry']]
y = data['Close']

# Normalize/Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.10, random_state=42)

# Model 1: XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds_train = xgb_model.predict(X_train)
xgb_preds_test = xgb_model.predict(X_test)

# Model 2: Neural Network
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

nn_preds_train = nn_model.predict(X_train).ravel()
nn_preds_test = nn_model.predict(X_test).ravel()

# Ensemble (Weighted Averaging)
ensemble_train_preds = 0.6 * xgb_preds_train + 0.4 * nn_preds_train
ensemble_test_preds = 0.6 * xgb_preds_test + 0.4 * nn_preds_test

# Stacking with Ridge Regression
stack_train = np.column_stack((xgb_preds_train, nn_preds_train))
stack_test = np.column_stack((xgb_preds_test, nn_preds_test))

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(stack_train, y_train)
stack_preds_test = ridge_model.predict(stack_test)

# Evaluation Metrics
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

# Evaluate Individual and Ensemble Models
evaluate(y_test, xgb_preds_test, "XGBoost")
evaluate(y_test, nn_preds_test, "Neural Network")
evaluate(y_test, ensemble_test_preds, "Weighted Ensemble")
evaluate(y_test, stack_preds_test, "Stacked Model")


