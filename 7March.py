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
# Calculate Technical Indicators
def calculate_technical_indicators(data):
    # Exponential Moving Averages (EMA)
    data['EMA_7'] = data['Close'].ewm(span=7, adjust=False).mean()
    data['EMA_14'] = data['Close'].ewm(span=14, adjust=False).mean()
    
    # Simple Moving Averages (SMA)
    data['SMA_7'] = data['Close'].rolling(window=7).mean()
    
    # Moving Average Convergence Divergence (MACD)
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Momentum
    data['Momentum'] = data['Close'] - data['Close'].shift(4)
    
    # Bollinger Bands
    data['Bollinger_Mid'] = data['Close'].rolling(window=20).mean()
    data['Bollinger_Std'] = data['Close'].rolling(window=20).std()
    data['Bollinger_Upper'] = data['Bollinger_Mid'] + (2 * data['Bollinger_Std'])
    data['Bollinger_Lower'] = data['Bollinger_Mid'] - (2 * data['Bollinger_Std'])
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# Calculate Technical Indicators
data = calculate_technical_indicators(data)

# Drop rows with NaN values (created by rolling calculations)
data = data.dropna()

# Select Features
features = [
    'Strike Price', 'Open', 'High', 'Low', 'Close', 'Volume',  # Basic features
    'EMA_7', 'MACD', 'EMA_14', 'SMA_7', 'Momentum',           # Technical indicators
    'Bollinger_Upper', 'RSI', 'Bollinger_Mid', 'Bollinger_Std', 'MACD_Signal'
]

# Prepare features and target
X = data[features]
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
ensemble_test_preds = 0.6 * xgb_preds_test + 0.4 * np.squeeze(nn_preds_test)

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
