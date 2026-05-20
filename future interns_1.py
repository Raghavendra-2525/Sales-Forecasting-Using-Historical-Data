import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Step 1: Simulate Historical Business Data (2 Years)
# ---------------------------------------------------------
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

# Simulate a business with an upward growth trend and weekend sales spikes
trend = np.linspace(50, 150, len(dates)) 
seasonality = np.where(dates.weekday >= 5, 40, 0) # +40 extra units sold on weekends
noise = np.random.normal(0, 10, len(dates))       # Random day-to-day variance

sales = trend + seasonality + noise
df = pd.DataFrame({'Date': dates, 'Sales': sales})

# ---------------------------------------------------------
# Step 2: Data Cleaning & Time-Based Feature Engineering
# ---------------------------------------------------------
df.dropna(inplace=True) # Handle missing values

# Create features so the model can understand time, trends, and seasons
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Year'] = df['Date'].dt.year

# ---------------------------------------------------------
# Step 3: Train the Forecasting Model
# ---------------------------------------------------------
X = df[['DayOfWeek', 'Month', 'DayOfYear', 'Year']]
y = df['Sales']

# Chronological Split: Train on first 80% of time, test on the final 20%
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
dates_train, dates_test = df['Date'][:split_index], df['Date'][split_index:]

# Random Forest is excellent at capturing non-linear patterns like weekly seasonality
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 4: Model Evaluation and Error Analysis
# ---------------------------------------------------------
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} units")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} units\n")

# ---------------------------------------------------------
# Step 5: Business-Friendly Visualization
# ---------------------------------------------------------
plt.figure(figsize=(14, 6))

# Plot historical, actual future, and predicted future data
plt.plot(dates_train, y_train, label='Historical Training Data', color='lightgray', alpha=0.8)
plt.plot(dates_test, y_test, label='Actual Future Sales', color='blue', alpha=0.5)
plt.plot(dates_test, predictions, label='Forecasted Sales (Model)', color='red', linestyle='--')

plt.title('Daily Sales Forecast vs Actual Demand', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Units Sold', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)

# Highlight a business insight on the chart
plt.annotate('Model successfully captures\nweekend sales spikes', 
             xy=(dates_test.iloc[20], predictions[20]), 
             xytext=(dates_test.iloc[20], predictions[20] + 50),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10)

plt.tight_layout()
plt.show()