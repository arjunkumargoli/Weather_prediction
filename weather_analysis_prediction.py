
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample historical weather data (5 years of monthly average temperatures)
np.random.seed(42)
months = pd.date_range(start="2019-01-01", periods=60, freq='M')
temperatures = 10 + np.sin(np.linspace(0, 6 * np.pi, 60)) * 10 + np.random.normal(0, 2, 60)

weather_df = pd.DataFrame({
    'Date': months,
    'AvgTemperature': temperatures
})

# Convert Date to ordinal for regression
weather_df['DateOrdinal'] = weather_df['Date'].map(pd.Timestamp.toordinal)

# Train a linear regression model
X = weather_df[['DateOrdinal']]
y = weather_df['AvgTemperature']
model = LinearRegression()
model.fit(X, y)

# Predict next 12 months
future_dates = pd.date_range(start="2024-01-01", periods=12, freq='M')
future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
predictions = model.predict(future_ordinals)

# Create prediction DataFrame
prediction_df = pd.DataFrame({
    'Date': future_dates,
    'PredictedAvgTemperature': predictions
})

# Save datasets
weather_df[['Date', 'AvgTemperature']].to_csv("historical_weather.csv", index=False)
prediction_df.to_csv("future_predictions.csv", index=False)

# Optional: Plot the results
plt.figure(figsize=(10, 5))
plt.plot(weather_df['Date'], weather_df['AvgTemperature'], label='Historical Data')
plt.plot(prediction_df['Date'], prediction_df['PredictedAvgTemperature'], label='Predicted Data', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.title('Weather Data Analysis and Prediction')
plt.legend()
plt.tight_layout()
plt.savefig("temperature_trend.png")
plt.show()
