import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data creation
# Replace this with your real consumption data
# Suppose we have monthly consumption data (e.g., energy consumption) over 24 months
data = {
    'Month': pd.date_range(start='2022-01-01', periods=24, freq='M'),
    'Consumption': [300, 320, 310, 330, 360, 380, 390, 400, 420, 410, 430, 450,
                    470, 460, 480, 490, 500, 520, 510, 530, 550, 540, 560, 580]
}

df = pd.DataFrame(data)

# Feature engineering
# Convert month to numeric for the linear regression model
df['Month_numeric'] = np.arange(len(df))

# Split the data into training and testing sets
X = df[['Month_numeric']]
y = df['Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Consumption'], label='Actual Consumption', marker='o')
plt.plot(df['Month'].iloc[-len(y_test):], y_pred, label='Predicted Consumption', linestyle='--', marker='x')

plt.title('Consumption Forecast')
plt.xlabel('Month')
plt.ylabel('Consumption')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Forecast future consumption
# Let's say we want to predict the consumption for the next 6 months
future_months = np.arange(len(df), len(df) + 6).reshape(-1, 1)
future_consumption = model.predict(future_months)

print("Future consumption predictions for the next 6 months:")
for i, pred in enumerate(future_consumption):
    print(f"Month {i+1}: {pred}")
