# import numpy as np
# from sklearn.linear_model import Ridge
# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# # Read data from CSV file
# df = pd.read_csv('AMZN(Aug 1-Sep1).csv')

# # Extract X and Y values from the DataFrame
# x_data = np.array(df['Xvalues'])
# y_data = np.array(df['Yvalues'])

# # Scale the data
# scaler = StandardScaler()
# x_data_scaled = scaler.fit_transform(x_data.reshape(-1, 1))

# # Prepare data for polynomial regression
# poly = PolynomialFeatures(degree=7, include_bias=False)
# poly_features = poly.fit_transform(x_data_scaled)

# # Create and fit the Ridge regression model
# ridge_reg_model = Ridge(alpha=1.0)
# ridge_reg_model.fit(poly_features, y_data)

# # Predict values for the first 7 days of the next month
# next_month_start_day = x_data[-1] + 1
# next_month_x_values = np.arange(next_month_start_day, next_month_start_day + 7)
# next_month_x_values_scaled = scaler.transform(next_month_x_values.reshape(-1, 1))
# next_month_x_features = poly.transform(next_month_x_values_scaled)
# next_month_y_predicted = ridge_reg_model.predict(next_month_x_features)

# # Print or use the predictions for the next 7 days
# for day, predicted_value in zip(next_month_x_values, next_month_y_predicted):
#     print(f"Predicted value for day {int(day)}: {predicted_value}")



import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Read data from CSV file
df = pd.read_csv('DIS.csv')

# Extract relevant features (e.g., date, close, high, low)
features = df[['Date', 'Close']].copy()  # Make sure the column names match your CSV file and use copy()

# Convert date to datetime object using .loc
features.loc[:, 'Date'] = pd.to_datetime(features['Date'])

# Create a feature representing days since the start
min_date = features['Date'].min()
features['DaysSinceStart'] = features['Date'].apply(lambda x: (x - min_date).days)

# Use the past month's data for training
train_data = features.tail(30)

# Prepare data for polynomial regression
x_data = np.array(train_data['DaysSinceStart']).reshape(-1, 1)
close_data = np.array(train_data['Close'])

poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(x_data)

# Create and fit the polynomial regression model for close prices
close_reg_model = LinearRegression()
close_reg_model.fit(poly_features, close_data)

# Predict the close price for the next day
next_day = features['DaysSinceStart'].max() + 1
next_day_feature = poly.transform(np.array([[next_day]]))
next_day_close_predicted = close_reg_model.predict(next_day_feature)

print(f"Predicted close price for the next day: {next_day_close_predicted[0]}")

# Optionally, plot the historical close prices and the regression line
plt.figure(figsize=(10,6))
plt.scatter(features['DaysSinceStart'], features['Close'], label='Historical Close Prices', color='blue')
plt.plot(x_data, close_reg_model.predict(poly_features), label='Polynomial Regression (Close)', color='red')
plt.scatter(next_day, next_day_close_predicted, label='Predicted Close Value', color='green', marker='x')
plt.xlabel('Days Since Start')
plt.ylabel('Close Price')
plt.legend()
plt.show()


