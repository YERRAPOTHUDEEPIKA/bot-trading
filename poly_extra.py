import numpy as np
#pip install scikit-learn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=120, include_bias=False)

# Assuming x_data and y_data are your observed data
#x_data = np.array([1,2,3,4,5,6,7])
#y_data = np.array([23, 64, 23, 56, 78, 92, 94])

df = pd.read_csv('META.csv')
x_data = np.array(df['Xvalues'].tolist())
y_data = np.array(df['Yvalues'].tolist())
#print(x_data)
#print(y_data)

# Reshape x_data to be a column vector
#x_data = x_data.reshape(-1, 1)
poly_features = poly.fit_transform((x_data).reshape(-1, 1))

poly_reg_model = LinearRegression()
# Fit the linear model
poly_reg_model.fit(poly_features, (y_data))


#model = LinearRegression().fit(np.log(x_data), y_data)
#new_x_values = np.array([101])
#new_x_values = new_x_values.reshape(-1, 1)
y_predicted = poly_reg_model.predict(poly_features)

print(y_predicted[len(y_predicted)-3])

# Print or use predictions as needed
plt.figure(figsize=(10,6))
plt.title('Sample regression')
plt.plot(x_data, y_data, c="green")
plt.plot(x_data, y_predicted, c="red")
plt.show()

