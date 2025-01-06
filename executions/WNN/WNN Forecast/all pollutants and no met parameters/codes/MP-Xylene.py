import numpy as np
import pandas as pd
import pywt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess the time series data
data = pd.read_excel(r"C:\Users\Vaishnavi G\Desktop\IITH OG\DATA\shadipur processed.xlsx")
  # Update with the actual path to your data file
dates = pd.to_datetime(data['From Date'])
target_variable = 'MP-Xylene (ug/m3)'  # Update with the desired target pollutant

# Filter the data for the desired date range
train_data = data[(dates >= '2013') & (dates <= '2020')]
test_data = data[(dates >= '2021') & (dates <= '11-02-2024')]

# Define the input pollutants
input_pollutants = ['NO (ug/m3)', 'NO2 (ug/m3)', 'NOx (ppb)', 'SO2 (ug/m3)',
                    'Benzene (ug/m3)', 'Benzene (ug/m3)', 'Toluene (ug/m3)', 'Eth-Benzene (ug/m3)',
                    'MP-Xylene (ug/m3)']

# Extract the target variable from the filtered data
target_train = train_data[target_variable].values
target_test = test_data[target_variable].values

# Define the number of time steps
n_steps = 10  # Number of time steps to consider

# Split the train data into input (X) and output (y)
def create_sequences(data, target, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(target[i+n_steps])
    return np.array(X), np.array(y)

# Create input sequences with multiple variables
X_train, y_train = create_sequences(train_data[input_pollutants].values, target_train, n_steps)

# Preprocess the input data using wavelet transform
wavelet_name = 'db4'  # Wavelet type
X_train_wavelet = []
for i in range(len(X_train)):
    coeffs = []
    for j in range(X_train.shape[2]):
        c = pywt.wavedec(X_train[i, :, j], wavelet_name, level=3)
        coeffs.extend(c)
    X_train_wavelet.append(np.concatenate(coeffs))
X_train_wavelet = np.array(X_train_wavelet)

# Train the WNN model
model = Sequential()
model.add(Dense(10, input_shape=(X_train_wavelet.shape[1],), activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train_wavelet, y_train, epochs=10, batch_size=32)

# Generate forecasts on the test data using walk-forward validation
X_test, y_test = create_sequences(test_data[input_pollutants].values, target_test, n_steps)
X_test_wavelet = []
y_pred = []
for i in range(len(X_test)):
    coeffs = []
    for j in range(X_test.shape[2]):
        c = pywt.wavedec(X_test[i, :, j], wavelet_name, level=3)
        coeffs.extend(c)
    X_test_wavelet.append(np.concatenate(coeffs))
    X_test_wavelet_arr = np.array(X_test_wavelet)

    # Generate forecast for the current step using the trained model
    y_pred_step = model.predict(X_test_wavelet_arr[i].reshape(1, -1))
    y_pred.append(y_pred_step[0][0])

# Reverse the wavelet transform on the forecasts if necessary

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a DataFrame with the forecasted and actual values
forecast_data = pd.DataFrame({'Date': test_data['From Date'][n_steps:], 'Actual': y_test, 'Forecast': y_pred})

# Write the DataFrame to an Excel file
forecast_data.to_excel('WNNMP-Xyleneforecast_results.xlsx', index=False)

# Plot the forecasted and actual values
fig = plt.figure(figsize=(12, 6))
plt.plot(test_data['From Date'][n_steps:], y_test, label='Actual')
plt.plot(test_data['From Date'][n_steps:], y_pred, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Forecast vs Actual for ' + target_variable)
plt.legend()
plt.show()
fig.savefig('WNNMP-Xyleneall.png')

# Print the evaluation metrics
print("RMSE:", rmse)
print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)
