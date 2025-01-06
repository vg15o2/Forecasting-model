import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pywt
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Load and preprocess the time series data
data = pd.read_excel(r"C:\Users\Vaishnavi G\Desktop\IITH OG\Final Model\DATA\shadipur processed.xlsx")
# Update with the actual filename or path
dates = pd.to_datetime(data['From Date'])
target = data['Benzene (ug/m3)'].values

# Filter the data for the desired date range
train_data = data[(dates >= '2013') & (dates <= '2020')]
test_data = data[(dates >= '2021') & (dates <= '2025')]

# Extract the target variable from the filtered data
target_train = train_data['Benzene (ug/m3)'].values
target_test = test_data['Benzene (ug/m3)'].values

# Perform any necessary preprocessing on the target variable

# Define the number of time steps
n_steps = 10  # Number of time steps to consider

# Split the train data into input (X) and output (y)
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(target_train, n_steps)

# Perform k-fold cross-validation
k = 5  # Number of folds for cross-validation
tscv = TimeSeriesSplit(n_splits=k)
fold = 1

for train_index, val_index in tscv.split(X_train):
    print("Fold:", fold)

    # Split the data into train and validation sets for the current fold
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Preprocess the input data using wavelet transform
    wavelet_name = 'db4'  # Wavelet type
    X_train_wavelet = []
    X_val_wavelet = []

    for i in range(len(X_train_fold)):
        coeffs = pywt.wavedec(X_train_fold[i], wavelet_name, level=3)
        X_train_wavelet.append(np.concatenate(coeffs))

    for i in range(len(X_val_fold)):
        coeffs = pywt.wavedec(X_val_fold[i], wavelet_name, level=3)
        X_val_wavelet.append(np.concatenate(coeffs))

    X_train_wavelet = np.array(X_train_wavelet)
    X_val_wavelet = np.array(X_val_wavelet)

    # Train the SVR model
    model = SVR(kernel='rbf')
    model.fit(X_train_wavelet, y_train_fold)

    # Generate forecasts on the test data
    X_test_wavelet = []
    for i in range(len(target_test) - n_steps + 1):
        test_input = target_test[i:i + n_steps]
        coeffs = pywt.wavedec(test_input, wavelet_name, level=3)
        X_test_wavelet.append(np.concatenate(coeffs))

    X_test_wavelet = np.array(X_test_wavelet)

    # Generate forecasts using the trained model
    y_pred = model.predict(X_test_wavelet)

    # Reverse the wavelet transform on the forecasts if necessary

    fold += 1

# Plot the forecasted and actual values
fig = plt.figure(figsize=(12, 6))
plt.plot(test_data['From Date'][n_steps-1:], target_test[n_steps-1:], label='Actual')
plt.plot(test_data['From Date'][n_steps-1:], y_pred, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Forecast vs Actual of Benzene'.format(fold))
plt.legend()
plt.show()
fig.savefig('SVRBenzene_fold{}.png'.format(fold))

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(target_test[n_steps-1:], y_pred))
mse = mean_squared_error(target_test[n_steps-1:], y_pred)
mae = mean_absolute_error(target_test[n_steps-1:], y_pred)
r2 = r2_score(target_test[n_steps-1:], y_pred)
mpe = np.mean((target_test[n_steps-1:] - y_pred) / target_test[n_steps-1:]) * 100
mape = np.mean(np.abs((target_test[n_steps-1:] - y_pred) / target_test[n_steps-1:])) * 100

print("RMSE:", rmse)
print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)
print("MPE:", mpe)
print("MAPE:", mape)

# Write the forecasted and actual values to an Excel file
forecast_data = pd.DataFrame({'Date': test_data['From Date'][n_steps-1:], 'Actual': target_test[n_steps-1:], 'Forecast': y_pred})
forecast_data.to_excel('SVRBenzeneforecast_results_fold{}.xlsx'.format(fold), index=False)