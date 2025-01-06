import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and preprocess the time series data
# Assuming you have your time series data loaded into a variable named 'data'
data = pd.read_excel(r"C:\Users\Vaishnavi G\Desktop\IITH OG\Final Model\DATA\shadipur processed.xlsx")
# Update with the actual filename or path

# Extract the 'From Date' and 'Toluene (ug/m3)' columns
dates = pd.to_datetime(data['From Date'])
target = data['Toluene (ug/m3)'].values

# Filter the data for the desired date range
train_data = data[(dates >= '2013') & (dates <= '2020')]
test_data = data[(dates >= '2021') & (dates <= '2025')]

# Extract the target variable from the filtered data
target_train = train_data['Toluene (ug/m3)'].values
target_test = test_data['Toluene (ug/m3)'].values

# Perform any necessary preprocessing on the target variable

# Normalize the target variable
scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled_train = scaler.fit_transform(target_train.reshape(-1, 1))
target_scaled_test = scaler.transform(target_test.reshape(-1, 1))

# Define the number of time steps
n_steps = 10  # Number of time steps to consider

# Split the train data into input (X) and output (y)
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(target_scaled_train, n_steps)

# Reshape the input data to be 2-dimensional
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

# Define the MLP model
model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', random_state=42)

# Perform k-fold cross-validation
k = 5  # Number of folds for cross-validation
tscv = TimeSeriesSplit(n_splits=k)
fold = 1

for train_index, val_index in tscv.split(X_train):
    print("Fold:", fold)

    # Split the data into train and validation sets for the current fold
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Train the model
    model.fit(X_train_fold, y_train_fold)

    # Generate forecasts on the test data
    X_test, y_test = create_sequences(target_scaled_test, n_steps)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    y_pred = model.predict(X_test)

    # Reverse the scaling on the forecasts and actual values
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))



    fold += 1

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mpe = np.mean((y_test - y_pred) / y_test) * 100
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("RMSE:", rmse)
print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)
print("MPE:", mpe)
print("MAPE:", mape)
    
import matplotlib.pyplot as plt

# Plot the forecasted and actual values
fig=plt.figure(figsize=(12, 6))
plt.plot(test_data['From Date'].values[n_steps:], y_test.flatten(), label='Actual')
plt.plot(test_data['From Date'].values[n_steps:], y_pred.flatten(), label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Comparison of Forecast and Actual Values of Toluene')
plt.legend()
plt.show()
fig.savefig('MLPToluene.png')

import pandas as pd

# Extract the 'From Date' column for the test data
dates_test = test_data['From Date'].values[n_steps:]

# Create a DataFrame to store the forecasted, actual, and date values
results = pd.DataFrame({'Date': dates_test, 'Actual': y_test.flatten(), 'Forecast': y_pred.flatten()})

# Write the DataFrame to an Excel file
results.to_excel('MLPTolueneforecast_results.xlsx', index=True)



