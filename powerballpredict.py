import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("powerball.csv")

# Extract the required columns
required_data = data[["Num1", "Num2", "Num3", "Num4", "Num5", "Powerball"]]

# Preprocess dataset: separate main numbers and powerball number
main_numbers = required_data[["Num1", "Num2", "Num3", "Num4", "Num5"]]
powerball_number = required_data["Powerball"]

# Scale main numbers between 1 and 69, powerball number between 1 and 26
scaler_main = MinMaxScaler(feature_range=(1, 69))
scaler_powerball = MinMaxScaler(feature_range=(1, 26))

scaled_main_numbers = scaler_main.fit_transform(main_numbers)
scaled_powerball_number = scaler_powerball.fit_transform(np.array(powerball_number).reshape(-1, 1))

# Join back the scaled columns
scaled_data = np.column_stack((scaled_main_numbers, scaled_powerball_number))

# Split into training and testing data
train_data, test_data = train_test_split(scaled_data, test_size=0.2)

# Reshape to [samples, time_steps, n_features]
train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))

# Create model
model = keras.Sequential()
model.add(keras.layers.LSTM(50, input_shape=(train_data.shape[1], train_data.shape[2])))
model.add(keras.layers.Dense(train_data.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(train_data, train_data, epochs=20, batch_size=1, verbose=2)

# Generate prediction
prediction = model.predict(test_data)

# Convert the scaled prediction back to original range and round it to nearest integer
prediction = np.rint(np.column_stack((scaler_main.inverse_transform(prediction[:, :5]), scaler_powerball.inverse_transform(prediction[:, 5:]))))

# Display the first five predicted results
for i in range(len(prediction)):
    # Order main numbers
    ordered_prediction = np.sort(prediction[i][:5])
    # Ensure all main numbers are unique and within valid range by replacing invalid ones
    while len(set(ordered_prediction)) != len(ordered_prediction) or (ordered_prediction < 1).any() or (ordered_prediction > 69).any():
        # Find first invalid number (either duplicate or out of range)
        invalid = next(x for x in ordered_prediction if ordered_prediction.tolist().count(x) > 1 or x < 1 or x > 69)
        # Replace it with a new random number within the valid range
        ordered_prediction[ordered_prediction.tolist().index(invalid)] = np.random.randint(1, 70)
    # Re-sort main numbers after potential changes
    ordered_prediction = np.sort(ordered_prediction)
    # Make sure Powerball number is within valid range
    powerball = prediction[i][5]
    while powerball < 1 or powerball > 26:
        # Generate a new random number within the valid range
        powerball = np.random.randint(1, 27)
    # Append the Powerball number (without sorting)
    final_prediction = np.append(ordered_prediction, powerball)
    print(f"Predicted Draw {i+1}: {final_prediction}")
