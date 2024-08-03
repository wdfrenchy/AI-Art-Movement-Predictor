import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Path to the new CSV file
new_csv_path = '/Users/frenwd24/Desktop/AIPython/image_nn/output_sample_pixels.csv'

# Load new data from CSV
new_data = pd.read_csv(new_csv_path)
X_new = new_data.iloc[:, 1:].values  # Assume first column is label and rest are pixel values
y_new = new_data.iloc[:, 0].values  # Assume first column contains the labels

# Reshape the input data to be a proper image format (adjust dimensions as needed)
image_height = 40
image_width = 40
num_channels = 3  # RGB images
X_new_reshaped = X_new.reshape(-1, image_height, image_width, num_channels)

# Normalize pixel values to [0, 1] (use the same range as the training data)
scaler = MinMaxScaler()
X_new_scaled = scaler.fit_transform(X_new.reshape(-1, X_new.shape[1])).reshape(X_new_reshaped.shape)

# Load the previously saved model
model = load_model('my_model.keras')


# Evaluate the model on the new data
loss, accuracy = model.evaluate(X_new_scaled, y_new)
print("Accuracy on new data:", accuracy)

# Make predictions and interpret them
predictions = model.predict(X_new_scaled)
predicted_labels = ['Impressionism' if p > 0.5 else 'Non-Impressionism' for p in predictions]

# Display the actual labels vs predicted labels
actual_labels = ['Impressionism' if y == 1 else 'Non-Impressionism' for y in y_new]
for actual, predicted in zip(actual_labels, predicted_labels):
    print(f"Actual: {actual}, Predicted: {predicted}, {'Correct' if actual == predicted else 'Incorrect'}")
