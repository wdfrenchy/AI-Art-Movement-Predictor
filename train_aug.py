import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load data from CSV
data = pd.read_csv('files/output_pixels.csv')
X = data.iloc[:, 1:].values  # Exclude the first column which contains the labels
y = data.iloc[:, 0].values  # Labels are in the first column

# Assuming images are RGB and each image is stored as a flat row of pixels (50x40 pixels in your example)
image_height = 40
image_width = 40
num_channels = 3  # RGB images

# Reshape the input data to be a proper image format
X_reshaped = X.reshape(-1, image_height, image_width, num_channels)

# Normalize pixel values to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[1])).reshape(X_reshaped.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Neural network model
model = Sequential([
    Flatten(input_shape=(image_height, image_width, num_channels)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (impressionism or not)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=10),
    validation_data=(X_test, y_test),
    epochs=50,
    verbose=2
)
model.save('my_model.keras')  # Saves the model to a HDF5 file

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)
