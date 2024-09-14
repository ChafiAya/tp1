# Step 1: Preparation of the Environment

# Install necessary libraries if not already installed
# Uncomment and run the following lines if needed
# !pip install tensorflow keras matplotlib

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Loading and Normalizing the Dataset

# 1. Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Normalize the images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Method for normalizing images:
# We scale the pixel values to the range [0, 1] by dividing by 255.

# 3. Display an image from the dataset and its label
def display_image(image, label):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

# Example: Displaying the first image and its label
display_image(train_images[0], train_labels[0])

# Mapping labels to their corresponding class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Step 3: Creating the CNN Model

# 1. Reshape the images to include the color channel (required for CNN input)
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# 2. Create the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # Output layer for 10 classes
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Important parameters during model compilation:
# - Optimizer: 'adam' (adaptively adjusts learning rates)
# - Loss function: SparseCategoricalCrossentropy (suitable for multi-class classification)
# - Metrics: Accuracy (measures the proportion of correctly classified instances)

# Step 4: Training the Model

# 1. Train the model
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

# 2. Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Step 5: Visualizing Predictions

# 1. Make predictions on test data
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# 2. Function to display image, model prediction, and true label
def display_prediction(image, true_label, prediction):
    plt.figure(figsize=(5, 5))
    plt.imshow(image.squeeze(), cmap=plt.cm.gray)
    predicted_label = np.argmax(prediction)
    plt.title(f'True label: {class_names[true_label]}\nPredicted label: {class_names[predicted_label]}')
    plt.axis('off')
    plt.show()

# Example: Displaying the first image with its prediction
display_prediction(test_images[0], test_labels[0], predictions[0])

