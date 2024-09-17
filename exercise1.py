# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Define constants
EPOCHS = 5
BATCH_SIZE = 64
IMG_HEIGHT, IMG_WIDTH = 28, 28
NUM_CLASSES = 10

# Step 1: Load and preprocess data
def load_and_preprocess_data():
    # Load the Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize the images to the range [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Add a color channel dimension
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    return (train_images, train_labels), (test_images, test_labels)

# Step 2: Create CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Step 3: Train the model
def train_model(model, train_images, train_labels, test_images, test_labels):
    history = model.fit(
        train_images, train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(test_images, test_labels),
        verbose=2  # Provides a progress bar during training
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    
    return history

# Step 4: Plot training results
def plot_results(history):
    plt.figure(figsize=(12, 6))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.tight_layout()
    plt.show()

# Step 5: Make and visualize predictions
def make_and_visualize_predictions(model, test_images, test_labels):
    # Create a probability model
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    def display_prediction(image, true_label, prediction, index):
        plt.figure(figsize=(5, 5))
        plt.imshow(image.squeeze(), cmap=plt.cm.gray)
        predicted_label = np.argmax(prediction)
        plt.title(f'Index {index}\nTrue label: {class_names[true_label]}\nPredicted label: {class_names[predicted_label]}')
        plt.axis('off')
        plt.savefig(f'prediction_{index}.png')
        plt.close()

    # Example: Displaying the first prediction image
    display_prediction(test_images[0], test_labels[0], predictions[0], 0)

# Mapping labels to their corresponding class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Execute steps
(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
model = create_model()
history = train_model(model, train_images, train_labels, test_images, test_labels)
plot_results(history)
make_and_visualize_predictions(model, test_images, test_labels)
