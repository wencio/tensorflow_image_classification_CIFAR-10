# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define class names for the CIFAR-10 labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Create a figure to display some training images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)  # Create 5x5 subplots
    plt.xticks([])  # Hide x-axis ticks
    plt.yticks([])  # Hide y-axis ticks
    plt.grid(False)  # Hide grid lines
    plt.imshow(train_images[i])  # Display the image
    # CIFAR labels are arrays, so an extra index is needed
    plt.xlabel(class_names[train_labels[i][0]])  # Set the label as the class name
plt.show()  # Display the figure

# Create a sequential model
model = models.Sequential()
# Add a Conv2D layer with 32 filters, a 3x3 kernel, ReLU activation, and input shape of 32x32x3
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# Add a MaxPooling2D layer with a 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))
# Add another Conv2D layer with 64 filters and a 3x3 kernel
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Add another MaxPooling2D layer
model.add(layers.MaxPooling2D((2, 2)))
# Add another Conv2D layer with 64 filters and a 3x3 kernel
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Display the model summary so far
model.summary()

# Flatten the output of the previous layers to create a dense layer
model.add(layers.Flatten())
# Add a dense layer with 64 neurons and ReLU activation
model.add(layers.Dense(64, activation='relu'))
# Add the output dense layer with 10 neurons (one for each class) without final activation
model.add(layers.Dense(10))

# Display the complete model summary
model.summary()

# Compile the model using Adam optimizer, categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model with the training data for 10 epochs and validate with the test data
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Plot the accuracy and validation accuracy over the epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')  # Label for x-axis
plt.ylabel('Accuracy')  # Label for y-axis
plt.ylim([0.5, 1])  # Set y-axis limits
plt.legend(loc='lower right')  # Display legend in the lower right corner
plt.show()  # Show the plot

# Evaluate the model on the test data and print the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)  # Print the test accuracy
