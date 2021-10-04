# Created By: Xavier De Carvalho
# Created On: 04/10/2021
# Updated By:
# Updated On:
# Version: FMCNN0.0.01
# Reference: https://www.coursera.org/learn/introduction-tensorflow

# Import Packages
import tensorflow as tf
print(tf.__version__)

# Stop training when accuracy is above desired value
class accuracy_stop(tf.keras.callbacks.Callback):
    def __init__(self):
        self.training_threshold = 0.9

    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') and logs.get('accuracy') >= self.training_threshold):
            print(f'\nAccuracy reached {int(self.training_threshold*100)}% so cancelling training!')
            self.model.stop_training = True

callbacks = accuracy_stop()

# Get MNIST Fashion Data
mnist = tf.keras.datasets.fashion_mnist

# Training and Test sets
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# Build Model
model = tf.keras.models.Sequential([
    # Conv2D(num_conv, size_conv, act_func, shape_input_data)
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Shape not required in this Conv2D layer - only the first layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train Model
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# Evaluate Model
test_loss = model.evaluate(test_images, test_labels)