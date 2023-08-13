"""
Here, we specify the architecture of our neural network tailored for CIFAR-10 image classification.
The model leverages Convolutional Neural Networks (CNNs) and associated layers, capturing intricate patterns
in the images. The architecture defined in this script is crucial for the model's performance on the dataset.
"""

from keras import layers, models
from config import NEURONS

def build_model(shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(filters = NEURONS[0], kernel_size = (3, 3), strides = (1, 1), padding = 'same', input_shape=shape, name="1"))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters = NEURONS[0], kernel_size = (3, 3), strides = (1, 1), padding='same', name="2"))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters = NEURONS[0], kernel_size = (3, 3), strides = (1, 1), padding='same', name="3"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(strides = (2, 2), padding='valid'))

    model.add(layers.Conv2D(filters = NEURONS[1], kernel_size = (3, 3), strides = (1, 1), padding = 'same', name="4"))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters = NEURONS[1], kernel_size = (3, 3), strides = (1, 1), padding='same', name="5"))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters = NEURONS[1], kernel_size = (3, 3), strides = (1, 1), padding='same', name="6"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(strides = (2, 2), padding='valid'))

    model.add(layers.Conv2D(filters = NEURONS[1], kernel_size = (3, 3), strides = (1, 1),  padding = 'same', name="7"))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters = NEURONS[1], kernel_size = (1, 1), strides = (1, 1), padding='valid', name="8"))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(filters = NEURONS[2], kernel_size = (1,1), padding='valid', name="9"))
    model.add(layers.Dropout(rate = 0.5))
    model.add(layers.Flatten())

    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate = 0.5))
    model.add(layers.Dense(units = 1024, activation='relu', name="10"))
    model.add(layers.Dense(units = 512, activation='relu', name="11"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate = 0.5))

    model.add(layers.Dense(units = num_classes, activation='softmax', name="12"))

    return model

if __name__ == "__main__":
    model = build_model((32, 32, 3), 10)

    for layer in model.layers[:-8]:
        layer.trainable=False

    for layer in model.layers[-8:]:
        layer.trainable=True

    for layer in model.layers:
        print(layer.name, layer.trainable)

    model.summary()
