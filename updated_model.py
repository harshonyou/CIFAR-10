import tensorflow as tf

from tensorflow.keras import datasets, layers, models

def build_unet(shape, num_classes):
    model = models.Sequential()
    # model.add(layers.Conv2D(48, kernel_size = 3, activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(48, 3, activation='relu', padding='same', input_shape=shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(48, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(48, 5, activation='relu', padding='same', strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, 5, activation='relu', padding='same', strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, 5, activation='relu', padding='same', strides=2))
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, 4, activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


if __name__ == "__main__":
    model = build_unet((32, 32, 1), 10)
    # (None, 512, 512, 18) 1170
    model.summary()
