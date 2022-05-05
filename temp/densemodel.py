import tensorflow as tf

from tensorflow.keras import datasets, layers, models

def dense_model(shape, num_classes):
    model = models.Sequential()

    # model.add(layers.Conv2D(48, 3, activation='relu', padding='same', input_shape=shape))
    model.add(layers.Input(shape))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024,activation='relu'))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


if __name__ == "__main__":
    model = dense_model((32, 32, 3), 10)
    # (None, 512, 512, 18) 1170
    # for layer in model.layers[:-8]:
    #     layer.trainable=False

    # for layer in model.layers[-8:]:
    #     layer.trainable=True
    model.summary()
