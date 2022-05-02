# This is the least trust worthy model.

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

def build_unet(input_shape, num_classes):
    inputs = KL.Input(shape=input_shape)

    c = KL.Conv2D(64, (3,3), padding="valid", activation=tf.nn.relu)(inputs)
    m = KL.MaxPool2D((2,2), (2,2)) (c)

    c = KL.Conv2D(128, (3,3), padding="valid", activation=tf.nn.relu)(m)
    m = KL.MaxPool2D((2,2), (2,2)) (c)

    c = KL.Conv2D(256, (3,3), padding="valid", activation=tf.nn.relu)(m)
    m = KL.MaxPool2D((2,2), (2,2)) (c)

    f = KL.Flatten() (m)
    outputs = KL.Dense(num_classes, activation=tf.nn.softmax) (f)

    model = KM.Model(inputs, outputs, name="U-Net")
    return model


if __name__ == "__main__":
    model = build_unet((32*2, 32*2, 1), 10)
    # (None, 512, 512, 18) 1170
    model.summary()
