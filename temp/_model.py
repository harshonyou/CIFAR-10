from tensorflow.keras.layers import Input, Conv2DTranspose, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

import tensorflow as tf

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    # s1, p1 = encoder_block(inputs, 64)
    # s2, p2 = encoder_block(p1, 128)
    # s3, p3 = encoder_block(p2, 256)
    # s4, p4 = encoder_block(p3, 512)

    # b1 = conv_block(p4, 1024)

    # d1 = decoder_block(b1, s4, 512)
    # d2 = decoder_block(d1, s3, 256)
    # d3 = decoder_block(d2, s2, 128)
    # d4 = decoder_block(d3, s1, 64)

    s1, p1 = encoder_block(inputs, 48)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 86)
    s4, p4 = encoder_block(p3, 124)

    b1 = conv_block(p4, 256)

    d1 = decoder_block(b1, s4, 124)
    d2 = decoder_block(d1, s3, 86)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 48)

    # outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    f = Flatten() (d4)
    outputs = Dense(10) (f)
    outputs = Activation("softmax") (outputs)
    # outputs = tf.
    #  outputs = KL.Dense(10, activation=tf.nn.softmax) (f)

    model = Model(inputs, outputs, name="U-Net")
    return model

if __name__ == "__main__":
    model = build_unet((32, 32, 3))
    model.summary()