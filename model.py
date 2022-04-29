from tensorflow.keras.layers import Input, Conv2DTranspose, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

from config import NEURONS

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
    # x = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same')(x)
    # x = Conv2D(8, (3, 3), padding='same')(UpSampling2D(2)(x))

    # x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    # x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Conv2D(num_filters, (2, 2), padding='same')(UpSampling2D(2)(input))
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, NEURONS[0])
    s2, p2 = encoder_block(p1, NEURONS[1])
    s3, p3 = encoder_block(p2, NEURONS[2])
    s4, p4 = encoder_block(p3, NEURONS[3])

    b1 = conv_block(p4, NEURONS[4])

    d1 = decoder_block(b1, s4, NEURONS[3])
    d2 = decoder_block(d1, s3, NEURONS[2])
    d3 = decoder_block(d2, s2, NEURONS[1])
    d4 = decoder_block(d3, s1, NEURONS[0])

    # outputs = Conv2D(num_classes, 1, padding="same", activation="sigmoid")(d4)

    f = Flatten() (d4)
    outputs = Dense(num_classes) (f)
    outputs = Activation("softmax") (outputs)

    model = Model(inputs, outputs, name="U-Net")
    return model


if __name__ == "__main__":
    model = build_unet((32, 32, 1), 10)
    # (None, 512, 512, 18) 1170
    model.summary()
