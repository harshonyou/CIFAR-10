from tensorflow.keras.layers import Input, Conv2DTranspose, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

from config import NEURONS

def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x


def build_unet(shape, num_classes):
    inputs = Input(shape)

    """ Encoder """
    x1, p1 = conv_block(inputs, 16, pool=True)
    x2, p2 = conv_block(p1, 32, pool=True)
    z1 = Dropout(0.2)(p2)
    x3, p3 = conv_block(z1, 48, pool=True)
    x4, p4 = conv_block(p3, 64, pool=True)
    z2 = Dropout(0.3)(p4)

    """ Bridge """
    b1 = conv_block(z2, 128, pool=False)

    """ Decoder """
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 64, pool=False)

    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 48, pool=False)

    z3 = Dropout(0.4)(x6)

    u3 = UpSampling2D((2, 2), interpolation="bilinear")(z3)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False)

    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False)

    z4 = Dropout(0.5)(x8)

    """ Output layer """
    f1 = Flatten() (z4)
    d1 = Dense(512, activation="relu") (f1)
    d2 = Dense(num_classes, activation="softmax") (d1)
    # output = Conv2D(num_classes, 1, padding="same", activation="softmax")(d9)

    return Model(inputs, d2)


if __name__ == "__main__":
    model = build_unet((32, 32, 3), 10)
    # (None, 512, 512, 18) 1170
    model.summary()
