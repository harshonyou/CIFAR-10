import sys
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from data import load_train_and_test
from config import SHAPE, EPOCHS, CSVPATH, NUMBERSOFATTRIBUTES, LOSS, MODELSAVELOCATION, MODELEXTENSTION, CSVEXTENSION

from updated_model import build_unet as geg_unet


def args_handler(args):
    return 4
    # if "-low" in args:
    #     return 1
    # elif "-medium" in args:
    #     return 2
    # elif "-high" in args:
    #     return 3
    # elif "-updated" in args:
    #     return 4

if __name__ == "__main__":
    """ Arguments """
    args = sys.argv[1:]
    model_choice = args_handler(args)

    path = os.getcwd()
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y %H_%M_%S")

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Dataset """
    (X_train, y_train), (X_test, y_test) = load_train_and_test()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    print(
        f"Dataset: Train: {len(X_train)} - Test: {len(X_test)}")

    """ Hyperparameters """
    shape = SHAPE
    epochs = EPOCHS
    csv_path = CSVPATH
    csv_ext = CSVEXTENSION
    num_attributes = NUMBERSOFATTRIBUTES
    loss = LOSS
    model_path = MODELSAVELOCATION
    model_ext = MODELEXTENSTION

    """ Model """
    if(model_choice == 4):
        model = geg_unet(shape, num_attributes)

    # for layer in model.layers[:-7]:
    #     layer.trainable=False

    # for layer in model.layers[-7:]:
    #     layer.trainable=True

    # for l in model.layers:
    #     print(l.name, l.trainable)

    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                         zoom_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")

    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(), metrics=["acc", "categorical_accuracy"])

    callbacks = [
        ModelCheckpoint(path + model_path + str(model_choice) + " " + dt_string + model_ext, verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="categorical_accuracy", patience=3,
                          factor=0.1, verbose=1, min_lr=1e-6),
        CSVLogger(path + csv_path + str(model_choice) +" " + dt_string + csv_ext),
        EarlyStopping(monitor="categorical_accuracy", patience=5, verbose=1)
    ]

    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              steps_per_epoch=len(X_train) // 32,
              epochs=epochs,
              callbacks=callbacks
              )
