"""
The main training script for the CIFAR-10 image classification task. This script integrates functionalities
from the associated modules, orchestrating the end-to-end training process. From data loading, preprocessing,
model construction, to the actual training phase, this script provides a holistic approach to model training.
"""

import sys
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from data import load_train_and_test
from config import SHAPE, EPOCHS, CSVPATH, NUMBERSOFATTRIBUTES, LOSS, MODELSAVELOCATION, MODELEXTENSTION, CSVEXTENSION

from model import build_model as builder

def args_handler(args):
    if "-test" in args:
        return 1
    else:
        return 2

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
    if(model_choice == 1):
        exit()
    if(model_choice == 2):
        model = builder(shape, num_attributes)

    """ Data Augmentation """
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                         zoom_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")

    validgen = ImageDataGenerator()

    """ Data Configuration """
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(), metrics=["acc", "categorical_accuracy"])

    callbacks = [
        ModelCheckpoint(path + model_path + str(model_choice) + " " + dt_string + model_ext, verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="loss", patience=3,
                          factor=0.1, verbose=1, min_lr=1e-6),
        CSVLogger(path + csv_path + str(model_choice) +" " + dt_string + csv_ext),
        EarlyStopping(monitor="loss", patience=5, verbose=1)
    ]

    """ Training """
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              validation_data = validgen.flow(X_test, y_test, batch_size=8),
              steps_per_epoch=len(X_train) // 32,
              validation_steps=len(X_test) // 8,
              epochs=epochs,
              callbacks=callbacks
              )
