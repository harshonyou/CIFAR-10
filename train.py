import sys
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

from data import load_data, load_train_and_test
from config import SHAPE, EPOCHS, CSVPATH, NUMBERSOFATTRIBUTES, LOSS, MODELSAVELOCATION, MODELEXTENSTION, CSVEXTENSION

from model import build_unet as a_unet
from bmodel import build_unet as b_unet
from cmodel import build_unet as c_unet
from updated_model import build_unet as geg_unet


def args_handler(args):
    if "-low" in args:
        return 1
    elif "-medium" in args:
        return 2
    elif "-high" in args:
        return 3
    elif "-updated" in args:
        return 4

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
        model = c_unet(shape, num_attributes)
    elif(model_choice == 2):
        model = b_unet(shape, num_attributes)
    elif(model_choice == 3):
        model = a_unet(shape, num_attributes)
    elif(model_choice == 4):
        model = geg_unet(shape, num_attributes)

    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(), metrics=["acc", "categorical_accuracy"])

    callbacks = [
        ModelCheckpoint(path + model_path + str(model_choice) + " " + dt_string + model_ext, verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="categorical_accuracy", patience=3,
                          factor=0.1, verbose=1, min_lr=1e-6),
        CSVLogger(path + csv_path + str(model_choice) +" " + dt_string + csv_ext),
        EarlyStopping(monitor="categorical_accuracy", patience=5, verbose=1)
    ]

    model.fit(X_train, y_train,
              epochs=epochs,
              callbacks=callbacks
              )
