import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

from data import load_data, load_train_and_test
# from model import build_unet
from cmodel import build_unet
from config import SHAPE, EPOCHS, CSVPATH, NUMBERSOFATTRIBUTES, LOSS, MODELSAVELOCATION

if __name__ == "__main__":
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
    num_attributes = NUMBERSOFATTRIBUTES
    loss = LOSS
    model_path = MODELSAVELOCATION

    """ Model """
    model = build_unet(shape, num_attributes)

    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(), metrics=["acc"])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3,
                          factor=0.1, verbose=1, min_lr=1e-6),
        CSVLogger(csv_path),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    ]

    model.fit(X_train, y_train,
              epochs=epochs,
              callbacks=callbacks
              )
