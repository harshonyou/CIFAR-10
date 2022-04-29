import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

from data import load_data, load_train_and_test
from model import build_unet

if __name__ == "__main__":
    """ Dataset """
    (_, _), (X_test, y_true) = load_train_and_test()
    print(f"Dataset: Test: {len(X_test)}")

    """ Model """
    model = tf.keras.models.load_model("model.h5")

    y_pred = model.predict(X_test)

    total_pred = len(y_true)
    true_pred = 0

    for i in range(total_pred):
        if y_true[i].argmax() == y_pred[i].argmax():
            true_pred += 1

    df = pd.DataFrame(columns=['Quality', 'Quantity'])
    df = df.append({'Quality': "Total Predictions", 'Quantity': total_pred}, ignore_index=True)
    df = df.append({'Quality': "True Predictions", 'Quantity': true_pred}, ignore_index=True)
    df = df.append({'Quality': "True Accuracy", 'Quantity': (true_pred/total_pred*100)}, ignore_index=True)
    df.to_csv('accuracy.csv', mode='a', index=False)

    print(true_pred/total_pred)