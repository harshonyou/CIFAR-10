"""
This script is dedicated to evaluating the trained model's performance on the CIFAR-10 test dataset.
By loading a specified trained model, it performs predictions on the test set. Subsequently, it computes
the accuracy, providing insights into how well the model generalizes to unseen data.
"""

import os
import tensorflow as tf
import pandas as pd

from data import load_train_and_test

if __name__ == "__main__":
    path = os.getcwd()

    """ Dataset """
    (_, _), (X_test, y_true) = load_train_and_test()
    X_test =  X_test.astype('float32') / 255
    print(f"Dataset: Test: {len(X_test)}")

    """ Model """
    print("Enter the name of the model you want to check the accuracy for")
    model = tf.keras.models.load_model(path+"/trained/"+input())

    """ Prediction """
    y_pred = model.predict(X_test)

    total_pred = len(y_true)
    true_pred = 0

    """ Calculation """
    for i in range(total_pred):
        if y_true[i].argmax() == y_pred[i].argmax():
            true_pred += 1

    """ Output """
    df = pd.DataFrame(columns=['Quality', 'Quantity'])
    df = df.append({'Quality': "Total Predictions", 'Quantity': total_pred}, ignore_index=True)
    df = df.append({'Quality': "True Predictions", 'Quantity': true_pred}, ignore_index=True)
    df = df.append({'Quality': "True Accuracy", 'Quantity': (true_pred/total_pred*100)}, ignore_index=True)
    df.to_csv(path+'/stats/'+'accuracy.csv', mode='a', index=False)

    print(true_pred/total_pred)