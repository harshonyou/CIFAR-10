import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

def get_database():
    return datasets.cifar10.load_data()

def neautralize_x(X_train, X_test):
    return X_train, X_test


def neautralize_y(y_train, y_test, num_classes):
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    return y_train, y_test


def load_train_and_test():
    (X_train, y_train), (X_test, y_test) = get_database()
    (X_train, X_test) = neautralize_x(X_train, X_test)
    (y_train, y_test) = neautralize_y(y_train, y_test, 10)
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    print('ayo!')