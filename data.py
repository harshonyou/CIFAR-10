"""
This utility script focuses on operations related to the CIFAR-10 dataset. It provides functionalities
to load the dataset, preprocess the images and labels, and prepare them for training and evaluation.
By leveraging Keras's utilities, it ensures seamless data operations, allowing the main training script
to focus solely on the model training aspect.
"""

from keras import datasets
from keras.utils import np_utils

def get_datasets():
    return datasets.cifar10.load_data()

def neautralize_x(X_train, X_test):
    return X_train, X_test

def neautralize_y(y_train, y_test, num_classes):
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    return y_train, y_test

def load_train_and_test():
    (X_train, y_train), (X_test, y_test) = get_datasets()
    (X_train, X_test) = neautralize_x(X_train, X_test)
    (y_train, y_test) = neautralize_y(y_train, y_test, 10)
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    print('Utility file, not executable!')
