import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

def getDataset():
    return datasets.cifar10.load_data()

# def xNeutralizer(X_data):
#     Neutralizer = lambda img : np.mean(img, axis=2)/255
#     X_update = np.zeros(shape=(len(X_data), 32, 32))
#     for i in range(len(X_data)):
#         X_update[i] = Neutralizer(X_data[i])
#     return X_update

def neautralize_x(X_train, X_test):
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    #channel mean and std to do normalization
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    return X_train, X_test

# def yNeutralizer(y_data):
#     y_update = np.zeros(shape=(len(y_data), 10))
#     for i in range(len(y_data)):
#         y_update[i][y_data[i]] = 1
#     return y_update

def neautralize_y(y_train, y_test, num_classes):
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    return y_train, y_test

# def getTrainningDataset():
#     (_X_train, _y_train), (_, _) = getDataset()
#     X_train = xNeutralizer(_X_train)
#     y_train = yNeutralizer(_y_train)
#     return X_train, y_train


# def getTestingDataset():
#     (_, _), (_X_test, _y_test) = getDataset()
#     X_test = xNeutralizer(_X_test)
#     y_test = yNeutralizer(_y_test)
#     return X_test, y_test

# def load_data():
#     (X_train, y_train) = getTrainningDataset()
#     (X_test, y_test) = getTestingDataset()

#     X_train, X_valid = train_test_split(X_train, test_size=0.2, random_state=42)
#     y_train, y_valid = train_test_split(y_train, test_size=0.2, random_state=42)

#     return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

# def load_train_and_test():
#     (X_train, y_train) = getTrainningDataset()
#     (X_test, y_test) = getTestingDataset()
#     return (X_train, y_train), (X_test, y_test)

def load_train_and_test():
    (X_train, y_train), (X_test, y_test) = getDataset()
    (X_train, X_test) = neautralize_x(X_train, X_test)
    (y_train, y_test) = neautralize_y(y_train, y_test, 10)
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data()
    print(f"Dataset: Train: {len(X_train)} - Valid: {len(X_valid)} - Test: {len(X_test)}")
    # print(y_test)