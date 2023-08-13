"""
This configuration file provides essential settings required to train and evaluate the neural network model
on the CIFAR-10 dataset. It comprises dimensions of the input data, data-related configurations like paths,
model-specific settings, training hyperparameters, and other external configurations such as loss types
and file-saving locations. Adjusting these configurations can influence the model's performance and training behavior.
"""

"""Dimentions"""
H = 32
W = 32
CHANNEL = 3

"""Data"""
NUMBERSOFATTRIBUTES  = 10

"""Model"""
NEURONS = [128, 256, 384]

"""Hyperparameters"""
SHAPE = (H, W, CHANNEL)
LEARNINGRATE = 1e-4
BATCHSIZE = 32
EPOCHS = 300
CSVPATH = "\\stats\\"
CSVEXTENSION = ".csv"

"""Externals"""
LOSS = "categorical_crossentropy" # or "binary_crossentropy"
MODELSAVELOCATION = "\\models\\"
MODELEXTENSTION = ".h5"

if __name__ == "__main__":
    print("Config file, not executable!")
