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
# LOSS = "binary_crossentropy" categorical_crossentropy
LOSS = "categorical_crossentropy"
MODELSAVELOCATION = "\\models\\"
MODELEXTENSTION = ".h5"

if __name__ == "__main__":
    print("Config File")
