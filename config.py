"""Dimentions"""
H = 32
W = 32

"""Data"""
NUMBERSOFATTRIBUTES  = 10

"""Model"""
# NEURONS = [64, 128, 256, 512, 1024]
# NEURONS = [32, 64, 128, 256, 512]
NEURONS = [8, 16, 32, 48, 64]

"""Hyperparameters"""
SHAPE = (32, 32, 1)
LEARNINGRATE = 1e-4
BATCHSIZE = 16
EPOCHS = 50
CSVPATH = "data.csv"

"""Externals"""
# LOSS = "binary_crossentropy" categorical_crossentropy
LOSS = "categorical_crossentropy"
MODELSAVELOCATION = "model.h5"

if __name__ == "__main__":
    for x in range(10):
        print(x)
