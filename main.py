import cnn
from cnn import layers
from cnn import activations

if __name__ == "__main__":
    # for testing
    model = cnn.Sequential()

    # example
    # model = cnn.Sequential(
    #     [
    #         layers.Dense(2, activation="relu", name="layer1"),
    #         layers.Dense(3, activation="relu", name="layer2"),
    #         layers.Dense(4, name="layer3"),
    #     ]
    # )