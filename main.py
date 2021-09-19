import cnn
from cnn import layers
from cnn import activations

if __name__ == "__main__":

    model = cnn.Sequential()

    model.add(layers.Dense(3, input_size=3))
    model.add(layers.Dense(2))
    model.add(layers.Dense(2))

    model.summary()

    model.fit([[1, 2, 3]], [1])
    