import cnn
from cnn import layers
from cnn import activations

if __name__ == "__main__":

    model = cnn.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(3))
    model.add(layers.Dense(2))
    model.add(layers.Dense(2))

    model.summary()

    # model.fit([[1, 2, 3]], [1])
        