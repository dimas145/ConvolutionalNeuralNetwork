import cnn
from cnn import layers
from cnn import activations

if __name__ == "__main__":

    model = cnn.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Pooling(pool_mode="max", pool_size=(3, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Pooling(pool_mode="max", pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    print()

    modelz = cnn.Sequential()

    modelz.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    modelz.add(layers.Pooling(pool_mode="average"))
    modelz.add(layers.Conv2D(16, (3, 3), activation='relu'))
    modelz.add(layers.Pooling(pool_mode="average"))
    modelz.add(layers.Flatten())
    modelz.add(layers.Dense(120, activation='relu'))
    modelz.add(layers.Dense(84, activation='relu'))     
    modelz.add(layers.Dense(10))

    modelz.summary()
        