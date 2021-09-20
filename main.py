import cnn
from cnn import layers
from cnn import activations


def test():
    model = cnn.Sequential()

    model.add(
        layers.Conv2D(32, (3, 3),
                      activation=activations.ReLU,
                      input_shape=(32, 32, 3)))
    model.add(layers.Pooling(pool_mode="max", pool_size=(3, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation=activations.ReLU))
    model.add(layers.Pooling(pool_mode="max", pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activations.ReLU))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=activations.ReLU))
    model.add(layers.Dense(10, activation=activations.Softmax))

    model.summary()
    print()


def test_lenet_5():
    model = cnn.Sequential()

    model.add(
        layers.Conv2D(6, (5, 5),
                      activation=activations.ReLU,
                      input_shape=(32, 32, 1)))
    model.add(layers.Pooling(pool_mode="average"))
    model.add(layers.Conv2D(16, (3, 3), activation=activations.ReLU))
    model.add(layers.Pooling(pool_mode="average"))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation=activations.ReLU))
    model.add(layers.Dense(84, activation=activations.ReLU))
    model.add(layers.Dense(10, activation=activations.Softmax))

    model.summary()
    print()


def test_forward():
    x = [
        [[4, 1, 3, 5, 3], [2, 1, 1, 2, 2], [5, 5, 1, 2, 3], [2, 2, 4, 3, 2],
         [5, 1, 3, 4, 5]],
    ]

    w = [[[1, 2, 3], [4, 7, 5], [3, -32, 25]],
         [[12, 18, 12], [18, -74, 45], [-92, 45, -18]]]

    conv2d = layers.Conv2D(2, (3, 3),
                           activation=activations.ReLU,
                           input_shape=(5, 5, 1))

    conv2d.weights = w

    conv2d.forward_propagation(x)

    x = conv2d.neurons

    pooling = layers.Pooling(pool_mode="max",
                             pool_size=(3, 3),
                             pool_strides=(1, 1))

    pooling.pooling(x)

    flatten = layers.Flatten()

    flatten.flattening(pooling.neurons)

    dw1 = [[0, 1, 3], [0, 2, -4]]

    dense_1 = layers.Dense(2, activation=activations.ReLU)

    dense_1.weights = dw1

    dense_1.forward_propagation([0] + flatten.neurons)

    dense_2 = layers.Dense(10, activation=activations.Softmax)

    dw2 = [
        [0, 0.09, 0.02],
        [0, 0.08, 0.03],
        [0, 0.07, 0.03],
        [0, 0.06, 0.02],
        [0, 0.05, 0.01],
        [0, 0.04, 0.02],
        [0, 0.03, 0.07],
        [0, 0.04, 0.08],
        [0, 0.05, 0.05],
        [0, 0.01, 0.01],
    ]

    dense_2.weights = dw2

    dense_2.forward_propagation([0] + dense_1.neurons)


if __name__ == "__main__":
    test()

    test_lenet_5()

    test_forward()
