import cnn
from cnn import layers
from cnn import activations

import numpy as np

# for testing
from keras.datasets import mnist
import matplotlib.pyplot as plt


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
        [
            [4, 1, 3, 5, 3],
            [2, 1, 1, 2, 2],
            [5, 5, 1, 2, 3],
            [2, 2, 4, 3, 2],
            [5, 1, 3, 4, 5],
        ],
    ]
    w = [
        [
            [1, 2, 3],
            [4, 7, 5],
            [3, -32, 25],
        ],
        [
            [12, 18, 12],
            [18, -74, 45],
            [-92, 45, -18],
        ],
    ]

    conv2d = layers.Conv2D(2, (3, 3),
                           activation=activations.ReLU,
                           input_shape=(5, 5, 1))

    conv2d._weights = w

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


def test_mnist():
    img_rows, img_cols = 28, 28

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.array(X_train[:2])
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant').tolist()

    for i in range(2):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train[i], cmap='gray')
        plt.axis('off')

    print("Print Sample:")
    plt.show()

    lenet5 = cnn.Sequential()

    lenet5.add(
        layers.Conv2D(6, (5, 5),
                      activation=activations.ReLU,
                      input_shape=(32, 32, 1)))
    lenet5.add(layers.Pooling(pool_mode="average"))
    lenet5.add(layers.Conv2D(16, (3, 3), activation=activations.ReLU))
    lenet5.add(layers.Pooling(pool_mode="average"))
    lenet5.add(layers.Flatten())
    lenet5.add(layers.Dense(120, activation=activations.ReLU))
    lenet5.add(layers.Dense(84, activation=activations.ReLU))
    lenet5.add(layers.Dense(10, activation=activations.Softmax))

    lenet5.summary()

    X_train = list(map(lambda x: [x], X_train))

    lenet5.forward_propagation(X_train[0])
    print(np.array(lenet5.layers[-1].neurons))
    print()

    lenet5.forward_propagation(X_train[1])
    print(np.array(lenet5.layers[-1].neurons))
    print()

def propagation():
    x = [
        [
            [4, 1, 3, 5, 3],
            [2, 1, 1, 2, 2],
            [5, 5, 1, 2, 3],
            [2, 2, 4, 3, 2],
            [5, 1, 3, 4, 5],
        ],
    ]

    w = [
        [
            [1, 2, 3],
            [4, 7, 5],
            [3, -32, 25],
        ],
        [
            [12, 18, 12],
            [18, -74, 45],
            [-92, 45, -18],
        ],
    ]

    dw1 = [[0, 1, 2], [0, 3, -4]]

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

    model = cnn.Sequential()

    model.add(layers.Conv2D(2, (3, 3), activation=activations.ReLU, input_shape=(5, 5, 1)))
    model.add(layers.Pooling(pool_mode="max", pool_size=(3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation=activations.ReLU))
    model.add(layers.Dense(10, activation=activations.Softmax))

    model.summary()

    model.layers[0].set_weights(w)
    model.layers[3].set_weights(dw1)
    model.layers[4].set_weights(dw2)

    model.forward_propagation(x, 9)

    model.backward_propagation(9)

def propagation_random():

    X = np.random.rand(1, 32, 32).tolist()

    lenet5 = cnn.Sequential()

    lenet5.add(layers.Conv2D(6, (5, 5), activation=activations.ReLU,input_shape=(32, 32, 1)))
    lenet5.add(layers.Pooling(pool_mode="average"))
    lenet5.add(layers.Conv2D(16, (3, 3), activation=activations.ReLU))
    lenet5.add(layers.Pooling(pool_mode="average"))
    lenet5.add(layers.Flatten())
    lenet5.add(layers.Dense(120, activation=activations.ReLU))
    lenet5.add(layers.Dense(84, activation=activations.ReLU))
    lenet5.add(layers.Dense(10, activation=activations.Softmax))

    lenet5.summary()

    lenet5.forward_propagation(X, 9)

    lenet5.backward_propagation(9)

if __name__ == "__main__":
    # test()

    # test_lenet_5()

    # test_forward()

    # test_mnist()

    propagation_random()
