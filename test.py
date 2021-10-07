import cnn
from cnn import layers
from cnn import activations
import numpy as np
import matplotlib.pyplot as plt

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

# print('Conv2d 1')
# print(conv2d._neurons)

x = conv2d._neurons
pooling = layers.Pooling(pool_mode="max",
                         pool_size=(3, 3),
                         pool_strides=(1, 1))

pooling.pooling(x)

# print('Pooling 1')
# print(pooling._neurons)

flatten = layers.Flatten()

flatten.flattening(pooling._neurons)

# print('Flatten')
# print(flatten._neurons)

dw1 = [[0, 1, 3], [0, 2, -4]]

dense_1 = layers.Dense(2, activation=activations.ReLU)

dense_1.weights = dw1

dense_1.forward_propagation([0] + flatten._neurons)

# print('Dense 1')
# print(np.array(dense_1._neurons))

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

# print('Dense 2')
dense_2.forward_propagation([0] + dense_1._neurons, 9)


print(dense_2._neurons)

de_do = -1 / np.array(dense_2._neurons)
# print("============")
# print(de_do)
# print("============")

dense_2.backward_propagation(de_do, 1, 9)
dense_1.backward_propagation(dense_2._dE_do, 0, 9)
# dense_1.backward_propagation(dense_1._de_do, 0, 9)
