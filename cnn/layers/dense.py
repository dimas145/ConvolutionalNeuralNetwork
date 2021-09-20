import numpy as np
from cnn.activations import (
    ReLU,
    Sigmoid,
    Softmax,
)


class Matrix:
    def add(mat1, mat2):
        mat1 = np.array(mat1)
        mat2 = np.array(mat2)

        return np.add(mat1, mat2).tolist()

    def mult(mat1, mat2):
        mat1 = np.array(mat1)
        mat2 = np.array(mat2)

        return np.dot(mat1, mat2).tolist()


class Dense:
    def __init__(
        self,
        size,
        name="dense",
        activation=None,
        input_size=10,
    ):
        self._name = name
        self._size = size
        self.output_shape = (None, size)
        self.input_size = input_size

        if activation not in [ReLU, Sigmoid, Softmax]:
            raise Exception("Undefined activation")
        self._activation = activation

        self.neurons = [-1] * (self._size + 1)
        self._weights = []

    @property
    def activation(self):
        return self._activation

    @property
    def size(self):
        return self._size

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def get_input_neurons(self):
        return self.neurons

    def set_input_size(self, size):
        self.input_size = size

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def n_neurons(self):
        return len(self.neurons)

    def init_layer(self):
        self.init_weights()

    def init_weights(self):
        limit = np.sqrt(1 / float(self.input_size))
        self._weights = np.random.normal(0.0,
                                         limit,
                                         size=(self._size,
                                               self.input_size)).tolist()
        bias_weight = np.random.normal(0.0, limit)

        for i in range(len(self._weights)):
            self._weights[i].insert(0, bias_weight)

    def set_outputs_value_by_matrix(self, hk):
        self.neurons = hk

    def forward_propagation(self, input_neurons):
        input_neurons = list(map(lambda x: [x], input_neurons))

        print("Weight")
        print(self._weights)

        print("Input Neuron")
        print(input_neurons)

        ak = list(
            map(lambda x: x[0], Matrix.mult(self._weights, input_neurons)))
        hk = self._activation(ak).result

        self.set_outputs_value_by_matrix(hk)

        print("Dense Result")
        print(self.neurons)
