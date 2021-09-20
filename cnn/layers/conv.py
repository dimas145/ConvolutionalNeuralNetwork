import numpy as np
from cnn import activations
from cnn.activations import (
    ReLU,
    Sigmoid,
    Softmax,
)


class Conv2D:
    def __init__(
            self,
            filters,
            kernel_size,
            name="conv2d",
            strides=(1, 1),
            padding=(0, 0),
            input_shape=None,
            activation=None,
    ):
        self._name = name

        if activation not in [ReLU, Sigmoid, Softmax]:
            raise Exception("Undefined activation")
        self._activation = activation

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.input_shape = input_shape

        self.output_shape = None

        self.neurons = []
        self.weights = []
        self.biases = []

    def init_layer(self):
        self.calculate_output_spatial_size()
        self.init_weights()

    def init_weights(self):
        # weight (kernel[0] * kernel[1] * filters)
        limit = np.sqrt(1 / float(
            self.input_shape[1] * self.input_shape[2] * self.input_shape[3]))

        for _ in range(self.filters):
            self.weights.append(
                np.random.normal(0.0,
                                 limit,
                                 size=(self.kernel_size[1],
                                       self.kernel_size[0])).tolist())


    def init_biases(self):
        for _ in range(self.filters):
            self.biases.append([1] * self.kernel_size[0])

    @property
    def size(self):
        return self.output_shape

    def get_input_neurons(self):
        return self.neurons

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def set_input_size(self, shape):
        self.input_shape = shape

    def set_outputs_value_by_matrix(self, matrix):
        self.neurons = matrix

    def set_weights(self, weights):
        self.weights = weights

    def calculate_output_spatial_size(self):
        if (self.input_shape[0] is not None):
            self.input_shape = (None, self.input_shape[0], self.input_shape[1],
                                self.input_shape[2])
        W = self.input_shape[1]
        F = self.kernel_size[0]
        P = self.padding[0]
        S = self.strides[0]
        K = self.filters
        V = round(((W - F + (2 * P)) / S) + 1)

        self.output_shape = (None, V, V, K)

    def add_auto_padding(self, matrix):


        height = len(matrix)
        width = len(matrix[0])

        left_padding = 0
        right_padding = 0

        up_padding = 0
        down_padding = 0

        if (width % self.strides[1] != 0):
            for i in range(self.strides[1] - (width % self.strides[1])):
                if (i % 2 == 0):
                    right_padding += 1
                else:
                    left_padding += 1

        if (height % self.strides[0] != 0):
            for i in range(self.strides[0] - (height % self.strides[0])):
                if (i % 2 == 0):
                    down_padding += 1
                else:
                    up_padding += 1

        for i in range(height):
            matrix[i] += [0] * right_padding
            for _ in range(left_padding):
                matrix[i].insert(0, 0)

        for _ in range(up_padding):
            matrix.insert(0, [0] * len(matrix[0]))

        for _ in range(down_padding):
            matrix.append([0] * len(matrix[0]))

        return matrix

    def add_padding(self, matrix):
        
        height = len(matrix)
        width = len(matrix[0])

        left_padding = self.padding[1]
        right_padding = self.padding[1]

        up_padding = self.padding[0]
        down_padding = self.padding[0]

        for i in range(height):
            matrix[i] += [0] * right_padding
            for _ in range(left_padding):
                matrix[i].insert(0, 0)

        for _ in range(up_padding):
            matrix.insert(0, [0] * len(matrix[0]))

        for _ in range(down_padding):
            matrix.append([0] * len(matrix[0]))

        return matrix

    def convolution(self, matrix):

        print("Before Convolution")
        print(len(matrix))
        print(len(matrix[0]))
        print(len(matrix[0]))

        matrix = self.add_auto_padding(matrix)
        matrix = self.add_padding(matrix)

        height = len(matrix)
        width = len(matrix[0])

        conv = []
        self.init_biases()

        for z in range(self.filters):
            temp2 = []
            for i in range(0, height - self.kernel_size[0] + 1,
                           self.strides[0]):
                temp1 = []
                for j in range(0, width - self.kernel_size[1] + 1,
                               self.strides[1]):
                    sum = 0
                    for k in range(self.kernel_size[0]):
                        for l in range(self.kernel_size[1]):
                            sum += matrix[i + k][j + l] * self.weights[z][k][l]
                        sum += self.biases[z][k]
                    temp1.append(sum)
                temp2.append(temp1)
            conv.append(temp2)
        print("Convo Res")
        print(len(conv))

        return conv

    def detector(self, matrix):
        print("Convo Res")
        print(len([self._activation(row).result for row in matrix]))
        return [self._activation(row).result for row in matrix]


    def forward_propagation(self, input_neurons):

        print("Before Convo")
        print(len(input_neurons))
        print(len(input_neurons[0]))
        print(len(input_neurons[0][0]))


        convoluted = []
        for i in range(len(input_neurons)):
            convoluted.append(self.convolution(input_neurons[i]))

        detected = []
        for item in convoluted:
            for j in range(len(item)):
                detected.append(self.detector(item[j]))

        self.set_outputs_value_by_matrix(detected)

        print(len(self.neurons))
        print(len(self.neurons[0]))
        print(len(self.neurons[0][0]))
        print("Convo")
        print(self.output_shape)
        print(len(self.neurons))