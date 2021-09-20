import numpy as np


class Flatten:
    def __init__(self, name="flatten"):
        self._name = name

        self._size = 0
        self.neurons = []

        self.input_shape = None

        self.output_shape = (None, self._size)

    def init_layer(self):
        self.output_shape = (None, self.input_shape[1] * self.input_shape[2] *
                             self.input_shape[3])
        self._size = int(self.input_shape[1] * self.input_shape[2] *
                         self.input_shape[3])

    @property
    def size(self):
        return self._size

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def get_input_neurons(self):
        return self.neurons

    def set_input_size(self, shape):
        self.input_shape = shape
        self._size = int(self.input_shape[1] * self.input_shape[2] *
                         self.input_shape[3])

    def flattening(self, matrix):

        print(len(matrix))
        print(len(matrix[0]))
        print(len(matrix[0][0]))


        flattened = []

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                for k in range(len(matrix[i][j])):
                    flattened.append(matrix[i][j][k])

        self.neurons = flattened

        print("Flatten")
        print(len(self.neurons))

