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

    def set_input_size(self, shape):
        self.input_shape = shape
        self._size = int(self.input_shape[1] * self.input_shape[2] *
                         self.input_shape[3])

    def flattening(self, matrix):
        flattened = []

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                for k in range(len(matrix[i][j])):
                    flattened.append(matrix[i][j][k])

        self.neurons = flattened

        print("Flattening")
        print(np.array(matrix))
        print(np.array(flattened))
