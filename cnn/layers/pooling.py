import numpy as np


class Pooling:
    def __init__(
            self,
            pool_mode,
            name="pooling",
            pool_size=(2, 2),
            pool_strides=None,
            pool_padding=(0, 0),
    ):
        self._name = name

        self.pool_mode = pool_mode
        self.pool_strides = pool_strides if (pool_strides
                                             is not None) else pool_size
        self.pool_padding = pool_padding
        self.pool_size = pool_size

        self.neurons = []
        self.input_shape = None
        self.output_shape = None

    @property
    def size(self):
        return self.output_shape

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

    def init_layer(self):
        height = self.input_shape[1]
        width = self.input_shape[2]

        if (width % self.pool_strides[1] != 0):
            width += self.pool_strides[1] - (width % self.pool_strides[1])

        if (height % self.pool_strides[0] != 0):
            height += self.pool_strides[0] - (height % self.pool_strides[0])

        self.output_shape = (None, width // self.pool_size[1],
                             height // self.pool_size[0], self.input_shape[3])

    def add_auto_padding(self, matrix):
        height = len(matrix)
        width = len(matrix[0])

        left_padding = 0
        right_padding = 0

        up_padding = 0
        down_padding = 0

        if (width % self.pool_strides[1] != 0):
            for i in range(self.pool_strides[1] -
                           (width % self.pool_strides[1])):
                if (i % 2 == 0):
                    right_padding += 1
                else:
                    left_padding += 1

        if (height % self.pool_strides[0] != 0):
            for i in range(self.pool_strides[0] -
                           (height % self.pool_strides[0])):
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

        left_padding = self.pool_padding[1]
        right_padding = self.pool_padding[1]

        up_padding = self.pool_padding[0]
        down_padding = self.pool_padding[0]

        for i in range(height):
            matrix[i] += [0] * right_padding
            for j in range(left_padding):
                matrix[i].insert(0, 0)

        for _ in range(up_padding):
            matrix.insert(0, [0] * len(matrix[0]))

        for _ in range(down_padding):
            matrix.append([0] * len(matrix[0]))

        return matrix

    def max_pooling(self, matrix):
        matrix = self.add_auto_padding(matrix)
        matrix = self.add_padding(matrix)

        height = len(matrix)
        width = len(matrix[0])

        pooled = []

        for i in range(0, height - self.pool_size[0] + 1,
                       self.pool_strides[0]):
            temp1 = []
            for j in range(0, width - self.pool_size[1] + 1,
                           self.pool_strides[1]):
                max = matrix[i][j]
                for k in range(self.pool_size[0]):
                    for l in range(self.pool_size[1]):
                        if (matrix[i + k][j + l] > max):
                            max = matrix[i + k][j + l]
                temp1.append(max)
            pooled.append(temp1)

        return pooled

    def average_pooling(self, matrix):

        matrix = self.add_auto_padding(matrix)
        matrix = self.add_padding(matrix)

        height = len(matrix)
        width = len(matrix[0])

        pooled = []

        for i in range(0, height - self.pool_size[0] + 1,
                       self.pool_strides[0]):
            temp1 = []
            for j in range(0, width - self.pool_size[1] + 1,
                           self.pool_strides[1]):
                sum = 0
                for k in range(self.pool_size[0]):
                    for l in range(self.pool_size[1]):
                        sum += matrix[i + k][j + l]
                temp1.append(sum / (self.pool_size[0] * self.pool_size[1]))
            pooled.append(temp1)

        return pooled

    def pooling(self, matrix):

        if (self.pool_mode == "max"):
            res = [self.max_pooling(matrix[i]) for i in range(len(matrix))]
        elif self.pool_mode == "average":
            res = [self.average_pooling(matrix[i]) for i in range(len(matrix))]
        else:
            raise Exception("Undefined pooling mode!")

        self.neurons = res