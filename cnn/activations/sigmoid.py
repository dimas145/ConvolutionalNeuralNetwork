import math


class Sigmoid():
    def __init__(self, inputs):
        self._result = []
        for input in inputs:
            self._result.append(1 / (1 + math.exp(-1 * input)))

    @property
    def result(self):
        return self._result