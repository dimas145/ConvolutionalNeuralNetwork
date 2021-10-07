import math


class Sigmoid():
    def __init__(self, inputs):
        self._result = []
        self._prime_result = []

        for input in inputs:
            s = 1 / (1 + math.exp(-1 * input))

            self._result.append(s)
            self._prime_result.append(s * (1 - s))

    @property
    def result(self):
        return self._result

    @property
    def prime(self):
        return self._prime_result
