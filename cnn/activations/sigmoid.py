import math


class Sigmoid():
    def __init__(self, inputs):
        self.res = []
        for input in inputs:
            self.res.append(1 / (1 + math.exp(-1 * input)))

    @property
    def result(self):
        return self.res