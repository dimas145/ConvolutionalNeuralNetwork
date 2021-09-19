class ReLU():
    def __init__(self, inputs):
        self.res = []
        for input in inputs:
            self.res.append(max(input, 0))

    @property
    def result(self):
        return self.res
