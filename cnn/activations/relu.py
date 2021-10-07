class ReLU():
    def __init__(self, inputs):
        self._result = []
        self._prime_result = []

        for input in inputs:
            self._result.append(max(input, 0))
            self._prime_result.append(1 if input > 0 else 0)

    @property
    def result(self):
        return self._result

    @property
    def prime(self):
        return self._prime_result
