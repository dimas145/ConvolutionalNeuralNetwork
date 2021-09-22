from .layers import (
    Dense,
    Conv2D,
    Flatten,
    Pooling,
)


class Sequential:
    def __init__(self, layers=None):
        self.layers = []
        if layers != None:
            for layer in layers:
                self.add(layer)

        self.state = {Dense: 0, Conv2D: 0, Flatten: 0, Pooling: 0}

    def add(self, layer):
        if (len(self.layers) != 0):
            layer.input_size = self.layers[-1].output_size

        self.state[type(layer)] += 1

        if type(layer) in [Dense, Conv2D, Flatten, Pooling]:
            layer.name += "_" + str(self.state[type(layer)])

        layer.init_layer()

        self.layers.append(layer)

    def forward_propagation(self, X):
        for k in range(len(self.layers)):
            if (type(self.layers[k]) == Dense):
                if (k == 0):
                    self.layers[k].forward_propagation([0] + X)
                else:
                    self.layers[k].forward_propagation(
                        [0] + self.layers[k - 1].input_neurons)
            elif (type(self.layers[k]) == Conv2D):
                if (k == 0):
                    self.layers[k].forward_propagation(X)
                else:
                    self.layers[k].forward_propagation(
                        self.layers[k - 1].input_neurons)
            elif (type(self.layers[k]) == Flatten):
                self.layers[k].flattening(
                    self.layers[k - 1].input_neurons)
            elif (type(self.layers[k]) == Pooling):
                self.layers[k].pooling(self.layers[k -
                                                   1].input_neurons)

    def summary(self):
        col1 = 35
        col2 = 35
        col3 = 17

        print()

        print("Model: Sequential")
        print("=" * 80)
        print("Layer (type)" + " " * 23 + "Output Shape" + " " * 23 +
              "Param #" + " " * 7)
        print("=" * 80)

        total_params = 0

        for i in range(len(self.layers)):
            layer = self.layers[i]

            if (type(layer) == Dense):
                before = self.layers[i].input_size
                param = (before + 1) * self.layers[i].output_size
            elif (type(layer) == Conv2D):
                param = self.layers[i].output_size[3] * (
                    self.layers[i].kernel_size[0] *
                    self.layers[i].kernel_size[1] *
                    self.layers[i].input_shape[3] + 1)
            else:
                param = 0

            col1_text = self.layers[i].name + " " + "(" + type(
                self.layers[i]).__name__ + ")"
            col2_text = str(self.layers[i].output_size)
            col3_text = str(param)

            print(col1_text + " " * (col1 - len(col1_text)) + col2_text + " " *
                  (col2 - len(col2_text)) + col3_text + " " *
                  (col3 - len(col3_text)))

            if (i != len(self.layers) - 1):
                print("-" * 80)
            else:
                print("=" * 80)

            total_params += param

        print("Total params: " + "{:,}".format(total_params))
        print()