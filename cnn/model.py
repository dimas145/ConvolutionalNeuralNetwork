import sys
import time
import math
import numpy as np

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
        self.loss = sys.maxsize

    def add(self, layer):
        if (len(self.layers) != 0):
            layer.input_size = self.layers[-1].output_size

        self.state[type(layer)] += 1

        if type(layer) in [Dense, Conv2D, Flatten, Pooling]:
            layer.name += "_" + str(self.state[type(layer)])

        layer.init_layer()

        self.layers.append(layer)

    def forward_propagation(self, X, y=0):
        for k in range(len(self.layers)):
            if (type(self.layers[k]) == Dense):
                if (k == 0):
                    self.layers[k].forward_propagation([0] + X)
                else:
                    self.layers[k].forward_propagation([0] + self.layers[k - 1].neurons)
            elif (type(self.layers[k]) == Conv2D):
                if (k == 0):
                    self.layers[k].forward_propagation(X)
                else:
                    self.layers[k].forward_propagation(self.layers[k - 1].neurons)
            elif (type(self.layers[k]) == Flatten):
                self.layers[k].flattening(self.layers[k - 1].neurons)
            elif (type(self.layers[k]) == Pooling):
                self.layers[k].pooling(self.layers[k - 1].neurons)
        self.loss = math.log(self.layers[-1].neurons[y])
        print("LOSS")
        print(self.loss)

    def backward_propagation(self):
        pass

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
            if (type(layer) == Dense or type(layer) == Flatten):
                col2_text = str((None, self.layers[i].output_size))
            else:
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

    def fit(self, X_train, y_train, batch_size=128, epochs=15):

        if(len(X_train) < batch_size):
            batch_size = len(X_train)
        
        batch = len(X_train) // batch_size

        for i in range(epochs):
            sys.stdout.write("Epoch " + str(i+1) + "/" + str(epochs) + '\n')
            sys.stdout.flush()
            for j in range(batch):

                if(j < batch - 1):
                    loading = ("[" + "=" * int(((j+1) / batch ) * 30)) + "> "
                    sys.stdout.write("[" + str(j+1) + "/" + str(batch) + "] " + loading + '\r')
                else:
                    loading = ("[" + "=" * int(((j+1) / batch ) * 30)) + "] "
                    sys.stdout.write("[" + str(j+1) + "/" + str(batch) + "] " + loading + '\n')
                sys.stdout.flush()
                time.sleep(0.1)
    
    def weights_summary(self):
        for i in range(len(self.layers)):
            if(type(self.layers[i]) == Dense or type(self.layers[i]) == Conv2D):
                w = np.array(self.layers[i]._weights)
                print(self.layers[i]._name + " " + str(w.shape))
                print(w)
            
        
