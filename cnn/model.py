class Sequential:
    def __init__(self):
        self.layers = []
        self.state = { "Dense": 0, "Conv2D": 0, "Flatten": 0 }
    
    def add(self, layer):
        if(len(self.layers) != 0):
            layer.set_input_size(self.layers[-1].get_size())
        
        self.state[layer.get_type()] += 1
        
        if(layer.get_name() == "dense" or layer.get_name() == "conv2d" or layer.get_name() == "flatten"):
            layer.set_name(layer.get_name() + "_" + str(self.state[layer.get_type()]))
        
        layer.init_layer()
        
        self.layers.append(layer)
        
    def fit(self, X, y):
        for i in range(len(X)):
            for k in range(len(self.layers)):
                if(k == 0):
                    self.layers[k].forward_propagation([1] + X[i])
                else:
                    self.layers[k].forward_propagation(self.layers[k-1].get_input_neurons())
                    
    def summary(self):
        col1 = 35
        col2 = 35
        col3 = 17

        print()

        print("Model: Sequential")
        print("=" * 80)
        print("Layer (type)" + " " * 23 + "Output Shape" + " " * 23 + "Param #" + " " * 7)
        print("=" * 80)

        total_params = 0
        
        

        for i in range(len(self.layers)):
            
            layer = self.layers[i]
            
            if(layer.get_type() == "Dense"):
                before = self.layers[i].input_size
                param = (before + 1) * self.layers[i].size
            elif(layer.get_type() == "Conv2D"):
                param = self.layers[i].output_shape[3] * (self.layers[i].kernel_size[0] * self.layers[i].kernel_size[1] * self.layers[i].input_shape[2] + 1)
            else:
                param = 0

            col1_text = self.layers[i].name + " " + "(" + self.layers[i].type + ")"
            col2_text = str(self.layers[i].output_shape)
            col3_text = str(param)
            
            print(col1_text + " " * (col1 - len(col1_text)) + col2_text + " " * (col2 - len(col2_text)) + col3_text + " " * (col3 - len(col3_text)))
            
            if(i != len(self.layers) - 1):
                print("-" * 80)
            else:
                print("=" * 80)
            
            total_params += param
        
        print("Total params: " + str(total_params))
        print()