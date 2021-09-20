import numpy as np

class Flatten:
    def __init__(self, name="flatten"):
        self.type = "Flatten"
        self.name = name
        
        self.size = 0
        self.neurons = []
        
        self.input_shape = None
        
        self.output_shape = (None, self.size)
        
    def init_layer(self):
        self.output_shape = (None, self.input_shape[1] * self.input_shape[2] * self.input_shape[3])
        self.size = int(self.input_shape[1] * self.input_shape[2] * self.input_shape[3])
    
    def get_size(self):
        return self.size
    
    def get_type(self):
        return self.type
    
    def get_name(self):
        return self.name

    def get_input_neurons(self):
        return self.neurons
    
    def set_input_size(self, shape):
        self.input_shape = shape
        self.size = int(self.input_shape[1] * self.input_shape[2] * self.input_shape[3])
        
    def set_name(self, name):
        self.name = name
        
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
        