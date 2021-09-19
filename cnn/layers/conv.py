import numpy as np

class Conv2D:
    def __init__(self, filters, kernel_size, name="conv2d", strides=(1, 1), padding=(0, 0), input_shape=None, activation="relu"):
        self.type = "Conv2D"
        self.name = name
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        
        self.input_shape = input_shape
        
        self.output_shape = None
        
    def init_layer(self):
        self.calculate_output_spatial_size()
                 
    def get_size(self):
        return self.output_shape
    
    def get_type(self):
        return self.type
    
    def get_name(self):
        return self.name
    
    def set_input_size(self, shape):
        self.input_shape = shape
        
    def set_name(self, name):
        self.name = name
    
    def calculate_output_spatial_size(self):
        
        if(self.input_shape[0] is not None):
            self.input_shape = (None, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        W = self.input_shape[1]
        F = self.kernel_size[0]
        P = self.padding[0]
        S = self.strides[0]
        K = self.filters
        
        V = round(((W - F + (2 * P))/S) + 1)
        
        self.output_shape = (None, V, V, K)
    
        