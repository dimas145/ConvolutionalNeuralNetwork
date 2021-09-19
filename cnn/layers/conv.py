import numpy as np

class Conv2D:
    def __init__(self, filters, kernel_size, name="conv2d", strides=(1, 1), padding=(0, 0), input_shape=None, activation="relu", pool_mode=None, pool_size=(2,2), pool_strides=None, pool_padding=(0, 0)):
        self.type = "Conv2D"
        self.name = name
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        
        self.pool_mode = pool_mode
        self.pool_strides = pool_strides
        self.pool_padding = pool_padding
        
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
        W = self.input_shape[0]
        F = self.kernel_size[0]
        P = self.padding[0]
        S = self.strides[0]
        K = self.filters
        V = round(((W - F + (2 * P))/S) + 1)
        
        self.output_shape = (None, V, V, K)