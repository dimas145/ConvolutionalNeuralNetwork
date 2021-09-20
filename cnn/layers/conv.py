import math
import numpy as np

class Activation:
        
    def linear(listx):
        res = []
        for i in range(len(listx)):
            res.append(listx[i])
        return res

    def sigmoid(listx):
        res = []
        for i in range(len(listx)):
            res.append([1/(1+math.exp(-1*listx[i]))])
        return res

    def softmax(listx):
        sum = 0
        listx = list(map(lambda x: math.exp(x), listx))
        for i in range(len(listx)):
            sum = sum + listx[i]
        res = []
        for i in range(len(listx)):
            res.append(listx[i]/sum)
        return res

    def ReLU(listx):
        res = []
        for i in range(len(listx)):
            res.append(max(0, listx[i]))
        return res

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
        
        self.neurons = []
        self.weights = []

    def init_weights(self):
        
        # weight (kernel[0] * kernel[1] * filters)
        
        limit = np.sqrt(1 / float(self.input_shape[1] * self.input_shape[2] * self.input_shape[3]))
        
        for i in range(self.filters):
            self.weights.append(np.random.normal(0.0, limit, size=(self.kernel_size[1], self.kernel_size[0])).tolist())
        
    def init_layer(self):
        self.calculate_output_spatial_size()
        self.init_weights()
                 
    def get_size(self):
        return self.output_shape
    
    def get_type(self):
        return self.type
    
    def get_name(self):
        return self.name

    def get_input_neurons(self):
        return self.neurons
    
    def set_input_size(self, shape):
        self.input_shape = shape
        
    def set_name(self, name):
        self.name = name
        
    def set_outputs_value_by_matrix(self, matrix):
        self.neurons = matrix
        
    def set_weights(self, weights):
        self.weights = weights
    
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
        
        
    def add_auto_padding(self, matrix):
        height = len(matrix)
        width = len(matrix[0])     
        
        left_padding = 0
        right_padding = 0
        
        up_padding = 0
        down_padding = 0
        
        if(width % self.strides[1] != 0):
            for i in range(self.strides[1] - (width % self.strides[1])):
                if(i % 2 == 0):
                    right_padding += 1
                else:
                    left_padding += 1
                    
        if(height % self.strides[0] != 0):
            for i in range(self.strides[0] - (height % self.strides[0])):
                if(i % 2 == 0):
                    down_padding += 1
                else:
                    up_padding += 1
        
        for i in range(height):
            matrix[i] += [0] * right_padding
            for j in range(left_padding):
                matrix[i].insert(0, 0)

        for i in range(up_padding):
            matrix.insert(0, [0] * len(matrix[0]))
        
        for i in range(down_padding):
            matrix.append([0] * len(matrix[0]))
        
        return matrix
            
    def add_padding(self, matrix):
        height = len(matrix)
        width = len(matrix[0])     
        
        left_padding = self.padding[1]
        right_padding = self.padding[1]
        
        up_padding = self.padding[0]
        down_padding = self.padding[0]

        for i in range(height):
            matrix[i] += [0] * right_padding
            for j in range(left_padding):
                matrix[i].insert(0, 0)

        for i in range(up_padding):
            matrix.insert(0, [0] * len(matrix[0]))
        
        for i in range(down_padding):
            matrix.append([0] * len(matrix[0]))
            
        return matrix
    
    def activation_function_wrapper(self, listx):
        if(self.activation == "sigmoid"):
            return Activation.sigmoid(listx)
        elif(self.activation == "relu"):
            return Activation.ReLU(listx)
    
    def convolution(self, matrix):
        matrix = self.add_auto_padding(matrix)
        matrix = self.add_padding(matrix)
        
        height = len(matrix)
        width = len(matrix[0])
        
        conv = []
        
        for z in range(self.filters):
            temp2 = []
            for i in range(0, height - self.kernel_size[0] + 1, self.strides[0]):
                temp1 = []
                for j in range(0, width - self.kernel_size[1] + 1, self.strides[1]):
                    sum = 0
                    for k in range(self.kernel_size[0]):
                        for l in range(self.kernel_size[1]):
                            sum += matrix[i+k][j+l] * self.weights[z][k][l]
                    temp1.append(sum)
                temp2.append(temp1)
            conv.append(temp2)
            
        
        return conv
    
    def detector(self, matrix):
        
        detected = []
        for i in range(len(matrix)):
            detected.append(self.activation_function_wrapper(matrix[i]))
            
        return detected
    
    def forward_propagation(self, input_neurons):
        
        convoluted = []
        for i in range(len(input_neurons)):

            convoluted.append(self.convolution(input_neurons[i]))
            
        detected = []
        for i in range(len(convoluted)):
            for j in range(len(convoluted[i])):
                detected.append(self.detector(convoluted[i][j]))
        
        print("Detected")
        print(np.array(detected))

        self.set_outputs_value_by_matrix(detected)
        