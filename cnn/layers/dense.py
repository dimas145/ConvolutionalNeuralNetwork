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

class Matrix:
    def add(mat1,mat2):
        mat1 = np.array(mat1)
        mat2 = np.array(mat2)
        
        return np.add(mat1, mat2).tolist()
        
    def mult(mat1, mat2):
        mat1 = np.array(mat1)
        mat2 = np.array(mat2)
        
        return np.dot(mat1, mat2).tolist()

class Dense:
    def __init__(self, size, name="dense", activation="relu", input_size=10):
        self.type = "Dense"
        self.name = name
        self.size = size
        self.output_shape = (None, size)
        self.input_size = input_size
        self.activation = activation

        self.activation_type = activation
        
        self.neurons = [-1] * (self.size + 1)
        self.weights = []
        
        
    def get_size(self):
        return self.size
    
    def get_type(self):
        return self.type
    
    def get_name(self):
        return self.name
    
    def get_input_neurons(self):
        return self.neurons
    
    def set_input_size(self, size):
        self.input_size = size
        
    def set_name(self, name):
        self.name = name
        
    def set_weights(self, weights):
        self.weights = weights
        
    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        
    def n_neurons(self):
        return len(self.neurons)
    
    def set_activation_type(self, act_type):
        self.activation_type = act_type
        
    def init_layer(self):
        self.init_weights()
    
    def init_weights(self):
        
        limit = np.sqrt(1 / float(self.input_size))
        self.weights = np.random.normal(0.0, limit, size=(self.size, self.input_size)).tolist()
        
        bias_weight = np.random.normal(0.0, limit)

        for i in range(len(self.weights)):
            self.weights[i].insert(0, bias_weight)
            
        
    def set_outputs_value_by_matrix(self, hk):
        self.neurons = hk

    def activation_function_wrapper(self, ak):
        if(self.activation_type == "linear"):
            return Activation.linear(ak)
        elif(self.activation_type == "sigmoid"):
            return Activation.sigmoid(ak)
        elif(self.activation_type == "softmax"):
            return Activation.softmax(ak)
        elif(self.activation_type == "relu"):
            return Activation.ReLU(ak)
    
    def forward_propagation(self, input_neurons):
        
        input_neurons = list(map(lambda x: [x], input_neurons))
        
        print("Weight")
        print(self.weights)
        
        print("Input Neuron")
        print(input_neurons)
        
        ak = list(map(lambda x: x[0], Matrix.mult(self.weights, input_neurons)))
        hk = self.activation_function_wrapper(ak)

        self.set_outputs_value_by_matrix(hk)
        
        print("Dense Result")
        print(self.neurons)