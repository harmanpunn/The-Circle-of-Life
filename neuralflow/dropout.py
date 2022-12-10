import numpy as np

class DropoutLayer:
    def __init__(self, input, p =0.2):
        self.mask =None
        self.input_shape = input
        self.p = p
        if self.p < 0 or self.p > 1:
            raise ValueError("p must be a probability")

    def forward_propagation(self,input_data):
        self.mask = np.random.choice([1,0], self.input_shape, p=[1-self.p,self.p]).reshape((1,self.input_shape))
        f = lambda k : np.multiply(k,self.mask)
        return f(input_data)
    
    def backward_propagation(self,output_error,learning_rate):
        f = lambda k : np.multiply(k,self.mask)
        return f(output_error)