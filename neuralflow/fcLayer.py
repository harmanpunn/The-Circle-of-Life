from neuralflow.activation import ActivationFunction
import numpy as np

class FCLayer:
    def __init__(self, input_size, output_size, weights = None, bias = None):
        
        self.prevWgrads = []
        self.prevBgrads = []
        self.horizon = 10

        if (not weights is None) and (not bias is None):
            self.weights = weights
            self.bias = bias
        else:
            self.weights = np.random.rand(input_size, output_size) - 0.5
            self.bias = np.random.rand(1, output_size) - 0.5
    
    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error
        if len(self.prevWgrads)==self.horizon:
            self.prevWgrads.pop(0)
            self.prevBgrads.pop(0)
        self.prevWgrads.append(weights_error)
        self.prevBgrads.append(output_error)

        # update parameters
        if len(self.prevWgrads)!=self.horizon:
            self.weights -= learning_rate * weights_error
            self.bias -= learning_rate * output_error
        else:
            # RMSProp
            Gw = self.prevWgrads[0]**2
            Gb = self.prevBgrads[0]**2
            for i in range(1,self.horizon):
                Gw = np.add(self.prevWgrads[i]**2,Gw)
                Gb = np.add(self.prevBgrads[i]**2,Gb)
            self.weights -= learning_rate * np.divide(weights_error,Gw,out=np.zeros_like(weights_error), where=Gw!=0)
            self.bias -= learning_rate * np.divide(output_error,Gb,out=np.zeros_like(output_error), where=Gb!=0)
            
        return input_error