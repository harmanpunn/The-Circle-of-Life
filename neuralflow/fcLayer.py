from neuralflow.activation import ActivationFunction
import numpy as np
import math

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
        
        self.Gwm = np.zeros(self.weights.shape) 
        self.Gwv = np.zeros(self.weights.shape)

        self.Gbm = np.zeros(self.bias.shape) 
        self.Gbv = np.zeros(self.bias.shape)

        self.b1 = 0.8
        self.b2 = 0.99

        self.eps = 1e-8

        self.t = 0
    
    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate=0.001):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error
        if len(self.prevWgrads)==self.horizon:
            self.prevWgrads.pop(0)
            self.prevBgrads.pop(0)
        self.prevWgrads.append(weights_error)
        self.prevBgrads.append(output_error)

        # update parameters
        # if len(self.prevWgrads)!=self.horizon:
        # self.weights -= learning_rate * weights_error
        # self.bias -= learning_rate * output_error
        # else:
        # Adam

        
        self.Gwm = self.b1*self.Gwm + (1.0-self.b1)*weights_error 
        self.Gwv = self.b2*self.Gwv + (1.0-self.b2)*(weights_error**2)

        self.Gbm = self.b1*self.Gbm + (1.0-self.b1)*output_error
        self.Gbv = self.b2*self.Gbv + (1.0-self.b2)*(output_error**2)

        GwmH = self.Gwm /(1.0 - self.b1**(self.t+1))
        GwvH = self.Gwv /(1.0 - self.b2**(self.t+1))
        
        GbmH = self.Gbm /(1.0 - self.b1**(self.t+1))
        GbvH = self.Gbv /(1.0 - self.b2**(self.t+1))

        self.weights -= learning_rate * np.divide(GwmH,np.sqrt(GwvH)+self.eps)
        self.bias -= learning_rate * np.divide(GbmH,np.sqrt(GbvH)+self.eps)

        self.t +=1
            
        return input_error