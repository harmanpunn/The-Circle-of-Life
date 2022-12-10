from __future__ import annotations

from enum import Enum
import math
import numpy as np

class ActivationLayer:
    def __init__(self, activation : ActivationFunction):
        self.activation = ActivationFunction.getFunction(activation)
        self.activation_prime = ActivationFunction.getGradient(activation)

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class ActivationFunction(Enum):
    linear = 0
    relu = 1
    leakyrelu = 2
    tanh = 3
    sigmoid = 4
    
    def getFunction(activation : ActivationFunction):
        if activation == ActivationFunction.linear:
            return ActivationFunction.linearFunc
        elif activation == ActivationFunction.relu:
            return ActivationFunction.reluFunc
        elif activation == ActivationFunction.leakyrelu:
            return  ActivationFunction.leakyReluFunc
        elif activation == ActivationFunction.tanh:
            return ActivationFunction.tanhFunc
        elif activation == ActivationFunction.sigmoid:
            return ActivationFunction.sigmoidFunc
        else:
            raise ValueError("Invalid Activation Function")
    
    def getGradient(activation : ActivationFunction):
        if activation == ActivationFunction.linear:
            return ActivationFunction.linearGradient
        elif activation == ActivationFunction.relu:
            return np.vectorize(ActivationFunction.reluGradient)
        elif activation == ActivationFunction.leakyrelu:
            return  np.vectorize(ActivationFunction.leakyReluGradient)
        elif activation == ActivationFunction.tanh:
            return ActivationFunction.tanhGradient
        elif activation == ActivationFunction.sigmoid:
            return ActivationFunction.sigmoidGradient
        else:
            raise ValueError("Invalid Activation Function")

    def linearFunc(x):
        return x
    def linearGradient(x):
        return np.ones(x.shape)
    
    def reluFunc(x):
        return np.maximum(0.0,x)
    def reluGradient(x):
        if x>0:
            return 1
        elif x<0:
            return 0
        else:
            return 1e-7

    def leakyReluFunc(x):
        return np.maximum(0,x) + 1e-2*np.minimum(0,x)
    def leakyReluGradient(x):
        if x>0:
            return 1
        elif x<0:
            return 1e-2
        else:
            return 1e-7

    def tanhFunc(x):
        return np.tanh(x)
    def tanhGradient(x):
        return 1 - ActivationFunction.tanhFunc(x)**2
    
    def sigmoidFunc(x):
        return 1/(1+ np.exp(-x))
    def sigmoidGradient(x):
        return ActivationFunction.sigmoidFunc(x)*(1-ActivationFunction.sigmoidFunc(x))


    