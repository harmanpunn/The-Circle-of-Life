from neuralflow.fcLayer import FCLayer
from neuralflow.activation import ActivationFunction, ActivationLayer
import numpy as np
from tqdm import tqdm
import math

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mseGrad(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def BinaryCrossEntropy(y_true, y_pred):
    y_truex = y_true.reshape(-1,1)
    y_predx = y_pred.reshape(-1,1)
    term_0 = (1-y_truex) * np.log(1-y_predx)
    term_1 = y_truex * np.log(y_predx)
    return -np.mean(term_0+term_1, axis=0)[0]
    
def BinaryCrossEntropyGrad(y_true, y_pred):
    term_0 = (1-y_true) * 1/(1-y_pred)
    term_1 = -1* y_true * 1/(y_pred)
    return term_0+term_1

class Model:
    def __init__(self,input):
        self.layers = []
        self.loss = None
        self.loss_prime = None

        self.description = []
        self.input = input

    # add layer to network
    def add(self, nodes, activation : ActivationFunction):
        if len(self.description)==0:
            self.description.append([self.input, nodes, activation])
        else:
            self.description.append([self.description[-1][1], nodes, activation])
    
    def initLayers(self):
        for d in self.description:
            self.layers.append(FCLayer(d[0],d[1]))
            self.layers.append(ActivationLayer(d[2]))

    def use(self,loss = "mse"):
        if loss=="mse":
            self.loss = mse
            self.loss_prime = mseGrad
        elif loss=="bce":
            self.loss = BinaryCrossEntropy
            self.loss_prime = BinaryCrossEntropyGrad

    # predict output for given input
    def predict(self, input_data):
        if len(self.layers)==0:
            raise MemoryError("Layers not initialised")

        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        lossHistory = []
        # training loop
        for i in range(epochs):
            err = 0
            for j in tqdm(range(samples)):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)
                if math.isnan(err):
                    raise ValueError("Issue with your parameters :)")

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            print(err)
            err /= samples
            print('epoch %d/%d   error=%.10f' % (i+1, epochs, err))
            lossHistory.append(err)

        return lossHistory