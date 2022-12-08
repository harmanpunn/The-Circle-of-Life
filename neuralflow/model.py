from neuralflow.fcLayer import FCLayer
from neuralflow.activation import ActivationFunction, ActivationLayer
from neuralflow.dropout import DropoutLayer
import numpy as np
from tqdm import tqdm
import math
import os
import pickle

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mseGrad(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def BinaryCrossEntropy(y_true, y_pred):
    y_true = y_true.reshape(y_true.shape[0],y_true.shape[1]) + 1e-8
    y_pred = y_pred.reshape(y_true.shape[0],y_true.shape[1]) + 1e-8
    # print(y_truex,y_predx)
    return np.average(- (y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)),axis=1).mean()
    
def BinaryCrossEntropyGrad(y_true, y_pred):
    return -1* y_true * 1/(y_pred + 1e-8) + (1-y_true) * 1/(1-y_pred + 1e-8)

class Model:
    def __init__(self,input):
        self.layers = []
        self.loss = None
        self.loss_prime = None

        self.description = {}
        self.input = input

        self.training =  True

        self.min_loss = float("inf")

    # add layer to network
    def add(self, nodes, activation : ActivationFunction = ActivationFunction.sigmoid, dropout=0.0):
        if len(self.description.keys())==0:
            self.description[0] = [self.input, nodes, activation,dropout]
        else:
            self.description[len(self.description.keys())] = [self.description[len(self.description.keys())-1][1], nodes, activation,dropout]
        return self
    
    def initLayers(self):
        for d in self.description:
            val = self.description[d]
            self.layers.append(FCLayer(val[0],val[1]))
            self.layers.append(ActivationLayer(val[2]))
            # Output does not need dropout
            # if d!=len(self.description)-1:
            #     self.layers.append(DropoutLayer(val[1],val[3]))
        return self

    def use(self,loss = "mse"):
        if loss=="mse":
            self.loss = mse
            self.loss_prime = mseGrad
        elif loss=="bce":
            self.loss = BinaryCrossEntropy
            self.loss_prime = BinaryCrossEntropyGrad
        return self

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
                if (not self.training) and isinstance(layer, DropoutLayer):
                    # print("Skipping Dropout")
                    continue
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs=1, quiet= False, learning_rate=0.001,validation_data = None,save=False,filePath = None,fromEpoch = 0):
        # sample dimension first
        samples = len(x_train)

        lossHistory = []
        epochBar = range(epochs) if not quiet else tqdm(range(epochs))
        # training loop
        for i in epochBar:
            err = 0
            bar = range(samples) if quiet else tqdm(range(samples))
            for j in bar:
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)
                # dat = {
                #     "training_loss":err/(j+1)
                # }
                # if not validation_data is None:
                #     dat["val_loss"] = self.loss(validation_data[0],np.array(self.predict(validation_data[1])))

                # bar.set_postfix(dat)
                if math.isnan(err):
                    raise ValueError("Issue with your parameters :)")

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            # print(err)
            err /= samples
            if not validation_data is None:
                valErr = self.loss(validation_data[0],np.array(self.predict(validation_data[1])))
                if not quiet:
                    print('epoch %d/%d  || training_error=%.10f ; val_error=%.10f' % (i+1, epochs+fromEpoch, err,valErr))
            else:
                if not quiet:
                    print('epoch %d/%d   error=%.10f' % (fromEpoch + i+1, epochs+fromEpoch, err))
            if save and err<self.min_loss:
                self.min_loss = err
                if not quiet:
                    print("Saving model to %s"%(filePath+str(fromEpoch + i)))
                self.save(filePath+str(fromEpoch + i))

            lossHistory.append(err)
            for layer in self.layers:
                if isinstance(layer,FCLayer):
                    layer.t = 0

        return lossHistory

    def save(self,path="./checkpoint"):
        tmp = {"desc":self.description,"weights":[],"bias":[]}
        for d in self.description:
            tmp["weights"].append(self.layers[2*d].weights)
            tmp["bias"].append(self.layers[2*d].bias)
        
        pickle.dump(tmp,open(path,"wb"))
    
    def load(self,path="./checkpoint"):
        if not os.path.exists(path):
            raise ValueError("Path does not exist")
        
        dump = pickle.load(open(path,"rb")) 
        self.description = dump["desc"]
        self.layers = []
        for d in self.description:
            val = self.description[d]
            self.layers.append(FCLayer(val[0],val[1], weights=dump["weights"][d],bias=dump["bias"][d]))
            self.layers.append(ActivationLayer(val[2]))
            # if d!=len(self.description)-1:
            #     self.layers.append(DropoutLayer(val[1],val[3]))
        
        return self