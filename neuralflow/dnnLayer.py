from neuralflow.activation import ActivationFunction
import numpy as np

class DNNLayer:
    def __init__(self, input, output, weights = None, bias = None, activation = ActivationFunction.sigmoid) -> None:
        self.activationFunction = ActivationFunction.getFunction(activation)
        self.activationGradient = ActivationFunction.getGradient(activation)
        
        self.input = input
        self.output = output

        self.lastLayer = False

        self.weights = np.random.rand(self.output, self.input) if weights == None else weights
        self.bias = np.random.rand(self.output,1) if bias == None else bias

        self.memory = {
            "aPrev" : None,
            "z": None
        }
        self.memoryFlag = False

    def forward(self, x1):
        x = np.array(x1)
        if x.shape[0]!=self.input:
            raise ValueError("Invalid input shape, expected "+str(self.input)+" got "+str(x.shape[0]))
    
        self.memory["aPrev"] = x
        # print("==========")
        # print("IN: ",self.memory["in"].shape)
        # print("Weights: ",self.weights.shape)
        try:
            out =  np.matmul(self.weights, x).transpose()
            out = out + self.bias.transpose()
            out = out.transpose()
        except ValueError:
            print(self.bias.shape)
            print("Weights: ",self.weights.shape)
            raise ValueError


        # A( W x X )
        self.memory["z"] = out
        # print("OUT: ",self.memory["out"].shape)
        self.memoryFlag = True

        return self.activationFunction(out)

    def backward(self, wNext= None, outGradNext=None, lossGrad = None, lastlayer = False):
        if (lastlayer and lossGrad is None):
            raise ValueError("Invalid Back prop last layer")
        if (not lastlayer and wNext is None and outGradNext is None):
            raise ValueError("Invalid back prop")
        if not self.memoryFlag:
            raise MemoryError("Run a proper forward pass before back prop")
        

        if not lastlayer:
            self.memory["biasGrad"] = np.multiply(np.matmul(wNext.transpose(),outGradNext),self.activationGradient(self.memory["z"]))
        else:
            self.memory["biasGrad"] = lossGrad
        self.memory["weightGrad"] = None
        for i in range(self.memory["biasGrad"].shape[1]):
            if self.memory["weightGrad"] is None:
                self.memory["weightGrad"] = np.matmul(self.memory["biasGrad"][:,i:i+1],self.memory["aPrev"].transpose()[i:i+1,:])
            else:
                self.memory["weightGrad"] = np.add(self.memory["weightGrad"],np.matmul(self.memory["biasGrad"][:,i:i+1],self.memory["aPrev"].transpose()[i:i+1,:]))
            
        self.memory["weightGrad"] /= self.memory["biasGrad"].shape[1]
        self.memory["biasGrad"] = np.reshape(np.mean(self.memory["biasGrad"],axis=1),(self.output,1))
        
    def updateWeights(self):
        if not "weightGrad" in self.memory:
            raise MemoryError("Run a backward pass first")
        self.weights = np.subtract(self.weights,0.01*self.memory["weightGrad"])
        self.bias = np.subtract(self.bias,0.01*self.memory["biasGrad"])