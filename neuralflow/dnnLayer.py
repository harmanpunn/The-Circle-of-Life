from neuralflow.activation import ActivationFunction
import numpy as np

class DNNLayer:
    def __init__(self, input, output, weights = None, bias = None, activation = ActivationFunction.sigmoid) -> None:
        self.activationFunction = ActivationFunction.getFunction(activation)
        self.activationGradient = ActivationFunction.getGradient(activation)
        
        self.input = input
        self.output = output

        self.lastLayer = False

        self.weights = np.random.rand(self.output, self.input+1) if weights == None else weights

        self.memory = {
            "in" : None,
            "out": None
        }
        self.memoryFlag = False

    def forward(self, x1):
        x = np.array(x1)
        if x.shape[0]!=self.input+1:
            raise ValueError("Invalid input shape, expected "+str(self.input+1)+" got "+str(x.shape[0]))
    
        self.memory["in"] = x
        # print("==========")
        # print("IN: ",self.memory["in"].shape)
        # print("Weights: ",self.weights.shape)
        out =  np.matmul(self.weights, x)
        # A( W x X )
        self.memory["out"] = self.activationFunction(out)
        # print("OUT: ",self.memory["out"].shape)
        self.memoryFlag = True

        return self.memory["out"]

    def backward(self, wNext= None, outGradNext=None, lossGrad = None, lastlayer = False):
        if (lastlayer and lossGrad is None):
            raise ValueError("Invalid Back prop last layer")
        if (not lastlayer and wNext is None and outGradNext is None):
            raise ValueError("Invalid back prop")
        if not self.memoryFlag:
            raise MemoryError("Run a proper forward pass before back prop")
        

        if not lastlayer:
            temp = np.append([np.ones(self.memory["out"].shape[1])],self.memory["out"],axis=0)
            self.memory["outGrad"] = np.matmul(wNext.transpose(),np.multiply(outGradNext,self.activationGradient(np.matmul(wNext,temp))))[1:]
        else:
            self.memory["outGrad"] = lossGrad
        
        self.memory["weightGrad"] = np.matmul(np.multiply(self.memory["outGrad"],self.activationGradient(np.matmul(self.weights,self.memory["in"]))),self.memory["in"].transpose())
        
    def updateWeights(self):
        if not "weightGrad" in self.memory:
            raise MemoryError("Run a backward pass first")
        # print("=====================")
        # print("Out    change: ",np.linalg.norm(self.memory["weightGrad"]))
        # print("Weight change: ",np.linalg.norm(self.memory["outGrad"]))
        self.weights = np.subtract(self.weights,0.0001*self.memory["weightGrad"])