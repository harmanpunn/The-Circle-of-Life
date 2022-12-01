from activation import ActivationFunction
import numpy as np

class DNNLayer:
    def __init__(self, input, output, batchSize  = 1, weights = None, bias = None, activation = ActivationFunction.relu) -> None:
        self.activationFunction = ActivationFunction.getFunction(activation)
        self.activationGradient = ActivationFunction.getGradient(activation)
        
        self.input = input
        self.output = output
        self.batchSize = batchSize

        self.lastLayer = False

        self.weights = np.random.rand(self.output, self.input+1) if weights == None else weights

        self.memory = {
            "in" : None,
            "out": None
        }
        self.memoryFlag = False

    def forward(self, x1):
        x = np.array(x1)
        
        if x.shape[1]!=self.input:
            raise ValueError("Invalid input shape, expected "+str(self.input)+" got "+str(x.shape[1]))
        
        x = x.transpose()
        x = np.append([np.ones(x.shape[1])],x,axis=0)
        
        self.memory["in"] = x
        out =  np.matmul(self.weights, x)
        self.memory["out"] = out
        self.memoryFlag = True

        return self.activationFunction(self.memory["out"]).transpose()

    def backward(self, x):
        if not self.memoryFlag:
            raise MemoryError("Run a proper forward pass before back prop")
        
        print(self.memory["in"].shape)
        print(self.memory["out"].shape)
        gradient = np.matmul(self.activationGradient(self.memory["out"]),self.memory["in"].transpose())
        print(gradient)
        print(gradient.shape)
        print(self.weights.shape)
        self.memoryFlag = False

        # calculate gradient
        # if self.lastLayer:

            

    
layer = DNNLayer(2,10,2)

print(layer.forward([[1,2],[2,3]]))
print(layer.backward([1,2,3]))