from neuralflow.dnnLayer import DNNLayer
from neuralflow.activation import ActivationFunction
import numpy as np
from tqdm import tqdm

def mse(out,Y):
    return np.square(np.linalg.norm(np.subtract(out,Y),axis=0))/Y.shape[0]
def mseGrad(out,Y):
    return 2*np.subtract(out,Y)/Y.shape[0]

class Model:
    def __init__(self,input, loss = "mse") -> None:
        self.layers : list[DNNLayer] = []
        self.input = int(input)
        self.description = []

        self.loss = mse
        self.lossGrad = mseGrad

        self.initialised = False

    def addLayer(self,output,activation= ActivationFunction.sigmoid):
        if len(self.description)==0:
            self.description.append([self.input,output,activation])
        else:
            self.description.append([self.description[-1][1],output,activation])
    
    def initLayers(self):
        self.initialised = True
        for d in self.description:
            self.layers.append(DNNLayer(d[0],d[1],activation=d[2]))

    def train_step(self,X, Y):
        # forward
        out = self.predict(X)
        
        # backprop step
        for i, layer in reversed(list(enumerate(self.layers))):
            if i==len(self.layers)-1:
                layer.backward(lossGrad=self.lossGrad(out,Y).transpose(),lastlayer=True)
                continue
            layer.backward(wNext = self.layers[i+1].weights, outGradNext= self.layers[i+1].memory["outGrad"])
        # update weights
        for layer in self.layers:
            layer.updateWeights()

        #return loss
        return self.loss(out,Y)

    def train(self,X,Y,batch_size=8,epochs = 1):
        samplesCount = X.shape[0]
        batchCount = samplesCount // batch_size 
        if samplesCount%batch_size!=0:
            batchCount +=1
        print("Batch Count : ", batchCount)
        lossHistory = []
        for e in range(0,epochs):
            print("Epoch ",e+1)
            loss = 0.0
            for b in tqdm(range(0,batchCount)):
                if b!=batchCount-1:
                    batchInput = X[b*batch_size:b*batch_size + batch_size ]
                    batchOutput = Y[b*batch_size:b*batch_size + batch_size ]
                else:
                    batchInput = X[b*batch_size:]
                    batchOutput = Y[b*batch_size:]

                loss += np.linalg.norm(self.train_step(batchInput,batchOutput))
            loss /= batch_size
            print("Loss after epoch ",loss)
            lossHistory.append(loss)
        return lossHistory

    def predict(self, x):
        if not self.initialised:
            raise RuntimeError("Layers not initialised")
        
        x1 = np.array(x)
        if len(x1.shape)!=2:
            raise ValueError("Model only accepts array of inputs")
        if x1.shape[1]!=self.input:
            raise ValueError("Invalid input shape. "+str(x1.shape[1])+" != "+str(self.input))
        
        x1 = x1.transpose()
        for layer in self.layers:    
            x1 = np.append([np.ones(x1.shape[1])],x1,axis=0)
            x1 = layer.forward(x1)            
        
        return x1.transpose()