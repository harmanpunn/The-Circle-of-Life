from neuralflow.model import Model
import numpy as np

model = Model(3)
model.addLayer(5)
model.addLayer(1)

model.initLayers()

print(model.predict([[1,2,3],[4,5,6]]))

model.train_step([[1,2,3],[3,4,5]],np.array([[1],[2]]))