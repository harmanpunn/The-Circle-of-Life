from neuralflow.model import Model
import numpy as np

model = Model(3)
model.add(5)
model.add(1)

model.initLayers()

print(model.description)

print(model.predict([[1,2,3],[4,5,6]]))
model.save()
# model.train_step([[1,2,3],[3,4,5]],np.array([[1],[2]]))

model1 = Model(-1).load()

print(model.description)
print(model1.description)