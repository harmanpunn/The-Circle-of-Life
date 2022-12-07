import pickle
from graph import Graph
from environment import Environment
from valueIteration import getValues



environment = Environment(False,50)


for i in range(1,100):
    graph = Graph()
    values, probMatrix = getValues(graph)
    file = open('datadump/data-'+str(i),'wb')
    pickle.dump([graph.info, values], file)
