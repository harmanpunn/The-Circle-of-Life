from graphEntity import GraphEntity
from graph import Graph
from environment import Environment
from util import get_shortest_path
from valueIteration import getProbs
from neuralflow.model import Model
import numpy as np
import random

from time import sleep


class P3Agent1Pred(GraphEntity):

    someBigNumber = 200

    def __init__(self, graph: Graph) -> None:
        # super.__init__()
        self.type = 1
        while True:
            self.position = random.randint(0,Environment.getInstance().node_count-1)
            if not graph.node_states[self.position][0] and not graph.node_states[self.position][2]:
                break
        graph.allocate_pos(self.position, self.type)

        self.uModel = Model(-1).load("./modelDump/VModel").use()
        self.uModel.training = False
        self.values = None
        print("Initialised!")

    def getValues(self,graph, s_prime):
        dt = [[]]
        x = get_shortest_path(graph.info,s_prime[0],s_prime[1],find = s_prime[2])
        y = get_shortest_path(graph.info,s_prime[0],s_prime[2],find = s_prime[1])
        dt[0].append(y[0])
        dt[0].append(x[0])
        dt[0].append(y[1])
        dt[0].append(x[1])
        dt = np.array([dt])
        p = self.uModel.predict(dt)[0][0][0]
        print(dt," actual: ",self.values[s_prime],"; Predicted: ",p," : ",self.uModel.loss(self.values[s_prime],p))
        return p

    def plan(self, graph: Graph, info):
        state = (self.position, info["prey"],info["predator"])
        mnVal = float("inf")
        for action in graph.info[self.position]+[self.position]:
            # prey options
            prey_options = graph.info[state[1]] + [state[1]]
            # predator options
            predator_options = graph.info[state[2]]
            neigh_dist = []
            # calculating distances to new agent position
            for neigh in predator_options:
                neigh_dist.append(get_shortest_path(graph.info, neigh, action))

            # getting closest predator actions to agent location
            predator_close = [predator_options[i] for i in range(
                0, len(predator_options)) if neigh_dist[i] == min(neigh_dist)]

            valOfAction = 1
            for p in prey_options:
                for r in predator_options:
                    # tmpState = (action, p , r)
                    print("calculating probabilities")
                    probOfStateTransition = (1/len(prey_options))*(0.6*(1/len(predator_close)if r in predator_close else 0.0)+0.4/len(predator_options))
                    if action==r:
                        valOfAction += probOfStateTransition * 9999
                    elif action!=p:
                        valOfAction += probOfStateTransition * self.getValues(graph,(action, p , r))
            
            if valOfAction < mnVal:
                mnVal = valOfAction
                self.nextPosition = action
        return [1,1]