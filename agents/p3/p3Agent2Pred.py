from graphEntity import GraphEntity
from graph import Graph
from environment import Environment
from util import get_shortest_path
from valueIteration import getValues, getPolicyFromValues
import random
from util import getNewBeliefs, transitionProbabilities

from neuralflow.model import Model
import numpy as np
from time import sleep


class P3Agent2Pred(GraphEntity):

    someBigNumber = 200

    def __init__(self, graph: Graph, description = None,filePath = None,useV = False) -> None:
        self.node_count = Environment.getInstance().node_count
        self.type = 1
        while True:
            self.position = random.randint(0,Environment.getInstance().node_count-1)
            if not graph.node_states[self.position][0] and not graph.node_states[self.position][2]:
                break
        graph.allocate_pos(self.position, self.type)

        self.vals = None

        self.belief = [1.0/self.node_count]*self.node_count

        self.training = True

        self.databaseX = []
        self.databaseY = []
        self.fromEpoch = 0

        self.useV  = useV
        if not useV:
            if description is None and filePath is None:
                raise RuntimeError("Wrong initialisation of Agent")
            
            if description is not None:
                self.uModel = Model(4)
                self.uModel.description =description
                self.uModel.use().initLayers()
            else:
                self.uModel = Model(-1).load(filePath).use()
        else:
            self.uModel = Model(-1).load("./modelDump/VModel")
    
    def getInputFromState(self,graph,state):
        dt = [[0,0,0,0]]
        dt[0][0], path = get_shortest_path(graph,state[0],state[1],returnPath=True)
        # calculating Expected prey distance & parameters
        for i in range(0,len(state[2])):
            x = get_shortest_path(graph,state[0],i,find=state[1])
            y = 1 if i in path else 0
            # expected prey distance
            dt[0][1] += state[2][i]*x[0]
            # expected prey in pred path
            dt[0][2] += state[2][i]*y
            # expected pred in prey path
            dt[0][3] += state[2][i]*x[1]
        
        return np.array([dt])
    
    def store(self,graph,state,value):
        if self.training:
            self.databaseX.append(self.getInputFromState(graph,state))
            self.databaseY.append(np.array([[value]]))

    def getValueOfState(self,graph,tmpState):
        if tmpState[0]==tmpState[1]:
            # if terminal state
            return 9999
        # use model to process
        dt= self.getInputFromState(graph,tmpState)
        return self.uModel.predict(dt)[0][0][0]

        

    def plan(self, graph: Graph, info):
        # current state will be (position,info["pred"],self.beliefs)
        # print("Partial p3")
        #print("=====================")
        #print(" ==== Prob Sum 1 : ",str(sum(self.belief))," || ",max(self.belief))
        #print("Not at ",self.position)
        # current position belief update
        self.belief = getNewBeliefs(self.belief,self.position,False)

        # Pick max belief node to survey
        max_val = max(self.belief)
        max_beliefs = [i for i, v in enumerate(self.belief) if v==max_val]
        survey_node = random.choice(max_beliefs)
        survey_res = graph.survey(survey_node)[2]
       # print(" ==== Prob Sum 2 : ",str(sum(self.belief))," || ",max(self.belief))

         # Updating Priors with fact that prey not at survey location
        # print("Updating node : ",survey_node)
        # if not survey_res:
        #     print("Not at ",survey_node)
        # else:
        #     print("At ",survey_node)
        self.belief = getNewBeliefs(self.belief,survey_node, survey_res)

        knows = [1,0] 
        if max(self.belief)==1.0:
            knows = [1,1]
        # print(" ==== Prob Sum 3 : ",str(sum(self.belief))," || ",max(self.belief))
        # print("Transitioning")
        # # Transitioning prior probabilities        
        transitions  = transitionProbabilities(self.belief,graph.info)
        # print(" ==== Prob Sum 4 : ",str(sum(self.belief))," || ",max(self.belief))

        # new state will be (position, new Pred pos, transitioned probs)
        # we try to take the action now
        mnVal = float("inf")

        for action in graph.info[self.position] + [self.position]:
            predator_options = graph.info[info["predator"]]
            neigh_dist = []
            # calculating distances to new agent position
            for neigh in predator_options:
                neigh_dist.append(get_shortest_path(graph.info, neigh, action))

            # getting closest predator actions to agent location
            predator_close = [predator_options[i] for i in range(
                0, len(predator_options)) if neigh_dist[i] == min(neigh_dist)]

            valOfAction = 1
            for pred in predator_options:
                tmpState = (action, pred, transitions)
                probOfStateTransition = (0.6*(1/len(predator_close) if pred in predator_close else 0.0)+0.4/len(predator_options))  # prob of pred movement
                valOfTmpState = 0.0
                if self.training:
                    for p in range(len(transitions)):
                        if not self.useV:
                            valOfTmpState += transitions[p]* self.vals[(action,p,pred)]
                        else:
                            dt = [[]]
                            x = get_shortest_path(graph.info,action,p,find = pred)
                            y = get_shortest_path(graph.info,action,pred,find = p)
                            dt[0].append(y[0])
                            dt[0].append(x[0])
                            dt[0].append(y[1])
                            dt[0].append(x[1])
                            dt = np.array([dt])
                            predictedValue = self.uModel.predict(dt)[0][0][0]
                            valOfTmpState += transitions[p]*predictedValue
                else:
                    valOfTmpState = self.getValueOfState(graph.info,tmpState)
                valOfAction += probOfStateTransition * valOfTmpState
                self.store(graph.info,tmpState,valOfTmpState)

            if valOfAction < mnVal:
                mnVal = valOfAction
                self.nextPosition = action
        
        self.belief = transitions

        
        max_val = max(self.belief)
        max_beliefs = [i for i, v in enumerate(self.belief) if v==max_val]
        prey = random.choice(max_beliefs)
        Environment.getInstance().expected_prey = max_beliefs
        return knows