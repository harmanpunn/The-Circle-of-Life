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


class P3AgentQ(GraphEntity):

    someBigNumber = 200

    def __init__(self, graph: Graph, model = None,filePath = None) -> None:
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
        self.loss = 0.0

        self.databaseX = []
        self.databaseY = []
        self.fromEpoch = 0

        self.learning_rate = 0.001
        self.timeStamp = 0
        self.stTimeStamp = 0

        if model is None and filePath is None:
            raise RuntimeError("Wrong initialisation of Agent")
        
        if model is not None:
            self.uModel = model
        else:
            self.uModel = Model(-1).load(filePath).use()
    
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
            return 30
        # use model to process
        dt= self.getInputFromState(graph,tmpState)
        return self.uModel.predict(dt)[0][0][0]

        

    def plan(self, graph: Graph, info):
        # current position belief update
        self.belief = getNewBeliefs(self.belief,self.position,False)

        # Pick max belief node to survey
        max_val = max(self.belief)
        max_beliefs = [i for i, v in enumerate(self.belief) if v==max_val]
        survey_node = random.choice(max_beliefs)
        survey_res = graph.survey(survey_node)[2]
         # Updating Priors with fact that prey not at survey location
        self.belief = getNewBeliefs(self.belief,survey_node, survey_res)

        knows = [1,0] 
        if max(self.belief)==1.0:
            knows = [1,1]
        # # Transitioning prior probabilities        
        transitions   = transitionProbabilities(self.belief,graph.info)

        # new state will be (position, new Pred pos, transitioned probs)
        # we try to take the action now
        mnVal = float("inf")

        if self.training:
            # Current state prediction
            s = self.getInputFromState(graph.info, [self.position,info["predator"],self.belief])[0]
            
            # Current state target
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
                    
                    valOfAction += probOfStateTransition * self.getValueOfState(graph.info,tmpState)
                
                if valOfAction < mnVal:
                    mnVal = valOfAction
                    self.nextPosition = action
            # print('mnVal ',mnVal)
            exp = self.getValueOfState(graph.info,[self.position,info["predator"],self.belief])
            # print("Learning from ",mnVal," to ",exp, ": ",self.learning_rate)
            self.loss += self.uModel.loss(np.array([[mnVal]]),np.array([[exp]]))
            self.uModel.singleFit(np.array(s),np.array([[mnVal]]),learning_rate= self.learning_rate)
        else:
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
                    valOfAction += probOfStateTransition * self.getValueOfState(graph.info,tmpState)
                    
                if valOfAction < mnVal:
                    mnVal = valOfAction
                    self.nextPosition = action

        self.belief = transitions
        
        max_val = max(self.belief)
        max_beliefs = [i for i, v in enumerate(self.belief) if v==max_val]
        prey = random.choice(max_beliefs)
        Environment.getInstance().expected_prey = max_beliefs

        # updating learning rate
        # if self.learning_rate>0.001:
        #     # print("Kam karo")
        #     self.learning_rate -= 0.01 * (0.5 ** (self.stTimeStamp/10))
        # else:
        #     self.learning_rate = 0.001
        self.timeStamp +=1
        # self.stTimeStamp +=1
        return knows