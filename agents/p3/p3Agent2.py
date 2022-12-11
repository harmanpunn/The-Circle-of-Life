from graphEntity import GraphEntity
from graph import Graph
from environment import Environment
from util import get_shortest_path
from valueIteration import getValues, getPolicyFromValues
import random
from util import getNewBeliefs, transitionProbabilities

from time import sleep


class P3Agent2(GraphEntity):

    someBigNumber = 200

    def __init__(self, graph: Graph) -> None:
        self.node_count = Environment.getInstance().node_count
        self.type = 1
        while True:
            self.position = random.randint(0,Environment.getInstance().node_count-1)
            if not graph.node_states[self.position][0] and not graph.node_states[self.position][2]:
                break
        graph.allocate_pos(self.position, self.type)

        self.vals = None



        self.belief = [1.0/self.node_count]*self.node_count


    def plan(self, graph: Graph, info):
        # current state will be (position,info["pred"],self.beliefs)
        # print("Partial p3")
        print("=====================")
        print(" ==== Prob Sum 1 : ",str(sum(self.belief))," || ",max(self.belief))
        print("Not at ",self.position)
        # current position belief update
        self.belief = getNewBeliefs(self.belief,self.position,False)

        # Pick max belief node to survey
        max_val = max(self.belief)
        max_beliefs = [i for i, v in enumerate(self.belief) if v==max_val]
        survey_node = random.choice(max_beliefs)
        survey_res = graph.survey(survey_node)[2]
        print(" ==== Prob Sum 2 : ",str(sum(self.belief))," || ",max(self.belief))

         # Updating Priors with fact that prey not at survey location
        # print("Updating node : ",survey_node)
        if not survey_res:
            print("Not at ",survey_node)
        else:
            print("At ",survey_node)
        self.belief = getNewBeliefs(self.belief,survey_node, survey_res)

        knows = [1,0] 
        if max(self.belief)==1.0:
            knows = [1,1]
        print(" ==== Prob Sum 3 : ",str(sum(self.belief))," || ",max(self.belief))
        print("Transitioning")
        # Transitioning prior probabilities        
        transitions  = transitionProbabilities(self.belief,graph.info)
        print(" ==== Prob Sum 4 : ",str(sum(self.belief))," || ",max(self.belief))

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
                for p in range(len(transitions)):
                    valOfTmpState += transitions[p]* self.vals[(action,p,pred)]

                valOfAction += probOfStateTransition * valOfTmpState

            if valOfAction < mnVal:
                mnVal = valOfAction
                self.nextPosition = action
        
        self.belief = transitions

        
        max_val = max(self.belief)
        max_beliefs = [i for i, v in enumerate(self.belief) if v==max_val]
        prey = random.choice(max_beliefs)
        Environment.getInstance().expected_prey = max_beliefs
        return knows