from collections import defaultdict
from random import *
from environment import Environment
from graph import Graph
from util import get_shortest_path

# environment = Environment(False,5)
# graph = Graph()


def getValues(graph: Graph):
    # values 
    values = {}
    # all transition probabilities
    prob_matrix = {}
    for agent in range(0, Environment.getInstance().node_count):
            for prey in range(0, Environment.getInstance().node_count):
                for predator in range(0, Environment.getInstance().node_count):
                    values[(agent, prey, predator)] = randint(0,99)
                    # game over
                    if agent==predator:
                        values[(agent, prey, predator)] = 9999
                    # win case
                    if agent == prey and agent!=predator:
                        values[(agent, prey, predator)] = 0


    states = values.keys()   
    # for all states
    for state in states:
        probs = {}
        # all possible actions
        actions = graph.info[state[0]] + [state[0]]
        for action in actions:
            action_probs = {}
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

            for p in prey_options:
                for r in predator_options:
                    # calculating probabilities
                    action_probs[(action, p, r)] = (1/len(prey_options))*(0.6*(1/len(predator_close)
                                                                                 if r in predator_close else 0.0)+0.4/len(predator_options))
            # recording probs according to actions
            probs[action] = action_probs
        # embedding probs of each state
        prob_matrix[state] = probs 


    i = 0  
    p = None
    # value iteration
    while p==None or (p!=None and p>1e-7):
        # TODO: Dump P to plot
        print('p:', p)
        p = 0.0
        # updating for each state
        for state in states:
            # skip game over case
            if state[0]==state[2]:
                continue
            # skip win case
            if state[0] == state[1] and state[0]!=state[2]:
                continue
            min_val = float('inf')
            # for each action to track min action
            for action in prob_matrix[state].keys():
                val = 1
                for s_prime in prob_matrix[state][action].keys():
                    val += prob_matrix[state][action][s_prime]*values[s_prime]

                min_val = min(val, min_val)    

            # accumulating change in each value
            p += (min_val - values[state]) ** 2
            # updating value
            values[state] = min_val
        i+=1
    # return all values & transitions
    return values, prob_matrix

"""
    Get policy from a value function
    parameters:
        values -> Value function
        prob_matrix -> Transition probabilities to prevent recalculation
"""
def getPolicyFromValues(values, prob_matrix):
    policy = {}
    for state in values.keys():
        min_val = float('inf')
        for action in prob_matrix[state].keys():
            val = 1
            # calculating value of an action
            for s_prime in prob_matrix[state][action].keys():
                val += prob_matrix[state][action][s_prime]*values[s_prime]

            if min_val > val:
                # taking min value action i.e, shortest steps to reach prey
                policy[state] = action
                min_val = min(val, min_val)
    #policy
    return policy
    

# values, probMatrix = getValues()         
# print(getPolicyFromValues(values,probMatrix))