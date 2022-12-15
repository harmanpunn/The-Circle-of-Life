from agents.p2.agent1 import Agent1
from environment import Environment
from graph import Graph
from readDump import readDumpFile
from valueIteration import getProbs, getPolicyFromValues
from tqdm import tqdm
import pickle

environment = Environment(False,50)
graph = Graph()

data = readDumpFile(1)
graph.info = data[0]
values = data[1]
states = values.keys()

matrix = getProbs(graph, values)
policy = getPolicyFromValues(values, matrix)
# print(policy)

print(len(states))
pos_dict = {}

for state in tqdm(states):
    agent = state[0]
    prey = state[1]
    predator = state[2]

    agent_next_pos = Agent1.get_next_position(prey, predator, graph.info, agent) 
    pos_dict[(agent, prey, predator)] = agent_next_pos

print(len(pos_dict))



num = pos_dict.keys()
i = 0

nums = []
moves = []

for n in num:
    if n[0]==n[2]:
        continue
    elif n[0]==n[1]:
        continue

    if pos_dict[n]!=policy[n]:        
        nums.append(n)  
        moves.append([pos_dict[n],policy[n]])  
        i+=1
    
    
print(i)

pickle.dump((nums,moves),open("./changeStates","wb"))
