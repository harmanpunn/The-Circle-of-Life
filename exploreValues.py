from graph import Graph
from valueIteration import getPolicyFromValues, getValues, getProbs
from renderer import Renderer
from environment import Environment
import pygame
import math
import pickle
import os

Environment(True,50)

filePath = "./test.dump"

graph = Graph()

if not os.path.exists(filePath):
    vals, probs = getValues(graph)
    policy = getPolicyFromValues(vals, probs)
    pickle.dump([graph.info,vals],open(filePath,"wb"))
else:
    info, vals = pickle.load(open(filePath,"rb"))
    graph.info = info
    probs = getProbs(graph,vals)
    policy = getPolicyFromValues(vals,probs)

renderer = Renderer(graph)

stateToGraph = {
    0 : 1,
    1 : 2,
    2 : 0
}


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
        if event.type == pygame.MOUSEBUTTONUP:
            x,y = pygame.mouse.get_pos()
            found = False
            if len(Environment.getInstance().state)<3:
                for i,center in enumerate(renderer.node_centers):
                    if math.sqrt((center[0]-x)**2 + (center[1]-y)**2)<10:
                        graph.node_states[i][stateToGraph[Environment.getInstance().mouse_counter]] = True 
                        found = True
                        Environment.getInstance().state.append(i)
                        break
                if found:
                    Environment.getInstance().mouse_counter+=1
                    Environment.getInstance().mouse_counter %= 4
                if len(Environment.getInstance().state) ==3:
                    tp = tuple(Environment.getInstance().state)
                    Environment.getInstance().stateInfo['value'] = vals[tp]
                    Environment.getInstance().stateInfo['move'] = policy[tp]
            else:
                Environment.getInstance().mouse_counter+=1
                Environment.getInstance().mouse_counter %= 4
                for i in Environment.getInstance().state:
                    graph.node_states[i] = [False]*3
                Environment.getInstance().state = []
            break
    
    renderer.__render__()
