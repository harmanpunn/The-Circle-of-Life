from graph import Graph
from valueIteration import getPolicyFromValues, getValues, getProbs
from renderer import Renderer
from environment import Environment
import pygame
import math
import pickle
import os
from time import sleep
from random import shuffle

env = Environment(True,50)

env.explore = True

filePath = "./data-1"
fileStatePath = "./changeStates"

graph = Graph()

print(os.path.exists(filePath))

info, _ = pickle.load(open(filePath,"rb"))
graph.info = info

states, moves = pickle.load(open("./changeStates2","rb"))

idx =  [i for i in range(0,len(states))]
shuffle(idx)

states = [states[i] for i in idx]
moves = [moves[i] for i in idx]


renderer = Renderer(graph)

stateToGraph = {
    0 : 1,
    1 : 2,
    2 : 0
}

i = 0

for x in range(0,3):
    graph.node_states[states[i][x]][stateToGraph[x]] = True

while True:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        for x in range(0,3):
            graph.node_states[states[i][x]][stateToGraph[x]] = False
        i -= 1
        i %= len(states)
        for x in range(0,3):
            graph.node_states[states[i][x]][stateToGraph[x]] = True
    if keys[pygame.K_RIGHT]:
        for x in range(0,3):
            graph.node_states[states[i][x]][stateToGraph[x]] = False
        i += 1
        i %= len(states)
        for x in range(0,3):
            graph.node_states[states[i][x]][stateToGraph[x]] = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
    Environment.getInstance().state = states[i]
    Environment.getInstance().stateInfo['value'] = i
    Environment.getInstance().stateInfo['move'] = str(moves[i])
    
    renderer.__render__()
    sleep(0.1)
