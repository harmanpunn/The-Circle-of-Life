import sys
import os
from time import sleep
import pandas as pd
import pickle

from graphEntity import GraphEntity
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from environment import Environment
from graph import Graph
from renderer import Renderer
from tqdm import tqdm
from valueIteration import getPolicyFromValues, getProbs
import pygame

from agents.p2.agent1 import Agent1
from agents.p2.agent3 import Agent3
from agents.p2.agent5 import Agent5
from agents.p2.agent7 import Agent7

from agents.p3.p3Agent1 import P3Agent1
from agents.p3.p3Agent1Pred import P3Agent1Pred
from agents.p3.p3Agent2 import P3Agent2
from agents.p3.p3Agent2Pred import P3Agent2Pred

import numpy as np

from predator import Predator
from prey import Prey

from helper import processArgs

get_class = lambda x: globals()[x]


def runGame(graph : Graph, data = None):

    # graph = Graph()
    renderer =  Renderer(graph)
    # print("Initialized")
    # startGame()
    step_count = 0
    game_state = -2
    prey = Prey(graph)
    predator = Predator(graph)

    if not Environment.getInstance().p3:
        if Environment.getInstance().agent % 2 == 0:
            Environment.getInstance().careful = True

        if Environment.getInstance().agent < 3:
            agent : GraphEntity = Agent1(graph)
        elif Environment.getInstance().agent < 5:
            agent : GraphEntity = Agent3(graph) 
        elif Environment.getInstance().agent < 7:
            agent : GraphEntity = Agent5(graph) 
            agent.belief = [1.0 if i==predator.getPosition() else 0.0 for i in range(0,Environment.getInstance().node_count)]
        else:
            agent : GraphEntity = Agent7(graph) 
            agent.predator_belief = [1.0 if i==predator.getPosition() else 0.0 for i in range(0,Environment.getInstance().node_count)]        

            # agent : GraphEntity = get_class("Agent"+str(Environment.getInstance().agent))(graph)

        if Environment.getInstance().agent==9:
            Environment.getInstance().noisy_agent = True
            Environment.getInstance().noisy = True
            Environment.getInstance().careful = True

        if Environment.getInstance().agent==10:
            Environment.getInstance().noisy = False
            Environment.getInstance().noisy_agent = False
            Environment.getInstance().careful = True
            Environment.getInstance().agentX = True

            if Environment.getInstance().agent==10:
                Environment.getInstance().noisy = False
                Environment.getInstance().noisy_agent = False
                Environment.getInstance().careful = True
                Environment.getInstance().agentX = True
    else:
        if Environment.getInstance().agent==1:
            agent : GraphEntity = P3Agent1(graph,vals=data["vals"])
        elif Environment.getInstance().agent==2:
            agent : GraphEntity = P3Agent1Pred(graph)
            agent.values = data["vals"]
        elif Environment.getInstance().agent==3:
            agent : GraphEntity = P3Agent2(graph)
            agent.vals = data["vals"]
        elif Environment.getInstance().agent==4:
            agent : GraphEntity = P3Agent2Pred(graph,filePath="./modelDump/VPartialModel")
            agent.training = False
        elif Environment.getInstance().agent==5:
            agent : GraphEntity = P3Agent2Pred(graph,useV=True)
            agent.training = False

    running = 1

    if Environment.getInstance().noisy:
        print("So NOISY!")
    
    if Environment.getInstance().careful:
        print("TipToe B)")

    knownRounds = None
    while True:
        if Environment.getInstance().ui:
            sleep(0.2)
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running =False
        if running==1:
            graph.surveyed = False

            info = {}
            if not Environment.getInstance().p3:
                if Environment.getInstance().agent<3:
                    info = {
                        'prey' : prey.getPosition(),
                        'predator' : predator.getPosition()
                    }
                elif Environment.getInstance().agent<5:
                    info = {
                        'predator' : predator.getPosition()
                    }
                elif Environment.getInstance().agent<7:
                    info = {
                        'prey' : prey.getPosition()
                    }
            else:
                if Environment.getInstance().agent <3:
                    info = {
                        'prey' : prey.getPosition(),
                        'predator' : predator.getPosition()
                    }
                else:
                    info = {
                        # 'prey' : prey.getPosition(),
                        'predator' : predator.getPosition()
                    }

            graph.node_states_blocked= True
            knows = agent.__update__(graph, info)

            if knownRounds!=None:
                knownRounds = [knows[i]+knownRounds[i] for i in range(0,len(knownRounds))]
            else:
                knownRounds = knows
                
            graph.node_states_blocked = False
            
            predator.__update__(graph, {'agent':agent.getPosition()})
            prey.__update__(graph)
            renderer.__render__(running)
           
            if agent.getPosition() == prey.getPosition():
                print('Agent Wins :)')
                running = 2
                # Agent catches its prey
                game_state = 1

            if agent.getPosition() == predator.getPosition():
                print('Agent Loses :(')
                running = 0
                # Agent caught by predator
                game_state = 0

            if step_count > 10000:
                running = 0
                # Timeout
                game_state = -1 
            step_count+=1
        else:
            if Environment.getInstance().ui:
                renderer.__render__(running)
                sleep(2)
            break
    graph.reset_states()    
    knownRounds = [k/step_count for k in knownRounds]
    if Environment.getInstance().ui:
        pygame.quit()
    return [step_count, game_state, knownRounds]  

def collectData(cached= False,path=None) -> None:
    stats_dict = dict()
    step_count_list = {0:[],1:[],-1:[]}
    game_state_list = list()
    type_list = list()
    totalConfidences = [[],[]]
    for i in  tqdm(range(0,Environment.getInstance().graphs)):
        graph = Graph()
        vals = None
        if cached:
            graph.info, vals = pickle.load(open("data-1","rb"))
        type = i
        confidencePerGraph = [0.0,0.0] 
        for _ in tqdm(range(0,Environment.getInstance().games),leave=False):
            [step_count, game_state, confidence] = runGame(graph,{"vals":vals}) 
            step_count_list[game_state].append(step_count)
            game_state_list.append(game_state)
            type_list.append(type)
            confidencePerGraph = [confidencePerGraph[i]+ confidence[i] for i in range(0, len(confidence))]
            
        confidencePerGraph = [x/Environment.getInstance().games for x in confidencePerGraph]
        for i in range(0,len(confidencePerGraph)):
            totalConfidences[i].append(confidencePerGraph[i])
        
    
    for k in step_count_list:
        if len(step_count_list[k])!=0:
            step_count_list[k] = "Mean:"+ str(np.array(step_count_list[k]).mean()) +" || Std: "+str(np.array(step_count_list[k]).std())
        else:
            step_count_list[k] = "N/A"
    for t in totalConfidences:
        t = np.array(t)
        
    

    win_count = game_state_list.count(1)
    lose_count = game_state_list.count(0)
    timeout_count = game_state_list.count(-1)
    sys.stdout = sys.__stdout__

    z = Environment.getInstance().games * Environment.getInstance().graphs
    print("========== GAME STATS ==========")
    print("Predator confidence : Mean: ",np.mean(totalConfidences[0])," || Standard Deviation: ",np.std(totalConfidences[0]))
    print("Prey confidence : Mean: ",np.mean(totalConfidences[1])," || Standard Deviation: ",np.std(totalConfidences[1]))
    print("Win Step Counts: ", step_count_list[1])
    print("Win %: ", (win_count/z) * 100)
    print("Lose Step Count: ",step_count_list[0])
    print("Lose %: ", (lose_count/z) * 100)
    print("Timeout Step Count: ",step_count_list[-1])
    print("Timeout %: ", (timeout_count/z) * 100)
    print("================================")
    '''    
    stats_dict['graph_type'] = 1
    stats_dict['step_count'] = step_count
    stats_dict['game_state'] = game_state
    '''
    # print(stats_dict) 
    # stats_df = pd.DataFrame(columns=['Graph Type', 'Step Count', 'Game State'])
    # stats_df.loc[len(stats_df.index)] = []
    # stats_df.loc[len()]
    # stats_df = pd.DataFrame(data = stats_dict)
    # stats_df.to_csv('Agent'+str(Environment.getInstance().agent)+'.csv', index=False)
    pass


if __name__ == "__main__":
    args = processArgs()
    env = Environment(True,50)
    for x in args.keys():
        setattr(env,x,args[x]) 
    
    if Environment.getInstance().quiet==True:
        sys.stdout = open(os.devnull, 'w')
    if 'mode' in args.keys() and args['mode']==1:
        print("Mode different")
        Environment.getInstance().ui = False
        if not Environment.getInstance().p3:
            collectData()
        else:
            collectData(True,"./datadump/data-")
    else:
        graph = Graph()

        if Environment.getInstance().p3:
            x = pickle.load(open("data-1","rb"))
            graph.info = x[0]
        runGame(graph,{"vals":x[1]})


