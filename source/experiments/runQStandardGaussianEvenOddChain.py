
import sys
import os
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')

import gym
import gym_ChainEnv

import matplotlib.pyplot as plt
import itertools 
import matplotlib 
import matplotlib.style 
import numpy as np 
from collections import defaultdict 
from source.gym_experiments.agents.tabularQLearning import tabularQLearning

plt.rcParams.update({'font.size': 22})

qTrailObsPerStateLst = []
qTrialActCountsLst = []
qMeanStpsToTermLst = []
qStdStpsToTermLst = []
qMeanObsToTermLst = []
qStdObsToTermLst = []

qMeanRewardLst = []
qStdRewardLst = []

for n in [0.0,0.1,0.2,0.3,0.4,0.5]:
    env = gym.make("StandardGausEvenOddChainEnv-v0", noise=n, renderMode=None)
    env.renderMode=None
    
    numActions = env.action_space.n  
    numStates = env.observation_space.n
    env.reset()
    
    numTrials = 50
    numEpisodes = 700
    measureBias=0.1
    
    qtrialsStpsToTerm = np.ndarray(shape=(0,numEpisodes))
    qtrialsObsToTerm = np.ndarray(shape=(0,numEpisodes))
    qtrailObsPerState = np.repeat(0.0, numEpisodes*numStates).reshape(numEpisodes,numStates)
    qtrialActCounts = np.repeat(0.0, numEpisodes*numActions).reshape(numEpisodes,numActions)
    qtrailRewardsCount = np.ndarray(shape=(0,numEpisodes))
    
    for r in range(numTrials):
        # time.sleep(5)
        agent = tabularQLearning(2, numStates,qInit=0.0)
        state, _ = env.reset()
        state = state[0]
        stpsToTerm = np.array([])
        obsToTerm = np.array([])
        rwdsAtTerm = np.array([])
        obsPerState = np.ndarray(shape=(0,numStates))
        actCounts = np.ndarray(shape=(0, numActions))
        for e in range(0, numEpisodes):
            stps = 0
            obs = 0
            epReward = 0
            epObsPerState = np.repeat(0,numStates)
            epActCounts = np.repeat(0, numActions)
            # plter = plotter.Plotter(xMin=0,xMax=10, xStart=0, xTerminal=11, epsNum=e)
            done = False
            while not done:
                stps += 1.0
                action = agent.epGreedyAction(state)
                next_state, reward, done, trunc, m = env.step(action)
                next_state = next_state[0]
                epReward += reward - 5 # minus measurement cost
                epActCounts[action] += 1
                epObsPerState[state-1] += 1.0
                obs += 1.0
                agent.updateQTable(state, action, reward, next_state)
                state = next_state          
                if done:
                    stpsToTerm = np.append(stpsToTerm, stps)
                    obsToTerm = np.append(obsToTerm, obs)
                    rwdsAtTerm = np.append(rwdsAtTerm, epReward)
                    obsPerState = np.concatenate([obsPerState, epObsPerState.reshape(1,numStates)])
                    actCounts = np.concatenate([actCounts, epActCounts.reshape(1,numActions)])
                    if e % 100 == 0:
                        print("Trial " +str(r) +" episode: " + str(e) +" -- steps to term: " + str(stps) + " obs to term: " + str(obs) + " rewards at term: " + str(epReward))
                    state, _ = env.reset()
                    # plter.close()
                    state = state[0]
        qtrialsStpsToTerm = np.concatenate([qtrialsStpsToTerm, stpsToTerm.reshape(1,numEpisodes)])
        qtrialsObsToTerm = np.concatenate([qtrialsObsToTerm, obsToTerm.reshape(1,numEpisodes)])
        qtrailRewardsCount = np.concatenate([qtrailRewardsCount, rwdsAtTerm.reshape(1,numEpisodes)])
        qtrailObsPerState += obsPerState
        qtrialActCounts += actCounts
    
    qTrailObsPerState = qtrailObsPerState / float(numTrials)
    qTrialActCounts   = qtrialActCounts  / float(numTrials)
    qMeanStpsToTerm = np.mean(qtrialsStpsToTerm, axis = 0)
    qStdStpsToTerm = np.std(qtrialsStpsToTerm, axis = 0)
    qMeanObsToTerm = np.mean(qtrialsObsToTerm, axis = 0)
    qStdObsToTerm = np.std(qtrialsObsToTerm, axis = 0)
    
    qMeanReward = np.mean(qtrailRewardsCount, axis = 0)
    qStdReward = np.std(qtrailRewardsCount, axis = 0)
    
    qTrailObsPerStateLst = qTrailObsPerStateLst+[qTrailObsPerState]
    qTrialActCountsLst = qTrialActCountsLst+[qTrialActCounts]
    qMeanStpsToTermLst = qMeanStpsToTermLst+[qMeanStpsToTerm]
    qStdStpsToTermLst = qStdStpsToTermLst+[qStdStpsToTerm]
    qMeanObsToTermLst = qMeanObsToTermLst+[qMeanObsToTerm]
    qStdObsToTermLst = qStdObsToTermLst+[qStdObsToTerm]
    
    qMeanRewardLst = qMeanRewardLst+[qMeanReward]
    qStdRewardLst = qStdRewardLst+[qStdReward]





qMeans = np.mean(np.vstack(qMeanRewardLst)[:,650:],axis=1)
qStds = np.mean(np.vstack(qMeanRewardLst)[:,650:],axis=1)


plt.plot(np.array([0.0,0.1,0.2,0.3,0.4,0.5]), qMeans, 'k', color='blue', label="Amrl")
plt.fill_between(np.array([0.0,0.1,0.2,0.3,0.4,0.5]), qMeans-qStds, qMeans+qStds,alpha=0.2, edgecolor='blue', facecolor='blue')
plt.ylim(0,300)

plt.legend()
plt.show()
