
import sys
import os
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')

import gym
import gym_ChainEnv
import matplotlib.pyplot as plt
import numpy as np 
import sys 

from source.agents.dynaQ import dynaQ

plt.rcParams.update({'font.size': 22})


dynaqTrailObsPerStateLst = []
dynaqTrialActCountsLst = []
dynaqMeanStpsToTermLst = []
dynaqStdStpsToTermLst = []
dynaqMeanObsToTermLst = []
dynaqStdObsToTermLst = []

dynaqMeanRewardLst = []
dynaqStdRewardLst = []

for n in [0.0,0.1,0.2,0.3,0.4,0.5]:
    env = gym.make("StandardGausEvenOddChainEnv-v0", noise=n, renderMode=None)
    env.renderMode=None
    
    numActions = env.action_space.n  
    numStates = env.observation_space.n
    env.reset()
    
    numTrials = 50
    numEpisodes = 700
    measureBias=0.1
    dynaModelSteps = 5
    
    dynaqtrialsStpsToTerm = np.ndarray(shape=(0,numEpisodes))
    dynaqtrialsObsToTerm = np.ndarray(shape=(0,numEpisodes))
    dynaqtrailObsPerState = np.repeat(0.0, numEpisodes*numStates).reshape(numEpisodes,numStates)
    dynaqtrialActCounts = np.repeat(0.0, numEpisodes*numActions).reshape(numEpisodes,numActions)
    dynaqtrailRewardsCount = np.ndarray(shape=(0,numEpisodes))
    
    for r in range(numTrials):
        agent = dynaQ(2, numStates,qInit=0.0)
        state, _ = env.reset()[0]
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
                next_state, reward, done, trun, m = env.step(action)
                next_state = next_state[0]
                epReward += reward - 5 # minus measurement cost
                epActCounts[action] += 1
                epObsPerState[state-1] += 1.0
                obs += 1.0
                agent.updateQTable(state, action, reward, next_state)
                agent.updateModel(state, action, reward, next_state)
                state = next_state
                for _ in range(dynaModelSteps):
                    dState, dAction, dReward, dNext_state = agent.sampleModel()
                    agent.updateQTable(dState, dAction, dReward, dNext_state)          
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
        dynaqtrialsStpsToTerm = np.concatenate([dynaqtrialsStpsToTerm, stpsToTerm.reshape(1,numEpisodes)])
        dynaqtrialsObsToTerm = np.concatenate([dynaqtrialsObsToTerm, obsToTerm.reshape(1,numEpisodes)])
        dynaqtrailRewardsCount = np.concatenate([dynaqtrailRewardsCount, rwdsAtTerm.reshape(1,numEpisodes)])
        dynaqtrailObsPerState += obsPerState
        dynaqtrialActCounts += actCounts
    
    dynaqTrailObsPerState = dynaqtrailObsPerState / float(numTrials)
    dynaqTrialActCounts   = dynaqtrialActCounts  / float(numTrials)
    dynaqMeanStpsToTerm = np.mean(dynaqtrialsStpsToTerm, axis = 0)
    dynaqStdStpsToTerm = np.std(dynaqtrialsStpsToTerm, axis = 0)
    dynaqMeanObsToTerm = np.mean(dynaqtrialsObsToTerm, axis = 0)
    dynaqStdObsToTerm = np.std(dynaqtrialsObsToTerm, axis = 0)
    
    dynaqMeanReward = np.mean(dynaqtrailRewardsCount, axis = 0)
    dynaqStdReward = np.std(dynaqtrailRewardsCount, axis = 0)
    
    dynaqTrailObsPerStateLst = dynaqTrailObsPerStateLst+[dynaqTrailObsPerState]
    dynaqTrialActCountsLst = dynaqTrialActCountsLst+[dynaqTrialActCounts]
    dynaqMeanStpsToTermLst = dynaqMeanStpsToTermLst+[dynaqMeanStpsToTerm]
    dynaqStdStpsToTermLst = dynaqStdStpsToTermLst+[dynaqStdStpsToTerm]
    dynaqMeanObsToTermLst = dynaqMeanObsToTermLst+[dynaqMeanObsToTerm]
    dynaqStdObsToTermLst = dynaqStdObsToTermLst+[dynaqStdObsToTerm]
    
    dynaqMeanRewardLst = dynaqMeanRewardLst+[dynaqMeanReward]
    dynaqStdRewardLst = dynaqStdRewardLst+[dynaqStdReward]





dynaqMeans = np.mean(np.vstack(dynaqMeanRewardLst)[:,650:],axis=1)
dynaqStds = np.mean(np.vstack(dynaqStdRewardLst)[:,650:],axis=1)
plt.plot(np.array([0.0,0.1,0.2,0.3,0.4,0.5]), dynaqMeans, 'k', color='blue', label="Amrl")
plt.fill_between(np.array([0.0,0.1,0.2,0.3,0.4,0.5]), dynaqMeans-dynaqStds, dynaqMeans+dynaqStds,alpha=0.2, edgecolor='blue', facecolor='blue')
plt.ylim(0,300)

plt.legend()
plt.show()



