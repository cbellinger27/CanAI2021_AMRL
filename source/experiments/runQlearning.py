
import sys
import os
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')

import gym
import matplotlib.pyplot as plt
import itertools 
import matplotlib 
import matplotlib.style 
import numpy as np 
from source.gym_experiments.agents.tabularQLearning import tabularQLearning
import gym_TabularPhaseChangeEnv
import gym_openAIwithObsCostsEnv
import gym_ChainEnv

qTrailObsPerStateLst = []
qTrialActCountsLst = []
qMeanStpsToTermLst = []
qStdStpsToTermLst = []
qMeanObsToTermLst = []
qStdObsToTermLst = []

qMeanRewardLst = []
qStdRewardLst = []

numTrials = 50
numEpisodes = 10000
obs_cost=0.1

envName = "JrSci"
env = gym.make("WaterTabularEpisodicWithLandingNoAutoTermination-v0")

# envName = "FrozenLake8x8_Slip"
# env = gym.make("costWrapper-v0",env=gym.make("FrozenLake8x8-v1",is_slippery=True),obs_cost=obs_cost)

# envName = "FrozenLake8x8_NSlip"
# env = gym.make("costWrapper-v0",env=gym.make("FrozenLake8x8-v1",is_slippery=False),obs_cost=obs_cost)

# envName = "Taxi-V3"
# env = gym.make("costWrapper-v0",env=gym.make("Taxi-v3"),obs_cost=obs_cost)

numActions = env.action_space.n 
numStates = env.observation_space.n
env.reset()

qtrialsStpsToTerm = np.ndarray(shape=(0,numEpisodes))
qtrialsObsToTerm = np.ndarray(shape=(0,numEpisodes))
qtrailObsPerState = np.repeat(0.0, numEpisodes*numStates).reshape(numEpisodes,numStates)
qtrialActCounts = np.repeat(0.0, numEpisodes*numActions).reshape(numEpisodes,numActions)
qtrailRewardsCount = np.ndarray(shape=(0,numEpisodes))

for r in range(numTrials):
    agent = tabularQLearning(numActions, numStates,qInit=0.0)
    state, _ = env.reset()
    env.renderMode = 'none'
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
            print(state)
            action = agent.epGreedyAction(state)
            next_state, reward, done, trun, m = env.step(action)
            reward -= obs_cost # account for the measurement cost
            epReward += reward 
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

np.save('qLrn'+envName +'RewardStd.npy', qStdRewardLst)
np.save('qLrn'+envName +'RewardMean.npy', qMeanRewardLst)
np.save('qLrn'+envName +'ObsMean.npy', qMeanObsToTermLst)
np.save('qLrn'+envName +'ObsStd.npy', qStdObsToTermLst)
np.save('qLrn'+envName +'StpsMean.npy', qMeanStpsToTermLst)
np.save('qLrn'+envName +'StpsStd.npy', qStdStpsToTermLst)
