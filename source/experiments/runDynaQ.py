import sys
import os
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')

import gym
import numpy as np 
import sys 
from source.gym_experiments.agents.dynaQ import dynaQ
import gym_TabularPhaseChangeEnv
import gym_openAIwithObsCostsEnv

numTrials = 50
numEpisodes = 10000
obs_cost=0.1
dynaModelSteps = 5

envName = "JrSci"
env = gym.make("WaterTabularEpisodicWithLandingNoAutoTermination-v0")

# envName = "FrozenLake8x8_Slip"
# env = gym.make("costWrapper-v0",env=gym.make("FrozenLake8x8-v1",is_slippery=True),obs_cost=obs_cost)

# envName = "FrozenLake8x8_NSlip"
# env = gym.make("costWrapper-v0",env=gym.make("FrozenLake8x8-v1",is_slippery=False),obs_cost=obs_cost)

# envName = "Taxi-V3"
# env = gym.make("costWrapper-v0",env=gym.make("Taxi-v3"),obs_cost=obs_cost)

# envName = "StandardGausEvenOddChainEnv"
# env = gym.make("StandardGausEvenOddChainEnv-v0", noise=n, renderMode=None)

numActions = env.action_space.n 
numStates = env.observation_space.n
env.reset()

dynaqTrailObsPerStateLst = []
dynaqTrialActCountsLst = []
dynaqMeanStpsToTermLst = []
dynaqStdStpsToTermLst = []
dynaqMeanObsToTermLst = []
dynaqStdObsToTermLst = []

dynaqMeanRewardLst = []
dynaqStdRewardLst = []

dynaqtrialsStpsToTerm = np.ndarray(shape=(0,numEpisodes))
dynaqtrialsObsToTerm = np.ndarray(shape=(0,numEpisodes))
dynaqtrailObsPerState = np.repeat(0.0, numEpisodes*numStates).reshape(numEpisodes,numStates)
dynaqtrialActCounts = np.repeat(0.0, numEpisodes*numActions).reshape(numEpisodes,numActions)
dynaqtrailRewardsCount = np.ndarray(shape=(0,numEpisodes))

for r in range(numTrials):
    agent = dynaQ(numActions, numStates,qInit=0.0)
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
        done = False
        while not done:
            stps += 1.0
            action = agent.epGreedyAction(state)
            next_state, reward, done, _, _ = env.step(action)
            reward -= obs_cost # account for the measurement cost
            epActCounts[action] += 1
            epReward += reward 
            agent.updateQTable(state, action, reward, next_state)
            agent.updateModel(state, action, reward, next_state)
            state = next_state
            for _ in range(dynaModelSteps):
                dState, dAction, dReward, dNext_state = agent.sampleModel()
                agent.updateQTable(dState, dAction, dReward, dNext_state)
            epActCounts[action] += 1
            epObsPerState[state-1] += 1.0
            obs += 1.0
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

np.save('dynaQ'+envName +'RewardStd.npy', dynaqStdRewardLst)
np.save('dynaQ'+envName +'RewardMean.npy', dynaqMeanRewardLst)
np.save('dynaQ'+envName +'ObsMean.npy', dynaqMeanObsToTermLst)
np.save('dynaQ'+envName +'ObsStd.npy', dynaqStdObsToTermLst)
np.save('dynaQ'+envName +'StpsMean.npy', dynaqMeanStpsToTermLst)
np.save('dynaQ'+envName +'StpsStd.npy', dynaqStdStpsToTermLst)
