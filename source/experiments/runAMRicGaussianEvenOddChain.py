import numpy as np
import gym
import sys
import os
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')

import gym
import gym_ChainEnv

from source.agents.actMeasureRic import tabularDynamicsModels, actMeasureRIC
from source.agents.actMeasureTabularQLearning import actMeasureTabularQLearning


trialsStpsToTermLst = []
trialsObsToTermLst = []
trailObsPerStateLst = []
trialActCountsLst = []
trailRewardsCount = []

for n in [0.0,0.1,0.2,0.3,0.4,0.5]:
	env = gym.make("MeasurementGausEvenOddChainEnv-v0", noise=n,renderMode=None)

	numActions = env.action_space.n  
	numStates = env.observation_space.n
	env.reset()

	numTrials = 50
	numEpisodes = 700
	measureBias=0.1

	trialsStpsToTerm = np.ndarray(shape=(0,numEpisodes))
	trialsObsToTerm = np.ndarray(shape=(0,numEpisodes))
	trailObsPerState = np.repeat(0.0, numEpisodes*numStates).reshape(numEpisodes,numStates)
	trialActCounts = np.repeat(0.0, numEpisodes*numActions).reshape(numEpisodes,numActions)
	trailRewardsCount = np.ndarray(shape=(0,numEpisodes))

	for r in range(numTrials):
		ric = actMeasureRIC(numActions, numStates,measureBias=measureBias,qInit=0.0)
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
			done = False
			while not done:
				stps += 1.0
				state, action, reward, next_state, done= ric.act(env, state)
				epReward += reward
				epActCounts[action] += 1
				if action % 2 != 0:
					epObsPerState[state-1] += 1.0
					obs += 1.0
				ric.updateQ(state, action, reward, next_state)
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
					state = state[0]
					# plter.close()
		trialsStpsToTerm = np.concatenate([trialsStpsToTerm, stpsToTerm.reshape(1,numEpisodes)])
		trialsObsToTerm = np.concatenate([trialsObsToTerm, obsToTerm.reshape(1,numEpisodes)])
		trailRewardsCount = np.concatenate([trailRewardsCount, rwdsAtTerm.reshape(1,numEpisodes)])
		trailObsPerState += obsPerState
		trialActCounts += actCounts

	ricTrailObsPerState = trailObsPerState / float(numTrials)
	ricTrialActCounts   = trialActCounts  / float(numTrials)
	ricMeanStpsToTerm = np.mean(trialsStpsToTerm, axis = 0)
	ricStdStpsToTerm = np.std(trialsStpsToTerm, axis = 0)
	ricMeanObsToTerm = np.mean(trialsObsToTerm, axis = 0)
	ricStdObsToTerm = np.std(trialsObsToTerm, axis = 0)

	ricMeanReward = np.mean(trailRewardsCount, axis = 0)
	ricStdReward = np.std(trailRewardsCount, axis = 0)



ricTrailObsPerStateBs01 = ricTrailObsPerState
ricTrialActCountsBs01   = ricTrialActCounts
ricMeanStpsToTermBs01 = ricMeanStpsToTerm
ricStdStpsToTermBs01 = ricStdStpsToTerm
ricMeanObsToTermBs01 = ricMeanObsToTerm
ricStdObsToTermBs01 = ricStdObsToTerm
ricMeanRewardBs01 = ricMeanReward
ricStdRewardBs01 = ricStdReward

ricTrailObsPerStateBs1 = ricTrailObsPerState
ricTrialActCountsBs1   = ricTrialActCounts
ricMeanStpsToTermBs1 = ricMeanStpsToTerm
ricStdStpsToTermBs1 = ricStdStpsToTerm
ricMeanObsToTermBs1 = ricMeanObsToTerm
ricStdObsToTermBs1 = ricStdObsToTerm
ricMeanRewardBs1 = ricMeanReward
ricStdRewardBs1 = ricStdReward

ricTrailObsPerStateBs2 = ricTrailObsPerState
ricTrialActCountsBs2   = ricTrialActCounts
ricMeanStpsToTermBs2 = ricMeanStpsToTerm
ricStdStpsToTermBs2 = ricStdStpsToTerm
ricMeanObsToTermBs2 = ricMeanObsToTerm
ricStdObsToTermBs2 = ricStdObsToTerm
ricMeanRewardBs2 = ricMeanReward
ricStdRewardBs2 = ricStdReward

ricTrailObsPerStateBs10 = ricTrailObsPerState
ricTrialActCountsBs10   = ricTrialActCounts
ricMeanStpsToTermBs10 = ricMeanStpsToTerm
ricStdStpsToTermBs10 = ricStdStpsToTerm
ricMeanObsToTermBs10 = ricMeanObsToTerm
ricStdObsToTermBs10 = ricStdObsToTerm

ricMeanRewardBs10 = ricMeanReward
ricStdRewardBs10 = ricStdReward

from matplotlib import pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 22})


plt.close()
plt.figure(figsize=(10,5))
# plt.plot(np.arange(len(ricMeanRewardBs05)), ricMeanRewardBs05, 'k', linestyle='dotted', color='darkorange', label="Bias=0.5")
# plt.fill_between(np.arange(len(ricMeanRewardBs05)), ricMeanRewardBs05-ricStdRewardBs05, ricMeanRewardBs05+ricStdRewardBs05,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
plt.plot(np.arange(len(ricMeanRewardBs01)), ricMeanRewardBs01, 'k', color='red', label="Beta=0.01, Noise=0")
plt.fill_between(np.arange(len(ricMeanRewardBs01)), ricMeanRewardBs01-ricStdRewardBs01, ricMeanRewardBs01+ricStdRewardBs01,alpha=0.2, edgecolor='red', facecolor='red')
plt.plot(np.arange(len(ricMeanRewardBs1)), ricMeanRewardBs1, 'k', linestyle='-.', color='#1B2ACC', label="Beta=1, Noise=0")
plt.fill_between(np.arange(len(ricMeanRewardBs1)), ricMeanRewardBs1-ricStdRewardBs1, ricMeanRewardBs1+ricStdRewardBs1,alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')
plt.plot(np.arange(len(ricMeanRewardBs2)), ricMeanRewardBs2, 'k', linestyle='--', color='seagreen', label="Beta=2, Noise=0")
plt.fill_between(np.arange(len(ricMeanRewardBs2)), ricMeanRewardBs2-ricStdRewardBs2, ricMeanRewardBs2+ricStdRewardBs2,alpha=0.2, edgecolor='seagreen', facecolor='seagreen')
plt.legend()
plt.ylim(0, 300)
plt.xlabel("Episodes")
plt.ylabel("Mean Costed Return")
plt.savefig("ricEvenOddNoise0ChainReturn.png")
plt.show()

plt.close()
plt.figure(figsize=(10,5))
# plt.plot(np.arange(len(ricMeanStpsToTermBs05)), ricMeanStpsToTermBs05, 'k', linestyle='dotted', color='darkorange', label="Bias=0.5")
# plt.fill_between(np.arange(len(ricMeanStpsToTermBs05)), ricMeanStpsToTermBs05-ricStdStpsToTermBs05, ricMeanStpsToTermBs05+ricStdStpsToTermBs05,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
plt.plot(np.arange(len(ricMeanStpsToTermBs01)), ricMeanStpsToTermBs01, 'k', color='red', label="Beta=0.1, Noise=0")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs01)), ricMeanStpsToTermBs01-ricStdStpsToTermBs01, ricMeanStpsToTermBs01+ricStdStpsToTermBs01,alpha=0.2, edgecolor='red', facecolor='red')
plt.plot(np.arange(len(ricMeanStpsToTermBs1)), ricMeanStpsToTermBs1, 'k', linestyle='-.', color='#1B2ACC', label="Beta=1, Noise=0")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs1)), ricMeanStpsToTermBs1-ricStdStpsToTermBs1, ricMeanStpsToTermBs1+ricStdStpsToTermBs1,alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')
plt.plot(np.arange(len(ricMeanStpsToTermBs2)), ricMeanStpsToTermBs2, 'k', linestyle='--', color='seagreen', label="Beta=2, Noise=0")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs2)), ricMeanStpsToTermBs2-ricStdStpsToTermBs2, ricMeanStpsToTermBs2+ricStdStpsToTermBs2,alpha=0.2, edgecolor='seagreen', facecolor='seagreen')
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Mean Steps")
plt.ylim(0, 200)
plt.savefig("ricEvenOddChainNoise0Steps.png")
plt.show()



#fixed bias with different levels of noise
env = gym.make("MeasurementEvenOddChainEnv-v0", noise=0)
env.renderMode=None

numActions = env.action_space.n  
numStates = env.observation_space.n
env.reset()


numTrials = 50
numEpisodes = 500
measureBias=0.1


trialsStpsToTerm = np.ndarray(shape=(0,numEpisodes))
trialsObsToTerm = np.ndarray(shape=(0,numEpisodes))
trailObsPerState = np.repeat(0.0, numEpisodes*numStates).reshape(numEpisodes,numStates)
trialActCounts = np.repeat(0.0, numEpisodes*numActions).reshape(numEpisodes,numActions)
trailRewardsCount = np.ndarray(shape=(0,numEpisodes))


for r in range(numTrials):
	# time.sleep(5)
	ric = actMeasureRIC(numActions, numStates,measureBias=measureBias,qInit=0.0)
	state = env.reset()[0]
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
			state, action, reward, next_state, done = ric.act(env, state)
			epReward += reward
			# print("current state: " + str(state) + " action: " + str(action) + " next state: " + str(next_state))
			epActCounts[action] += 1
			# if state < 20  and action % 2 != 0:
			if action % 2 != 0:
				epObsPerState[state-1] += 1.0
				obs += 1.0
			ric.updateQ(state, action, reward, next_state)
			state = next_state
			if done:
				# if e % 50: 
				# 	averagedQtables[int(e/50)] += ric.qAgent.Q
				stpsToTerm = np.append(stpsToTerm, stps)
				obsToTerm = np.append(obsToTerm, obs)
				rwdsAtTerm = np.append(rwdsAtTerm, epReward)
				obsPerState = np.concatenate([obsPerState, epObsPerState.reshape(1,numStates)])
				actCounts = np.concatenate([actCounts, epActCounts.reshape(1,numActions)])
				state = env.reset()
				# plter.close()
				state = state[0]
	trialsStpsToTerm = np.concatenate([trialsStpsToTerm, stpsToTerm.reshape(1,numEpisodes)])
	trialsObsToTerm = np.concatenate([trialsObsToTerm, obsToTerm.reshape(1,numEpisodes)])
	trailRewardsCount = np.concatenate([trailRewardsCount, rwdsAtTerm.reshape(1,numEpisodes)])
	trailObsPerState += obsPerState
	trialActCounts += actCounts

ricTrailObsPerState = trailObsPerState / float(numTrials)
ricTrialActCounts   = trialActCounts  / float(numTrials)
ricMeanStpsToTerm = np.mean(trialsStpsToTerm, axis = 0)
ricStdStpsToTerm = np.std(trialsStpsToTerm, axis = 0)
ricMeanObsToTerm = np.mean(trialsObsToTerm, axis = 0)
ricStdObsToTerm = np.std(trialsObsToTerm, axis = 0)

ricMeanReward = np.mean(trailRewardsCount, axis = 0)
ricStdReward = np.std(trailRewardsCount, axis = 0)



ricTrailObsPerStateBs05Noise0 = ricTrailObsPerState
ricTrialActCountsBs05Noise0   = ricTrialActCounts
ricMeanStpsToTermBs05Noise0 = ricMeanStpsToTerm
ricStdStpsToTermBs05Noise0 = ricStdStpsToTerm
ricMeanObsToTermBs05Noise0 = ricMeanObsToTerm
ricStdObsToTermBs05Noise0 = ricStdObsToTerm
ricMeanRewardBs05Noise0 = ricMeanReward
ricStdRewardBs05Noise0 = ricStdReward

ricTrailObsPerStateBs05Noise001 = ricTrailObsPerState
ricTrialActCountsBs05Noise001 = ricTrialActCounts
ricMeanStpsToTermBs05Noise001 = ricMeanStpsToTerm
ricStdStpsToTermBs05Noise001 = ricStdStpsToTerm
ricMeanObsToTermBs05Noise001 = ricMeanObsToTerm
ricStdObsToTermBs05Noise001 = ricStdObsToTerm
ricMeanRewardBs05Noise001 = ricMeanReward
ricStdRewardBs05Noise001 = ricStdReward


ricTrailObsPerStateBs05Noise005 = ricTrailObsPerState
ricTrialActCountsBs05Noise005 = ricTrialActCounts
ricMeanStpsToTermBs05Noise005 = ricMeanStpsToTerm
ricStdStpsToTermBs05Noise005 = ricStdStpsToTerm
ricMeanObsToTermBs05Noise005 = ricMeanObsToTerm
ricStdObsToTermBs05Noise005 = ricStdObsToTerm
ricMeanRewardBs05Noise005 = ricMeanReward
ricStdRewardBs05Noise005 = ricStdReward

ricTrailObsPerStateBs05Noise01 = ricTrailObsPerState
ricTrialActCountsBs05Noise01 = ricTrialActCounts
ricMeanStpsToTermBs05Noise01 = ricMeanStpsToTerm
ricStdStpsToTermBs05Noise01 = ricStdStpsToTerm
ricMeanObsToTermBs05Noise01 = ricMeanObsToTerm
ricStdObsToTermBs05Noise01 = ricStdObsToTerm
ricMeanRewardBs05Noise01 = ricMeanReward
ricStdRewardBs05Noise01 = ricStdReward

plt.rcParams.update({'font.size': 22})

plt.close()
plt.figure(figsize=(10,5))
# plt.plot(np.arange(len(ricMeanRewardBs05)), ricMeanRewardBs05, 'k', linestyle='dotted', color='darkorange', label="Bias=0.5")
# plt.fill_between(np.arange(len(ricMeanRewardBs05)), ricMeanRewardBs05-ricStdRewardBs05, ricMeanRewardBs05+ricStdRewardBs05,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
plt.plot(np.arange(len(ricMeanRewardBs05Noise0)), ricMeanRewardBs05Noise0, 'k', color='red', label="Beta=0.1, Noise=0")
plt.fill_between(np.arange(len(ricMeanRewardBs05Noise0)), ricMeanRewardBs05Noise0-ricStdRewardBs05Noise0, ricMeanRewardBs05Noise0+ricStdRewardBs05Noise0,alpha=0.2, edgecolor='red', facecolor='red')
plt.plot(np.arange(len(ricMeanRewardBs05Noise001)), ricMeanRewardBs05Noise001, 'k', linestyle='-.', color='#1B2ACC', label="Beta=0.1, Noise=0.01")
plt.fill_between(np.arange(len(ricMeanRewardBs05Noise001)), ricMeanRewardBs05Noise001-ricStdRewardBs05Noise001, ricMeanRewardBs05Noise001+ricStdRewardBs05Noise001,alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')
plt.plot(np.arange(len(ricMeanRewardBs05Noise005)), ricMeanRewardBs05Noise005, 'k', linestyle='--', color='seagreen', label="Beta=0.1, Noise=0.05")
plt.fill_between(np.arange(len(ricMeanRewardBs05Noise005)), ricMeanRewardBs05Noise005-ricStdRewardBs05Noise005, ricMeanRewardBs05Noise005+ricStdRewardBs05Noise005,alpha=0.2, edgecolor='seagreen', facecolor='seagreen')
# plt.plot(np.arange(len(ricMeanRewardBs05Noise01)), ricMeanRewardBs05Noise01, 'k', linestyle='dotted', color='darkorange', label="Beta=1, Noise=0.1")
# plt.fill_between(np.arange(len(ricMeanRewardBs05Noise01)), ricMeanRewardBs05Noise01-ricStdRewardBs05Noise01, ricMeanRewardBs05Noise01+ricStdRewardBs05Noise01,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
plt.legend()
plt.ylim(-100, 350)
plt.xlabel("Episodes")
plt.ylabel("Mean Costed Return")
plt.savefig("ricEvenOddBeta01ChainReturn.png")
plt.show()






plt.close()
plt.figure(figsize=(10,5))
# plt.plot(np.arange(len(ricMeanRewardBs05)), ricMeanRewardBs05, 'k', linestyle='dotted', color='darkorange', label="Bias=0.5")
# plt.fill_between(np.arange(len(ricMeanRewardBs05)), ricMeanRewardBs05-ricStdRewardBs05, ricMeanRewardBs05+ricStdRewardBs05,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
plt.plot(np.arange(len(ricMeanStpsToTermBs05Noise0)), ricMeanStpsToTermBs05Noise0, 'k', color='red', label="Beta=0.1, Noise=0")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs05Noise0)), ricMeanStpsToTermBs05Noise0-ricStdStpsToTermBs05Noise0, ricMeanStpsToTermBs05Noise0+ricStdStpsToTermBs05Noise0,alpha=0.2, edgecolor='red', facecolor='red')
plt.plot(np.arange(len(ricMeanStpsToTermBs05Noise001)), ricMeanStpsToTermBs05Noise001, 'k', linestyle='-.', color='#1B2ACC', label="Beta=0.1, Noise=0.01")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs05Noise001)), ricMeanStpsToTermBs05Noise001-ricStdStpsToTermBs05Noise001, ricMeanStpsToTermBs05Noise001+ricStdStpsToTermBs05Noise001,alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')
plt.plot(np.arange(len(ricMeanStpsToTermBs05Noise005)), ricMeanStpsToTermBs05Noise005, 'k', linestyle='--', color='seagreen', label="Beta=0.1, Noise=0.05")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs05Noise005)), ricMeanStpsToTermBs05Noise005-ricStdStpsToTermBs05Noise005, ricMeanStpsToTermBs05Noise005+ricStdStpsToTermBs05Noise005,alpha=0.2, edgecolor='seagreen', facecolor='seagreen')
# plt.plot(np.arange(len(ricMeanStpsToTermBs05Noise01)), ricMeanStpsToTermBs05Noise01, 'k', linestyle='dotted', color='darkorange', label="Beta=1, Noise=0.1")
# plt.fill_between(np.arange(len(ricMeanStpsToTermBs05Noise01)), ricMeanStpsToTermBs05Noise01-ricStdStpsToTermBs05Noise01, ricMeanStpsToTermBs05Noise01+ricStdStpsToTermBs05Noise01,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Mean Steps")
plt.ylim(0, 250)
plt.savefig("ricEvenOddChainBeta01Steps.png")
plt.show()
