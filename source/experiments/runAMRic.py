
import sys
import os
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import gym

from source.gym_experiments.agents.actMeasureRic import actMeasureRIC
import gym_TabularPhaseChangeEnv
import gym_openAIwithObsCostsEnv


numTrials = 50
numEpisodes = 10000
obs_cost=0.1
measureBias=10

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


trialsStpsToTerm = np.ndarray(shape=(0,numEpisodes))
trialsObsToTerm = np.ndarray(shape=(0,numEpisodes))
trailObsPerState = np.repeat(0.0, numEpisodes*numStates).reshape(numEpisodes,numStates)
trialActCounts = np.repeat(0.0, numEpisodes*numActions).reshape(numEpisodes,numActions)
trailRewardsCount = np.ndarray(shape=(0,numEpisodes))


for r in range(numTrials):
	ric = actMeasureRIC(numActions, numStates,measureBias=measureBias,qInit=0.0)
	state, _ = env.reset()
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



ricTrailObsPerStateBs05 = ricTrailObsPerState
ricTrialActCountsBs05   = ricTrialActCounts
ricMeanStpsToTermBs05 = ricMeanStpsToTerm
ricStdStpsToTermBs05 = ricStdStpsToTerm
ricMeanObsToTermBs05 = ricMeanObsToTerm
ricStdObsToTermBs05 = ricStdObsToTerm

ricMeanRewardBs05 = ricMeanReward
ricStdRewardBs05 = ricStdReward

ricTrailObsPerStateBs1 = ricTrailObsPerState
ricTrialActCountsBs1   = ricTrialActCounts
ricMeanStpsToTermBs1 = ricMeanStpsToTerm
ricStdStpsToTermBs1 = ricStdStpsToTerm
ricMeanObsToTermBs1 = ricMeanObsToTerm
ricStdObsToTermBs1 = ricStdObsToTerm

ricMeanRewardBs1 = ricMeanReward
ricStdRewardBs1 = ricStdReward

ricTrailObsPerStateBs5 = ricTrailObsPerState
ricTrialActCountsBs5   = ricTrialActCounts
ricMeanStpsToTermBs5 = ricMeanStpsToTerm
ricStdStpsToTermBs5 = ricStdStpsToTerm
ricMeanObsToTermBs5 = ricMeanObsToTerm
ricStdObsToTermBs5 = ricStdObsToTerm

ricMeanRewardBs5 = ricMeanReward
ricStdRewardBs5 = ricStdReward

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

plt.close()
plt.plot(np.arange(len(ricMeanRewardBs05)), ricMeanRewardBs05, 'k', linestyle='dotted', color='darkorange', label="Bias=0.5")
plt.fill_between(np.arange(len(ricMeanRewardBs05)), ricMeanRewardBs05-ricStdRewardBs05, ricMeanRewardBs05+ricStdRewardBs05,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
plt.plot(np.arange(len(ricMeanRewardBs1)), ricMeanRewardBs1, 'k', color='red', label="Bias=1")
plt.fill_between(np.arange(len(ricMeanRewardBs1)), ricMeanRewardBs1-ricStdRewardBs1, ricMeanRewardBs1+ricStdRewardBs1,alpha=0.2, edgecolor='red', facecolor='red')
plt.plot(np.arange(len(ricMeanRewardBs5)), ricMeanRewardBs5, 'k', linestyle='-.', color='#1B2ACC', label="Bias=5")
plt.fill_between(np.arange(len(ricMeanRewardBs5)), ricMeanRewardBs5-ricStdRewardBs5, ricMeanRewardBs5+ricStdRewardBs5,alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')
plt.plot(np.arange(len(ricMeanRewardBs10)), ricMeanRewardBs10, 'k', linestyle='--', color='seagreen', label="Bias=10")
plt.fill_between(np.arange(len(ricMeanRewardBs10)), ricMeanRewardBs10-ricStdRewardBs10, ricMeanRewardBs10+ricStdRewardBs10,alpha=0.2, edgecolor='seagreen', facecolor='seagreen')
plt.title("Env: " + envName)
plt.legend()
plt.show()

plt.close()
plt.plot(np.arange(len(ricMeanStpsToTermBs05)), ricMeanStpsToTermBs05, 'k', linestyle='dotted', color='darkorange', label="Bias=0.5")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs05)), ricMeanStpsToTermBs05-ricStdStpsToTermBs05, ricMeanStpsToTermBs05+ricStdStpsToTermBs05,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
plt.plot(np.arange(len(ricMeanStpsToTermBs1)), ricMeanStpsToTermBs1, 'k', color='red', label="Bias=1")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs1)), ricMeanStpsToTermBs1-ricStdStpsToTermBs1, ricMeanStpsToTermBs1+ricStdStpsToTermBs1,alpha=0.2, edgecolor='red', facecolor='red')
plt.plot(np.arange(len(ricMeanStpsToTermBs5)), ricMeanStpsToTermBs5, 'k', linestyle='-.', color='#1B2ACC', label="Bias=5")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs5)), ricMeanStpsToTermBs5-ricStdStpsToTermBs5, ricMeanStpsToTermBs5+ricStdStpsToTermBs5,alpha=0.2, edgecolor='#1B2ACC', facecolor='#1B2ACC')
plt.plot(np.arange(len(ricMeanStpsToTermBs10)), ricMeanStpsToTermBs10, 'k', linestyle='--', color='seagreen', label="Bias=10")
plt.fill_between(np.arange(len(ricMeanStpsToTermBs10)), ricMeanStpsToTermBs10-ricStdStpsToTermBs10, ricMeanStpsToTermBs10+ricStdStpsToTermBs10,alpha=0.2, edgecolor='seagreen', facecolor='seagreen')
plt.title("Env: " + envName)
plt.legend()
plt.show()
