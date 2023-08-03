
import sys
import os
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')

import numpy as np 
from source.agents.actMeasureTabularQLearning import actMeasureTabularQLearning


class statesActionCounts:
	def __init__(self, numStates, action):
		self.action = action
		self.transitions = np.repeat(0.0, numStates*numStates).reshape(numStates,numStates)
	def getNextState(self, state):
		return np.argmax(self.transitions[state-1,:])+1
	def updateTransitions(self, state, nextState):
		self.transitions[state-1,nextState-1] += 1

class tabularDynamicsModels:
	def __init__(self, numActions, numStates):
		self.numActions = numActions
		self.numStates = numStates
		self.transitionModels = []
		for i in range(numActions):
			self.transitionModels = self.transitionModels + [statesActionCounts(numStates, i)]
	def getNextState(self, state, action):
		return self.transitionModels[action].getNextState(state)
	def updateTransition(self, state, action, nextState):
		self.transitionModels[action].updateTransitions(state, nextState)

class actMeasureRIC:
	def __init__(self, numActions, numStates,measureBias=5.0, qInit=0.0):
		self.numActions = numActions
		self.numStates = numStates
		self.qAgent = actMeasureTabularQLearning(self.numActions, self.numStates, measureBias=measureBias, qInit=qInit)
		self.models = tabularDynamicsModels(int(self.numActions/2), self.numStates)
	def updateModel(self, state, action, nextState):
		return self.models.updateTransition(state, int(action/2), nextState)
	def getNextModelState(self, state, action):
		return self.models.getNextState(state, int(action/2))
	def getNextAction(self, state):
		return self.qAgent.epGreedyAction(state)
	def act(self, env, state):
		action = self.getNextAction(state)
		if action % 2 == 0:		# if don't measure
			_, reward, done, _, _ = env.step(action) #move
			next_state = self.models.getNextState(state, int(action/2))#estimate next state
		else:								# act and then measure state
			next_state, reward, done, _, _ = env.step(action)		# move and measure
			if type(next_state) is list:
				next_state = next_state[0]
			self.updateModel(state, action, next_state)
		return state, action, reward, next_state, done
	def greedyAct(self, state):
		return self.qAgent.greedyAction(state)
	def updateQ(self, state, action, reward, nextState):
		self.qAgent.updateQTable(state, action, reward, nextState)


