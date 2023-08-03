import numpy as np 
from source.agents.actMeasureTabularQLearningWrp import actMeasureTabularQLearning


class statesActionCounts:
	def __init__(self, numStates, action):
		self.action = action
		self.transitions = np.repeat(0.0, numStates*numStates).reshape(numStates,numStates)
	def getNextState(self, state):
		return np.argmax(self.transitions[state,:])
	def updateTransitions(self, state, nextState):
		self.transitions[state,nextState] += 1

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
		self.models = tabularDynamicsModels(self.numActions, self.numStates)
	def updateModel(self, state, action, nextState):
		return self.models.updateTransition(state, action, nextState)
	def getNextModelState(self, state, action):
		return self.models.getNextState(state, action)
	def getNextAction(self, state):
		return self.qAgent.epGreedyAction(state)
	def act(self, env, state):
		action = self.getNextAction(state)
		if action < self.numActions:		# act and use model to guess next state
			_, reward, done, _, _ = env.step(action)
			next_state = self.models.getNextState(state, action)
		else:								# act and then measure state
			_, reward, _, _, _ = env.step(action-self.numActions)				# act
			next_state, reward2, done, _, _ = env.step(self.numActions)		# measure
			reward += reward2
			next_state = next_state[0]
			# if len(next_state) > 0:
			# 	next_state = next_state[0]
			self.updateModel(state, action-self.numActions, next_state)
		return state, action, reward, next_state, done
	def greedyAct(self, state):
		return self.qAgent.greedyAction(state)
	def updateQ(self, state, action, reward, nextState):
		self.qAgent.updateQTable(state, action, reward, nextState)


