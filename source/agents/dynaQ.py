import numpy as np

class dynaQ:
	def __init__(self, actionSpaceSize, stateSpaceSize, discount_factor = 0.9, alpha = 0.1, epsilon = 0.1,qInit=0.0):
		self.actionSpaceSize = actionSpaceSize
		self.stateSpaceSize = stateSpaceSize
		self.epsilon = epsilon
		self.discountFactor = discount_factor
		self.alpha = alpha
		self.Q = np.repeat(qInit, actionSpaceSize*stateSpaceSize).reshape(stateSpaceSize,actionSpaceSize)
		self.model = {}

	def epGreedyAction(self, state): 
		ep = np.random.uniform(0,1,1)
		if ep <= self.epsilon:
			action = np.random.choice(np.arange(self.actionSpaceSize))  # pick a random action
		else: 
			action = self.greedyAction(state)
		return action
	def greedyAction(self, state):
		return np.argmax(self.Q[int(state-1)])

	def updateQTable(self, state, action, reward, nextState):
		bestNextAction = np.argmax(self.Q[int(nextState-1)])
		td_target = reward + self.discountFactor * self.Q[int(nextState-1)][bestNextAction]
		td_delta = td_target - self.Q[int(state-1)][action]  
		self.Q[int(state-1)][action] += self.alpha * td_delta 

	def updateModel(self, state, action, reward, nextState):
		# update model
		if state not in self.model.keys():
			self.model[state] = {}
		self.model[state][action] = (reward, nextState)

	def sampleModel(self):
		# randomly choose an state
		rand_idx = np.random.choice(range(len(self.model.keys())))
		state = list(self.model)[rand_idx]
		# randomly choose an action
		rand_idx = np.random.choice(range(len(self.model[state].keys())))
		action = list(self.model[state])[rand_idx]
		reward, nxtState = self.model[state][action]
		return state, action, reward, nxtState
