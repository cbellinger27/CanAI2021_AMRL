import numpy as np 
from collections import defaultdict 

class actMeasureTabularQLearning:
	def __init__(self, actionSpaceSize, stateSpaceSize, measureBias, qInit=0.0, discount_factor = 0.9, alpha = 0.1, epsilon = 0.1):
		self.actionSpaceSize = actionSpaceSize
		self.stateSpaceSize = stateSpaceSize
		self.epsilon = epsilon
		self.discountFactor = discount_factor
		self.alpha = alpha
		self.Q = np.repeat(qInit, actionSpaceSize*stateSpaceSize).reshape(stateSpaceSize,actionSpaceSize)	
		self.Q[:,np.arange(1,actionSpaceSize,2)] = np.repeat(measureBias,stateSpaceSize).reshape(stateSpaceSize,1)
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

