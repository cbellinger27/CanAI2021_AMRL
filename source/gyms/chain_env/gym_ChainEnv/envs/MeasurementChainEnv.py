import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import gym
import gym.spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


class MeasurementChainEnv(gym.Env):
	metadata = {'render.modes': ['human', 'none']}
	NAMES_ACTIONS = { 'right': 0,'left': 1, 'measure': 2}
	ACTIONS_NAMES = {0:'right', 1:'left', 2:'measure'}

	def __init__(self, xStart=0, xMin=0, xMax=2, transProb=1, renderMode='human'):
		self.xStart = xStart
		self.xState = xStart
		self.xMin = xMin
		self.xMax = xMax
		self.xTerminal = xMax+1
		self.numStates = self.xTerminal+1
		self.transProb = transProb
		self.observation_space = spaces.Discrete(self.xTerminal+1)
		self.measureAction = MeasurementChainEnv.NAMES_ACTIONS['measure']
		self.action_space  = spaces.Discrete(3)
		self.renderMode = renderMode
		# self.p = Plotter(stickyBarriers=self.stickyBarriers,yMax=self.yMax,yMin=self.yMin,xMax=self.xMax,xMin=self.xMin,xDestination=self.xDestination,yDestination=self.yDestination,xStart=self.xStart,yStart=self.yStart)

	def step(self, action):
		done      = False
		info      = {}
		assert action >= 0 and action <= 2, 'Invalid action.'
		observState = [-1.0, -1.0] # [state, confidence]
		if action == 1 and self.xState < self.xMax + 1:
			if self.transProb >= np.random.uniform(0,1,1):
				self.xState += 1
			elif self.xState > self.xMin:
				self.xState -= 1
			reward = self.reward(action)
		elif action == 0 and self.xState > self.xMin:
			if self.transProb >= np.random.uniform(0,1,1):
				self.xState -= 1
			elif self.xState < self.xMax + 1:
				self.xState += 1
			reward = self.reward(action)
		elif action == 2:
			observState[0] = self.xState
			observState[1] = 1.0
			reward = self.cost(action)
		else:
			reward = self.reward(action)
		
		if self.xState == self.xTerminal:
			done = True	
		
		self.render(mode=self.renderMode, isObserveAction=(action==2), action=action)	
		# observed state value and uncertainty of 0
		return observState, reward, done, False, info

	def reset(self, xStart=None):
		self._first_render = True
		if xStart == None:
			self.xState = self.xStart
		else:
			self.xStart = xStart
			self.xState = xStart
		self.render(mode=self.renderMode, isObserveAction=False, isInitial=True, action=None)
		observState = [0.0, 0.0]
		observState[0] = self.xStart
		observState[1] = 1.0
		return observState, {}

	def render(self, mode='human', isObserveAction=False, isInitial=False, action=None):
		if mode == 'human':				
			self.updatePlot(isObserveAction, isInitial, action)

	def updatePlot(self, isObserveAction, isInitial, action=None):
		if isObserveAction:
			print("Action: " + MeasurementChainEnv.ACTIONS_NAMES[action] + ", new state: " + str(self.xState))
		elif isInitial:
			print("Initial state: " + str(self.xState))
		elif self.xState == self.xTerminal:
			print("Reached terminal state. State number " + str(self.xTerminal) + ".")
		else:
			print("Action: Move " + MeasurementChainEnv.ACTIONS_NAMES[action] + ", new state: UNKNOWN")
		

	def close(self):
		self.p.close()

	def reward(self, action):
		if self.xState == self.xTerminal:
			return  1
		else: # self.positionInEnv > self.totalDistance
			return -0.01
	
	def cost(self, action):
		if action == self.measureAction:
			return  -0.05
			# return  -0.1
		else: 
			return 0

