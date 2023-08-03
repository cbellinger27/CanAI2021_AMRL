import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import gym
import gym.spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import time
from matplotlib.collections import PatchCollection
# from Plotter import Plotter

class StandardChainEnv(gym.Env):
	metadata = {'render.modes': ['human', 'none']}
	NAMES_ACTIONS = { 'right': 0,'left': 1}
	ACTIONS_NAMES = {0:'right', 1:'left'}

	def __init__(self, xStart=0, xMin=0, xMax=2, transProb=1, renderMode='human'):
		self.xStart = xStart
		self.xState = xStart
		self.xMin = xMin
		self.xMax = xMax
		self.xTerminal = xMax+1
		self.transProb = transProb
		self.observation_space = spaces.Discrete(self.xTerminal+1)
		self.numStates = self.xTerminal+1
		self.action_space  = spaces.Discrete(2)
		self.renderMode = renderMode
		# self.p = Plotter(stickyBarriers=self.stickyBarriers,yMax=self.yMax,yMin=self.yMin,xMax=self.xMax,xMin=self.xMin,xDestination=self.xDestination,yDestination=self.yDestination,xStart=self.xStart,yStart=self.yStart)

	def step(self, action):
		done      = False
		info      = {}
		assert action >= 0 and action <= 2, 'Invalid action.'
		observState = -1
		reward = 0
		if action == 1 and self.xState < self.xMax + 1:
			if self.transProb >= np.random.uniform(0,1,1):
				self.xState += 1
			elif self.xState > self.xMin:
				self.xState -= 1
			observState = self.xState
		elif action == 0 and self.xState > self.xMin:
			if self.transProb >= np.random.uniform(0,1,1):
				self.xState -= 1
			elif self.xState < self.xMax + 1:
				self.xState += 1
			observState = self.xState
	
		reward = self.reward(action)
		
		if self.xState == self.xTerminal:
			done = True
		
		self.render(mode=self.renderMode, isInitial=False, action=action)	
	
		return observState, reward, done, False, info

	def reset(self, xStart=None):
		self._first_render = True
		if xStart == None:
			self.xState = self.xStart
		else:
			self.xStart = xStart
			self.xState = xStart
		self.render(mode=self.renderMode, isInitial=True, action=None)
		return self.xState, {}

	def render(self, mode='human', isInitial=False, action=None):
		if mode == 'human':				
			self.updatePlot(isInitial=isInitial, action=action)

	def updatePlot(self, isInitial, action):
		if isInitial:
			print("Initial state: " + str(self.xState))
		else:
			print("Action: Move " + StandardChainEnv.ACTIONS_NAMES[action] + ", new state: " + str(self.xState))


	def reward(self, action):
		if self.xState == self.xTerminal:
			return  1
		else: # self.positionInEnv > self.totalDistance
			return  -0.01

