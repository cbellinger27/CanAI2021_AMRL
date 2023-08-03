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
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class StandardGausEvenOddChainEnv(gym.Env):
    metadata = {'render.modes': ['human', 'none']}
    NAMES_ACTIONS = {'action zero and measure': 0, 'action one and measure':1}
    ACTIONS_NAMES = {0:'action zero and measure', 1:'action one and measure'}
    
    def __init__(self, xStart=1, xMin=1, xMax=20, noise=0, renderMode='human'):
        assert xStart > 0 and xMin > 0, 'Start and minimum state should be greater than 0.'
        self.xStart = xStart
        self.xState = xStart
        self.xMin = xMin
        self.xMax = xMax
        self.xTerminal = xMax
        self.numStates = self.xTerminal+1
        self.noise = noise
        self.observation_space = spaces.Discrete(self.xTerminal)
        self.action_space  = spaces.Discrete(2)
        self.renderMode = renderMode
        # self.p = Plotter(stickyBarriers=self.stickyBarriers,yMax=self.yMax,yMin=self.yMin,xMax=self.xMax,xMin=self.xMin,xDestination=self.xDestination,yDestination=self.yDestination,xStart=self.xStart,yStart=self.yStart)
    
    def step(self, action):
        done      = False
        info      = {} 
        assert action >= 0 and action <= 1, 'Invalid action.'
        assert self.xState <= self.xTerminal, 'Episode already complete.'
        observState = [-1.0, -1.0] # [state, confidence]
        prob = np.repeat(self.noise/((self.xMax - self.xMin)), (self.xMax - self.xMin)+1)
        if self.xState % 2 == 0:
            if action == 1: #increases state by one
                stateValue = int(np.round(get_truncated_normal(mean=self.xState+1, sd=self.noise+0.001, low=self.xMin, upp=self.xMax).rvs()))
            elif action == 0: #decrease state by one
                stateValue = int(np.round(get_truncated_normal(mean=self.xState-1, sd=self.noise+0.001, low=self.xMin, upp=self.xMax).rvs()))
        else:
            if action == 0: #increases state by one
                stateValue = int(np.round(get_truncated_normal(mean=self.xState+1, sd=self.noise+0.001, low=self.xMin, upp=self.xMax).rvs()))
            elif action == 1: #decrease state by one
                stateValue = int(np.round(get_truncated_normal(mean=self.xState-1, sd=self.noise+0.001, low=self.xMin, upp=self.xMax).rvs()))
        self.xState = stateValue
        reward = self.reward(action)
        observState[0] = self.xState
        observState[1] = 1.0
        if self.xState == self.xTerminal:
            done = True
        
        self.render(mode=self.renderMode,action=action)	
        # observed state value and uncertainty of 0
        return observState, reward, done, False, info
    
    def reset(self, xStart=None):
        if xStart == None:
            self.xState = self.xStart
        else:
            self.xStart = xStart
            self.xState = xStart
        observState = [0.0, 0.0]
        observState[0] = self.xStart
        observState[1] = 1.0
        self.render(mode=self.renderMode)
        return observState, {}
    
    def render(self, mode='human', action=None):
        if mode == 'human':				
            self.updatePlot(action)
            # self.state_render(mode='plot')
    
    def updatePlot(self, action=None):
        if action is None:
            print("Starting new chain with minimum value " + str(self.xMin) + " and maximum value " + str(self.xMax))
        else:
            print("Action: " + StandardGausEvenOddChainEnv.ACTIONS_NAMES[action] + " resulting in state: " + str(self.xState))
    
    def close(self):
        print("closing...")

    
    def reward(self, action):
        if self.xState == self.xTerminal:
            return  300
        else: # self.positionInEnv > self.totalDistance
            return -1


