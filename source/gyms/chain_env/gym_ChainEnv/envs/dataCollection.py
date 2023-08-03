#Class to collect random episodes of open AI gym data for evaluating state dynamics models

import gym
import numpy as np
import re
import pickle

class Policy():
    def __init__(self, policy=None, policyName="Random"):
        self.name=policyName
        if policy is None:
            self.apply = self.randomPolicy
        else:
            self.apply = policy
    def randomPolicy(self, state, env):
        return(env.action_space.sample())

class DataCollector():
    def __init__(self, env):
        self.env = env
    def collect_data(self, policy, numEp):
        "data: ep number, policy name, state, action, next state, reward, done"
        data = []
        state = env.reset()
        ep = 0
        while ep < numEp:
            stpData = [ep+1, policy.name, state]
            action = policy.apply(state, self.env)
            stpData.append(action)
            nextState, reward, done, info = self.env.step(action)
            stpData = stpData+[nextState, reward, done]
            state = nextState
            data.append(stpData)
            stpData = []
            if done:
                ep += 1
                state = env.reset()
        return data
    
    def save_data(fileName, data, dataStr="epNumber policyName, state, action, nextState, reward, done"):
        dataStr = [dataStr]
        dataStr.append(data)
        pickle_out = open(fileName,"wb")
        pickle.dump(dataStr, pickle_out)
        pickle_out.close()
    
    def open_data(fileName):
        pickle_in = open(fileName,"rb")
        dataLst = pickle.load(pickle_in)
        return dataLst[0], dataLst[1]
    def returnSAS(data):
        return np.vstack(data)[:,[2,3,4]].astype(float)
    def returnSASInEpRange(data, lower, upper):
        toReturn = np.vstack(data)[:,[0,2,3,4]].astype(float)
        idx = np.intersect1d(np.where(toReturn[:,0]>=lower)[0],np.where(toReturn[:,0]<=upper)[0])
        return toReturn[idx,1:]


class Go1Policy():
	def policy(state, env):
		return 1



env = gym.make('FrozenLake-v0')