## Markov Chain Environment

We evaluate an 11-state, standard chain environment and a 20- state even-odd chain environment. In both cases, the episodes start in s0 and they conclude in goal states s10 and s19 respectively. Upon entering goal state, the agent receives a reward of r = 1 in the standard chain and r = 300 in the even-odd chain. In all other states, the agent receives a reward of r = −0.01 in the standard chain and r = −1 in the even-odd chain. The agent selects from action pairs: (move left, don’t measure) , (move left, measure), (move right, don’t measure), (move right, measure). In the standard chain, measuring the next state has a cost of c = 0.05. Alternatively, we analyze the impact of measurement costs between 0.1 and 25 in the even-odd chain. In odd numbered states of the even-odd chain, the left and right actions are reversed to move the agent in the opposite direction. In general, state transitions include Gaussian additive noise: st+1 ∼ P(st+1|st,at) + round(N(σ)), where σ determines the extent of stochaticity.

# Installation:

    pip install -e ./source/gyms/chain_env/

# Usage:

    import gym
    import gym_ChainEnv

    n = 0.1
    env = gym.make("StandardGausEvenOddChainEnv-v0", noise=n)
    env.reset()
    done = False
    trunc = False
    while not done and not trunc:
        a = env.action_space.sample()
        state , reward, done, trunc, info = env.step(a)
