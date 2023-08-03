## Junior Scientist Phase Change Environment

This environment emulates a student learning to manipulate an energy source to produce a desired state change in a target material. Specif- ically, the agent starts with a sealed container of water composed of an initial h0 percent ice, l0 percent water and g0 percent gas (h0 + l0 + g0 = 1). The agent learns to sequen- tially and incrementally adjust a heat source in order to change the ratio of ice, liquid, gas from (h0, l0, g0) to a goal ratio (h, l, g). The episode ends when the agent declares that it has reached the goal and it is correctly in the goal state. The action-space includes A = {decrease, increase, done}, where decrease and increase are fixed incremental adjust- ments in the energy source. The agent receives a reward of r = 1 when it reaches the goal and it correctly declares that it is done, and receives a reward of r = −0.05 at each time step. The agent is charged c = 0.01 for measuring the state of the environment. Measuring the state results in the environment returning the cumulative energy which has been added or removed from the system.

# Installation:

    pip install -e ./source/gyms/junior_scientist/

# Usage:

    import gym
    import gym_TabularPhaseChangeEnv

    env = gym.make("WaterTabularEpisodicWithLandingNoAutoTermination-v0")
    env.reset()
    done = False
    trunc = False
    while not done and not trunc:
        state , reward, done, trunc, info = env.step(env.action_space.sample())
        print("state: " + str(state) + " reward: " + str(reward) + " is done: " + str(done))
