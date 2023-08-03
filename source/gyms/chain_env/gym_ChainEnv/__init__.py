from gym.envs.registration import register

register(
	id='StandardChainEnv-v0',
	 max_episode_steps=250,
	entry_point='gym_ChainEnv.envs:StandardChainEnv'
)
register(
	id='MeasurementChainEnv-v0',
	 max_episode_steps=250,
	entry_point='gym_ChainEnv.envs:MeasurementChainEnv'
)
register(
	id='MeasurementEvenOddChainEnv-v0',
	 max_episode_steps=1000,
	entry_point='gym_ChainEnv.envs:MeasurementEvenOddChainEnv'
)
register(
	id='MeasurementGausEvenOddChainEnv-v0',
	 max_episode_steps=1000,
	entry_point='gym_ChainEnv.envs:MeasurementGausEvenOddChainEnv'
)
register(
	id='StandardGausEvenOddChainEnv-v0',
	 max_episode_steps=1000,
	entry_point='gym_ChainEnv.envs:StandardGausEvenOddChainEnv'
)
