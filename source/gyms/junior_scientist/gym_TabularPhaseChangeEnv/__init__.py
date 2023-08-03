from gym.envs.registration import register

register(
    id = 'WaterTable-v0',
    entry_point = 'gym_TabularPhaseChangeEnv.envs.WaterTabular:WaterEnvTab',
    max_episode_steps = 500,
)
register(
    id = 'WaterTableEpisodic-v0',
    entry_point = 'gym_TabularPhaseChangeEnv.envs.WaterTabularEpisodic:WaterEnvTabEpisodic',
    # max_episode_steps = 500,
)
register(
    id = 'WaterTabularEpisodicWithLanding-v0',
    entry_point = 'gym_TabularPhaseChangeEnv.envs.WaterTabularEpisodicWithLanding:WaterTabularEpisodicWithLanding',
    # max_episode_steps = 500,
)
register(
    id = 'WaterTabularEpisodicWithLandingNoAutoTermination-v0',
    entry_point = 'gym_TabularPhaseChangeEnv.envs.WaterTabularEpisodicWithLandingNoAutoTermination:WaterTabularEpisodicWithLandingNoAutoTermination',
    # max_episode_steps = 500,
)
