from gym.envs.registration import register

register(
    id='offroad-nav-v0',
    entry_point='gym_offroad_nav.envs:OffRoadNavEnv',
    timestep_limit=1000,
)
register(
    id='offroad-nav-extrahard-v0',
    entry_point='gym_offroad_nav.envs:OffRoadNavExtraHardEnv',
    timestep_limit=1000,
)