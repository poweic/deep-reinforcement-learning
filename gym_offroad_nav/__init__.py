from gym.envs.registration import register

register(
    id='OffRoadNav-v0',
    entry_point='gym_offroad_nav.envs:OffRoadNavEnv',
    timestep_limit=1000,
)
