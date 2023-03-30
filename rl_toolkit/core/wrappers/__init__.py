from gymnasium.envs.registration import register

from .dm_control import dmControlGetTasks, dmControlGymWrapper  # noqa

register(
    id="HumanoidRobot-v0",
    entry_point="rl_toolkit.core.wrappers.humanoid:HumanoidRobot",
    max_episode_steps=1000,
    reward_threshold=1.0,
)
