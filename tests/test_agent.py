from rl_toolkit.core import Tester


def test_pre_trained():
    agent = Tester(
        env_name="BipedalWalkerHardcore-v3",
        render=False,
        max_steps=2000,
        actor_units=[512, 256],
        clip_mean_min=-2.0,
        clip_mean_max=2.0,
        init_noise=-3.0,
        model_path="models/BipedalWalkerHardcore-v3.h5",
        enable_wandb=False,
    )

    try:
        agent.run()
    finally:
        agent.close()


def test_random():
    agent = Tester(
        env_name="BipedalWalkerHardcore-v3",
        render=False,
        max_steps=2000,
        actor_units=[512, 256],
        clip_mean_min=-2.0,
        clip_mean_max=2.0,
        init_noise=-3.0,
        model_path=None,
        enable_wandb=False,
    )

    try:
        agent.run()
    finally:
        agent.close()
