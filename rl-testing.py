import gym
import pybullet_envs
import wandb
import argparse

import tensorflow as tf

from policy.sac import Actor as ActorSAC
from policy.td3 import Actor as ActorTD3


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(
        prog="python3 rl-testing.py",
        description="RL toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # init args
    my_parser.add_argument(
        "-alg",
        "--algorithm",
        type=str,
        help="Select the algorithm, that making a decisions",
        default="sac",
    )
    my_parser.add_argument(
        "-env",
        "--environment",
        type=str,
        help="Only OpenAI Gym & PyBullet environments are available!",
        default="BipedalWalker-v3",
    )
    my_parser.add_argument(
        "-db", "--database", type=str, help="Database name", default="rl-agents"
    )
    my_parser.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Maximum number of interactions doing in environment",
        default=int(1e6),
    )
    my_parser.add_argument("--wandb", action="store_true", help="Logging to wanDB")
    my_parser.add_argument("-f", "--model", type=str, help="Actor's model file")

    # nacitaj zadane argumenty programu
    args = my_parser.parse_args()

    # Herne prostredie
    env = gym.make(args.environment)
    # env.render()

    # inicializuj prostredie Weights & Biases
    if args.wandb == True:
        wandb.init(project="rl-toolkit")

    # load actor model
    if args.algorithm == "td3":
        actor = ActorTD3(model_path=args.model)
    elif args.algorithm == "sac":
        actor = ActorSAC(model_path=args.model)
    else:
        raise NameError(f"algorithm '{args.algorithm}' is not defined")

    # hlavny cyklus hry
    total_steps, total_episodes = 0, 0
    while total_steps < args.max_steps:
        done = False
        episode_reward, episode_timesteps = 0.0, 0

        obs = env.reset()

        # collect rollout
        while not done:
            # env.render()

            if args.algorithm == "sac":
                action, _ = actor.predict(
                    tf.expand_dims(obs, axis=0), with_logprob=False, deterministic=True
                )
            elif args.algorithm == "td3":
                action = actor.model(tf.expand_dims(obs, axis=0))

            # perform action
            new_obs, reward, done, _ = env.step(action[0])

            episode_reward += reward
            episode_timesteps += 1
            total_steps += 1

            # super critical !!!
            obs = new_obs

        # after each episode
        total_episodes += 1

        print(f"Epoch: {total_episodes}")
        print(f"EpsReward: {episode_reward}")
        print(f"EpsSteps: {episode_timesteps}")
        print(f"TotalInteractions: {total_steps}")
        if args.wandb == True:
            wandb.log(
                {
                    "epoch": total_episodes,
                    "score": episode_reward,
                    "steps": episode_timesteps,
                }
            )

    env.close()
